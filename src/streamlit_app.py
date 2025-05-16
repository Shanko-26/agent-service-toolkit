import asyncio
import os
import urllib.parse
import uuid
import traceback
from collections.abc import AsyncGenerator
from datetime import datetime
import httpx

import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError
import pandas as pd

from client import AgentClient, AgentClientError, FileClient, FileClientError
from schema import ChatHistory, ChatMessage
from schema.task_data import TaskData, TaskDataStatus
from schema.automotive.models import MeasurementFile
from visualization.plot import (
    init_plot_state,
    update_plot_config, 
    handle_plot_request,
    handle_llm_tool_call,
    render_time_controls,
    prepare_plot_data,
    render_plot,
    download_data_as_csv
)

# A Streamlit app for interacting with the langgraph agent via a simple chat interface.
# The app has three main functions which are all run async:

# - main() - sets up the streamlit app and high level structure
# - draw_messages() - draws a set of chat messages - either replaying existing messages
#   or streaming new ones.
# - handle_feedback() - Draws a feedback widget and records feedback from the user.

# The app heavily uses AgentClient to interact with the agent's FastAPI endpoints.


APP_TITLE = "Automotive Data Copilot"
APP_ICON = "🚗"


async def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        menu_items={},
        layout="wide",
    )

    # --- Compact CSS for font and padding ---
    st.markdown(
        """
        <style>
        html, body {
            margin: 0 !important;
            padding: 0 !important;
            box-sizing: border-box !important;
        }
        [data-testid="stAppViewContainer"] {
            padding: 0 !important;
            margin: 0 !important;
            box-sizing: border-box !important;
        }
        [data-testid="stAppViewContainer"] > .main {
            padding: 0 !important;
            margin: 0 !important;
            box-sizing: border-box !important;
        }
        [data-testid="stAppViewContainer"] > .main .block-container {
            padding: 0 !important;
            margin: 0 !important;
            box-sizing: border-box !important;
        }
        /* Minimal gutter for content (optional, adjust as needed) */
        [data-testid="stAppViewContainer"] > .main .block-container {
            padding-left: 0rem !important;
            padding-right: 0rem !important;
        }
        /* --- Typography and UI tweaks --- */
        h1, h2, h3, h4, h5, h6, .stHeader, .stSubheader {
            font-size: 1.1em !important;
            margin-bottom: 0.15em !important;
            margin-top: 0.1em !important;
            padding-top: 0.1em !important;
            padding-bottom: 0.1em !important;
        }
        .stButton>button, .stTextInput>div>input, .stTextArea>div>textarea, .stSelectbox>div>div {
            font-size: 0.9em !important;
            padding: 0.2em 0.1em !important;
        }
        .stChatMessage, .stMarkdown, .stDataFrame, .stTable, .stMetric {
            font-size: 0.97em !important;
        }
        .stForm, .stFormContainer, .stExpander, .stPopover {
            padding: 0.3em 0.5em !important;
        }
        .stTabs, .stRadio, .stSelectbox, .stMultiselect {
            margin-bottom: 0.2em !important;
        }
        /* --- Shrink column gap and internal column padding --- */
        .css-1cpxqw2 { gap: 0.4rem !important; }
        [data-testid="column"] { padding-left: 0rem !important; padding-right: 0rem !important; }
        /* --- Chat panel tweaks remain unchanged --- */
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Hide the streamlit upper-right chrome
    st.html(
        """
        <style>
        [data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
            }
        </style>
        """,
    )
    if st.get_option("client.toolbarMode") != "minimal":
        st.set_option("client.toolbarMode", "minimal")
        await asyncio.sleep(0.1)
        st.rerun()

    # Initialize clients
    if "agent_client" not in st.session_state:
        load_dotenv()
        agent_url = os.getenv("AGENT_URL")
        if not agent_url:
            host = os.getenv("HOST", "127.0.0.1")
            port = os.getenv("PORT", 8080)
            agent_url = f"http://{host}:{port}"
        try:
            with st.spinner("Connecting to agent service..."):
                os.environ["AUTH_SECRET"] = "development_secret_123"
                st.session_state.agent_client = AgentClient(base_url=agent_url)
                try:
                    st.session_state.agent_client.update_agent("automotive")
                except AgentClientError as e:
                    st.warning(f"Could not set default agent to 'automotive': {e}. Using service default.")
                st.session_state.file_client = FileClient(base_url=agent_url)
        except (AgentClientError, FileClientError) as e:
            st.error(f"Error connecting to service at {agent_url}: {e}")
            st.markdown("The service might be booting up. Try again in a few seconds.")
            st.stop()
    agent_client: AgentClient = st.session_state.agent_client
    file_client: FileClient = st.session_state.file_client

    # Initialize session state
    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = str(uuid.uuid4())
            messages = []
        else:
            try:
                messages: ChatHistory = agent_client.get_history(thread_id=thread_id).messages
            except AgentClientError:
                st.error("No message history found for this Thread ID.")
                messages = []
        st.session_state.messages = messages
        st.session_state.thread_id = thread_id
    
    # Initialize file selection state
    if "selected_file_id" not in st.session_state:
        st.session_state.selected_file_id = None
    
    if "files" not in st.session_state:
        try:
            st.session_state.files = await file_client.list_files()
            st.toast(f"Loaded {len(st.session_state.files)} files on startup")
        except FileClientError:
            st.session_state.files = []
            st.toast("No files found or error loading files")
    
    # Initialize file metadata state
    if "selected_file_metadata" not in st.session_state:
        st.session_state.selected_file_metadata = None
    
    # Initialize file context for chat history
    if "file_context" not in st.session_state:
        st.session_state.file_context = {}
    
    # Initialize plot state
    init_plot_state()
    
    # If a file is selected, fetch its metadata
    if st.session_state.selected_file_id and (
        st.session_state.selected_file_metadata is None or 
        st.session_state.selected_file_metadata.file_id != st.session_state.selected_file_id
    ):
        try:
            with st.spinner("Loading file metadata..."):
                # Debug info
                st.toast(f"Fetching metadata for file ID: {st.session_state.selected_file_id}")
                metadata = await file_client.get_file_metadata(st.session_state.selected_file_id)
                st.session_state.selected_file_metadata = metadata
                
                # Update file context when a new file is selected
                st.session_state.file_context[st.session_state.selected_file_id] = {
                    "filename": metadata.filename,
                    "channel_count": len(metadata.channels) if metadata.channels else 0,
                    "duration": metadata.duration,
                    "start_time": metadata.start_time,
                    "end_time": metadata.end_time,
                    "channels": list(metadata.channels.keys()) if metadata.channels else [],
                    "last_accessed": datetime.now().isoformat(),
                }
                
                st.toast("Metadata loaded successfully!")
        except FileClientError as e:
            st.error(f"Error loading file metadata: {e}")
            st.session_state.selected_file_metadata = None

    # Add collapse/expand state for left and right panels
    if "collapse_left" not in st.session_state:
        st.session_state.collapse_left = False
    if "collapse_right" not in st.session_state:
        st.session_state.collapse_right = False

    # Set column widths based on collapse state
    left_width = 0.07 if st.session_state.collapse_left else 1
    right_width = 0.07 if st.session_state.collapse_right else 1
    center_width = 6 if st.session_state.collapse_left or st.session_state.collapse_right else 2
    left, center, right = st.columns([left_width, center_width, right_width], gap="large")

    # --- LEFT COLUMN: Controls, file upload, settings ---
    with left:
        if st.session_state.collapse_left:
            if st.button("➡️", key="expand_left_btn", help="Expand sidebar", use_container_width=True):
                st.session_state.collapse_left = False
                st.rerun()
            st.write("")
        else:
            if st.button("⬅️", key="collapse_left_btn", help="Collapse sidebar", use_container_width=True):
                st.session_state.collapse_left = True
                st.rerun()
            st.header(f"{APP_ICON} {APP_TITLE}")
            ""
            "Interact with automotive measurement data using natural language"
            ""
            if st.button(":material/chat: New Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.thread_id = str(uuid.uuid4())
                st.rerun()
            st.subheader("📁 Data Files")
            with st.form("file_upload_form", clear_on_submit=True):
                uploaded_file = st.file_uploader(
                    "Upload MDF File",
                    type=["mf4", "mdf", "dat"],
                    help="Upload automotive measurement data files (MDF3 or MDF4 format)"
                )
                file_description = st.text_area("Description (optional)", height=68)
                submit_button = st.form_submit_button("Upload")
                if submit_button and uploaded_file is not None:
                    try:
                        with st.spinner("Uploading and processing file..."):
                            file_content = uploaded_file.getvalue()
                            filename = uploaded_file.name
                            response = await file_client.upload_file(
                                file_content=file_content,
                                filename=filename,
                                description=file_description if file_description else None
                            )
                            if response.status == "ok":
                                st.success(f"Successfully uploaded: {filename}")
                                st.session_state.files = await file_client.list_files()
                                st.session_state.selected_file_id = response.file_id
                                st.session_state.selected_file_metadata = None
                                st.toast(f"File list updated, found {len(st.session_state.files)} files")
                            else:
                                st.warning(f"Upload issue: {response.message}")
                    except FileClientError as e:
                        st.error(f"Error uploading file: {e}")
            if len(st.session_state.files) > 0:
                st.write("Available files:")
                for file in st.session_state.files:
                    col1, col2 = st.columns([4, 1])
                    is_selected = st.session_state.selected_file_id == file.file_id
                    with col1:
                        if st.button(
                            f"{'📌 ' if is_selected else '📄 '}{file.filename}",
                            key=f"file_{file.file_id}",
                            help=f"{file.description or 'No description'}\nChannels: {file.channel_count}",
                            use_container_width=True,
                        ):
                            st.session_state.selected_file_id = file.file_id
                            st.session_state.selected_file_metadata = None
                            st.toast(f"Selected file: {file.filename}")
                            st.rerun()
                    with col2:
                        if st.button(
                            "🗑️",
                            key=f"delete_{file.file_id}",
                            help=f"Delete {file.filename}",
                        ):
                            try:
                                with st.spinner("Deleting file..."):
                                    await file_client.delete_file(file.file_id)
                                    st.session_state.files = await file_client.list_files()
                                    if st.session_state.selected_file_id == file.file_id:
                                        st.session_state.selected_file_id = None
                                        if file.file_id in st.session_state.file_context:
                                            del st.session_state.file_context[file.file_id]
                                    st.success(f"Deleted {file.filename}")
                                    st.rerun()
                            except FileClientError as e:
                                st.error(f"Error deleting file: {e}")
                if st.button("🔄 Refresh Files"):
                    try:
                        with st.spinner("Refreshing files..."):
                            st.session_state.files = await file_client.list_files()
                            st.success("Files refreshed")
                            st.rerun()
                    except FileClientError as e:
                        st.error(f"Error refreshing files: {e}")
            else:
                st.info("No files uploaded yet. Upload an MDF file to get started.")
            with st.popover(":material/settings: Settings", use_container_width=True):
                model_idx = agent_client.info.models.index(agent_client.info.default_model)
                model = st.selectbox("LLM to use", options=agent_client.info.models, index=model_idx)
                agent_list = [a.key for a in agent_client.info.agents]
                default_agent = "automotive"
                agent_idx = agent_list.index(default_agent) if default_agent in agent_list else agent_list.index(agent_client.info.default_agent)
                agent_client.agent = st.selectbox(
                    "Agent to use",
                    options=agent_list,
                    index=agent_idx,
                )
                use_streaming = st.toggle("Stream results", value=True)
            @st.dialog("Architecture")
            def architecture_dialog() -> None:
                st.image(
                    "https://github.com/JoshuaC215/agent-service-toolkit/blob/main/media/agent_architecture.png?raw=true"
                )
                "[View full size on Github](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/media/agent_architecture.png)"
                st.caption(
                    "App hosted on [Streamlit Cloud](https://share.streamlit.io/) with FastAPI service running in [Azure](https://learn.microsoft.com/en-us/azure/app-service/)"
                )
            if st.button(":material/schema: Architecture", use_container_width=True):
                architecture_dialog()
            with st.popover(":material/policy: Privacy", use_container_width=True):
                st.write(
                    "Prompts, responses and feedback in this app are anonymously recorded and saved to LangSmith for product evaluation and improvement purposes only."
                )
            @st.dialog("Share/resume chat")
            def share_chat_dialog() -> None:
                session = st.runtime.get_instance()._session_mgr.list_active_sessions()[0]
                st_base_url = urllib.parse.urlunparse(
                    [session.client.request.protocol, session.client.request.host, "", "", "", ""]
                )
                if not st_base_url.startswith("https") and "localhost" not in st_base_url:
                    st_base_url = st_base_url.replace("http", "https")
                chat_url = f"{st_base_url}?thread_id={st.session_state.thread_id}"
                st.markdown(f"**Chat URL:**\n```text\n{chat_url}\n```")
                st.info("Copy the above URL to share or revisit this chat")
            if st.button(":material/upload: Share/resume chat", use_container_width=True):
                share_chat_dialog()
            "[View the source code](https://github.com/JoshuaC215/agent-service-toolkit)"
            st.caption("Based on agent-service-toolkit by Joshua")

    # --- CENTER COLUMN: Plot area placeholder and file metadata ---
    with center:
        st.header("Plot Area")
        
        # Add a help expander to explain how to use the plot
        with st.expander("ℹ️ Plot Help", expanded=False):
            st.markdown("""
            ### How to use the plotting feature:
            
            1. **Select signals** to plot from the Signal Browser tab
            2. Click **Plot Selected Signals** to visualize the data
            3. **Interact with the plot**:
               - 📱 **Hover** over the lines to see exact values
               - 🖱️ **Click and drag** to zoom into a specific area
               - 📊 Use the **toolbar** in the upper right for more options
               - 📈 Use the **rangeslider** at the bottom to quickly navigate and select time ranges
            4. **Download the plotted data** as CSV for further analysis
            
            *Tip: You can also ask the AI in the chat panel to plot specific signals for you!*
            """)
        
        # Check if we have a selected file and plot configuration
        if st.session_state.selected_file_id:
            # No need for separate time range controls - Plotly provides this functionality
            
            # Check if we have channels selected for plotting
            if st.session_state.get("plot_config", {}).get("channels"):
                # Create placeholder for plot visualization
                plot_placeholder = st.empty()
                
                # Define a synchronous function to handle plotting
                def plot_signals_sync():
                    try:
                        # Get the configuration
                        config = st.session_state.get("plot_config", {})
                        channels = config.get("channels", [])
                        
                        if not channels:
                            st.info("No channels selected for plotting.")
                            return
                        
                        # Get client and file_id
                        file_client = st.session_state.file_client
                        file_id = st.session_state.selected_file_id
                        
                        if not file_id:
                            st.warning("No file selected. Please select a file first.")
                            return
                        
                        # Use a single spinner instead of multiple status updates
                        with st.spinner(f"Loading and plotting {len(channels)} signals..."):
                            # Get parameters for the data request
                            params = {
                                "channels": ",".join(channels)
                            }
                            if config.get("start_time") is not None:
                                params["start_time"] = config.get("start_time")
                            if config.get("end_time") is not None:
                                params["end_time"] = config.get("end_time")
                            
                            # Make HTTP request
                            try:
                                with httpx.Client() as client:
                                    response = client.get(
                                        f"{file_client.base_url}/files/{file_id}/data",
                                        params=params,
                                        headers=file_client._headers,
                                        timeout=30.0  # Increased timeout for large datasets
                                    )
                                    response.raise_for_status()
                                    signal_data = response.json()
                            except httpx.HTTPStatusError as e:
                                if e.response.status_code == 404:
                                    st.error(f"The data endpoint returned 404. Please make sure the backend API is running and the file exists.")
                                elif e.response.status_code == 400:
                                    st.error(f"Bad request: {e.response.text}")
                                else:
                                    st.error(f"HTTP error: {e.response.status_code} - {e.response.reason_phrase}")
                                return
                            except httpx.RequestError as e:
                                st.error(f"Request error: {str(e)}")
                                return
                            
                            # Check if we got valid data
                            if not signal_data:
                                st.error("No data returned from the server. The channels might not exist or contain no data.")
                                return
                            
                            # Process data
                            plot_data = prepare_plot_data(signal_data)
                            
                            if plot_data.empty:
                                st.warning("No data available for the selected channels in the specified time range.")
                                return
                            
                            # Render the plot
                            render_plot(plot_data, config)
                        
                        # Add download button for the data
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.success(f"📊 Plotted {len(channels)} channels with {len(plot_data)} data points")
                        with col2:
                            # Generate a meaningful filename based on selected channels
                            channels_str = "_".join(channels[:2])
                            if len(channels) > 2:
                                channels_str += f"_and_{len(channels)-2}_more"
                            filename = f"{channels_str}_data.csv"
                            download_data_as_csv(plot_data, filename)
                            
                    except Exception as e:
                        st.error(f"Error plotting data: {str(e)}")
                        with st.expander("Error Details", expanded=True):
                            st.code(traceback.format_exc())
                
                # Call the synchronous function directly
                plot_signals_sync()
            else:
                st.info("Select channels to plot data from the Signal Browser tab.")
        else:
            st.info("Select a file to plot data.")
        
        if st.session_state.selected_file_metadata:
            metadata = st.session_state.selected_file_metadata
            tab_names = ["📊 File Metadata", "📈 Signal Browser"]
            if "selected_tab" not in st.session_state:
                st.session_state.selected_tab = tab_names[0]
            selected_tab = st.radio("Select view", tab_names, key="selected_tab")
            if st.session_state.selected_tab == "📊 File Metadata":
                st.subheader(f"File: {metadata.filename}")
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("File Information")
                    st.write(f"**File ID:** {metadata.file_id}")
                    st.write(f"**Format:** {metadata.file_type}")
                    st.write(f"**Size:** {metadata.file_size_bytes / (1024*1024):.2f} MB")
                    st.write(f"**Description:** {metadata.description or 'N/A'}")
                with col2:
                    st.subheader("Time Information")
                    start_time = metadata.start_time
                    end_time = metadata.end_time
                    if isinstance(start_time, datetime):
                        start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        start_time_str = f"{start_time:.2f} s"
                    if isinstance(end_time, datetime):
                        end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        end_time_str = f"{end_time:.2f} s"
                    st.write(f"**Start Time:** {start_time_str}")
                    st.write(f"**End Time:** {end_time_str}")
                    st.write(f"**Duration:** {metadata.duration:.2f} s")
                    st.write(f"**Sample Count:** {metadata.sample_count:,}")
                st.subheader("Channel Overview")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Channels", len(metadata.channels) if metadata.channels else 0)
                    if metadata.channels:
                        ecu_counts = {}
                        for channel_id, channel in metadata.channels.items():
                            ecu = channel.ecu or "Unknown"
                            ecu_counts[ecu] = ecu_counts.get(ecu, 0) + 1
                        st.subheader("ECUs")
                        for ecu, count in sorted(ecu_counts.items(), key=lambda x: x[1], reverse=True):
                            st.write(f"**{ecu}**: {count} channels")
                with col2:
                    if metadata.channels:
                        sample_channels = list(metadata.channels.keys())[:5]
                        st.subheader("Sample Channels")
                        for channel in sample_channels:
                            st.write(f"- {channel}")
                        if len(metadata.channels) > 5:
                            st.write(f"*...and {len(metadata.channels) - 5} more*")
                    st.info("👉 Use the **Signal Browser** tab to explore and select channels for plotting")
            elif st.session_state.selected_tab == "📈 Signal Browser":
                st.subheader("Select Signals to Plot")
                # Prepare channel data for the signal browser
                channels_data = []
                if metadata.channels:
                    for channel_id, channel in metadata.channels.items():
                        channels_data.append({
                            "Name": channel.name,
                            "Unit": channel.unit or "N/A",
                            "Min": channel.min_value if channel.min_value is not None else "N/A",
                            "Max": channel.max_value if channel.max_value is not None else "N/A",
                            "Sampling Rate": f"{channel.sampling_rate:.2f} Hz" if channel.sampling_rate else "N/A",
                            "Description": channel.description or "N/A",
                            "ECU": channel.ecu or "N/A"
                        })
                if "selected_channels" not in st.session_state:
                    st.session_state.selected_channels = []
                col1, col2 = st.columns([2, 1])
                with col1:
                    signal_search = st.text_input(
                        "Search signals", 
                        placeholder="Enter signal name or description...",
                        key="signal_search_customtab"
                    )
                with col2:
                    all_ecus = sorted(list(set([
                        channel["ECU"] for channel in channels_data 
                        if channel["ECU"] != "N/A"
                    ])))
                    selected_ecu = st.selectbox(
                        "Filter by ECU", 
                        options=["All ECUs"] + all_ecus,
                        key="ecu_filter_customtab"
                    )
                filtered_channels = channels_data
                if signal_search:
                    filtered_channels = [
                        channel for channel in filtered_channels
                        if signal_search.lower() in channel["Name"].lower() or 
                           signal_search.lower() in (channel["Description"] or "").lower()
                    ]
                if selected_ecu != "All ECUs":
                    filtered_channels = [
                        channel for channel in filtered_channels
                        if channel["ECU"] == selected_ecu
                    ]
                st.write(f"Showing {len(filtered_channels)} of {len(channels_data)} signals")
                signal_options = [channel["Name"] for channel in filtered_channels]
                st.multiselect(
                    "Select signals to plot",
                    options=signal_options,
                    key="selected_channels"
                )
                if st.session_state.selected_channels:
                    st.write(f"Selected {len(st.session_state.selected_channels)} signals")
                    
                    # Create plot button with loading state
                    plot_button = st.button(
                        "📊 Plot Selected Signals", 
                        key="plot_selected_customtab", 
                        type="primary", 
                        use_container_width=True,
                        help="Click to plot the selected signals in the plot area above"
                    )
                    
                    if plot_button:
                        # Create a toast notification
                        st.toast(f"Plotting {len(st.session_state.selected_channels)} signals...")
                        
                        # Update plot configuration with selected signals
                        update_plot_config(channels=st.session_state.selected_channels, action_source="user")
                        
                        # Scroll to top to see the plot
                        st.query_params["view"] = "plots"
                        
                        # Manually trigger a rerun to show the plot
                        st.rerun()
                    
                    selected_channel_data = [
                        channel for channel in filtered_channels
                        if channel["Name"] in st.session_state.selected_channels
                    ]
                    if selected_channel_data:
                        with st.expander("Selected Signal Details", expanded=False):
                            st.dataframe(pd.DataFrame(selected_channel_data), use_container_width=True)
                else:
                    st.info("Select signals from the list above to plot them")
                if st.session_state.selected_channels and st.button("Clear Selection", key="clear_selection_customtab", use_container_width=True):
                    st.session_state.selected_channels = []
                    st.rerun()
                if len(filtered_channels) > 200:
                    st.warning(f"Too many signals to display ({len(filtered_channels)}). Please refine your search.")

    # --- RIGHT COLUMN: Chat UI ---
    with right:
        if st.session_state.collapse_right:
            if st.button("⬅️", key="expand_right_btn", help="Expand chat panel", use_container_width=True):
                st.session_state.collapse_right = False
                st.rerun()
            st.write("")
        else:
            if st.button("➡️", key="collapse_right_btn", help="Collapse chat panel", use_container_width=True):
                st.session_state.collapse_right = True
                st.rerun()
            st.header("Chat")
            # --- Scrollable chat message area ---
            chat_height = 700  # px, adjust as needed for your layout
            with st.container(height=chat_height):
                messages: list[ChatMessage] = st.session_state.messages
                if len(messages) == 0:
                    agent_message = "Hello! I'm your automotive data copilot. Upload MDF files and ask me questions about the data."
                    if st.session_state.selected_file_id:
                        file_metadata = next((f for f in st.session_state.files if f.file_id == st.session_state.selected_file_id), None)
                        if file_metadata:
                            agent_message += f"\n\nI see you've selected the file '{file_metadata.filename}'. What would you like to know about this data?"
                            if st.session_state.selected_file_metadata and st.session_state.selected_file_metadata.channels:
                                channel_count = len(st.session_state.selected_file_metadata.channels)
                                sample_channels = list(st.session_state.selected_file_metadata.channels.keys())[:3]
                                agent_message += f"\n\nThis file contains {channel_count} channels. Some examples include: {', '.join(sample_channels)}."
                                agent_message += "\n\nYou can ask me to analyze specific channels, plot data, find anomalies, or calculate statistics."
                    with st.chat_message("ai"):
                        st.write(agent_message)
                async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
                    for m in messages:
                        yield m
                await draw_messages(amessage_iter())
            # --- Chat input and feedback always visible below ---
            if user_input := st.chat_input():
                messages.append(ChatMessage(type="human", content=user_input))
                st.chat_message("human").write(user_input)
                try:
                    agent_config = {}
                    if st.session_state.selected_file_id:
                        file_metadata = next((f for f in st.session_state.files if f.file_id == st.session_state.selected_file_id), None)
                        if file_metadata:
                            file_context = {
                                "file_id": file_metadata.file_id,
                                "filename": file_metadata.filename,
                                "channel_count": file_metadata.channel_count,
                                "duration": file_metadata.duration,
                            }
                            if st.session_state.selected_file_metadata:
                                metadata = st.session_state.selected_file_metadata
                                file_context.update({
                                    "start_time": metadata.start_time,
                                    "end_time": metadata.end_time,
                                    "sample_count": metadata.sample_count,
                                    "file_type": metadata.file_type,
                                })
                                # Always include full channel list and details (unless too large)
                                channel_dict = metadata.channels or {}
                                channel_names = list(channel_dict.keys())
                                file_context["available_channels"] = channel_names
                                # If too many channels, only send schema
                                if len(channel_names) > 200:
                                    file_context["channel_schema_only"] = True
                                    file_context["channel_schema_warning"] = (
                                        f"File has {len(channel_names)} channels. Only channel names and types are included in context to avoid LLM token overflow. "
                                        "Ask for a specific channel to get details."
                                    )
                                    file_context["channel_types"] = {
                                        name: getattr(channel, "data_type", None) for name, channel in channel_dict.items()
                                    }
                                else:
                                    file_context["channel_details"] = {
                                        name: {
                                            "unit": channel.unit,
                                            "min_value": channel.min_value,
                                            "max_value": channel.max_value,
                                            "sampling_rate": channel.sampling_rate,
                                            "description": channel.description,
                                            "ecu": channel.ecu,
                                            "data_type": getattr(channel, "data_type", None),
                                        }
                                        for name, channel in channel_dict.items()
                                    }
                                # Still include selected channels for UI context
                                if hasattr(st.session_state, 'selected_channels') and st.session_state.selected_channels:
                                    file_context["selected_channels"] = st.session_state.selected_channels
                                    
                                # Add plot config for LLM context
                                plot_config = st.session_state.get("plot_config", {})
                                if plot_config and plot_config.get("channels"):
                                    file_context["current_plot"] = {
                                        "channels": plot_config.get("channels", []),
                                        "start_time": plot_config.get("start_time"),
                                        "end_time": plot_config.get("end_time"),
                                        "has_overlays": bool(plot_config.get("overlays")),
                                        "has_annotations": bool(plot_config.get("annotations")),
                                    }
                            agent_config["current_file"] = file_context
                    if use_streaming:
                        stream = agent_client.astream(
                            message=user_input,
                            model=model,
                            thread_id=st.session_state.thread_id,
                            agent_config=agent_config,
                        )
                        await draw_messages(stream, is_new=True)
                    else:
                        response = await agent_client.ainvoke(
                            message=user_input,
                            model=model,
                            thread_id=st.session_state.thread_id,
                            agent_config=agent_config,
                        )
                        messages.append(response)
                        st.chat_message("ai").write(response.content)
                    st.rerun()
                except AgentClientError as e:
                    st.error(f"Error generating response: {e}")
                    st.stop()
            if len(messages) > 0 and st.session_state.last_message:
                with st.session_state.last_message:
                    await handle_feedback()


async def draw_messages(
    messages_agen: AsyncGenerator[ChatMessage | str, None],
    is_new: bool = False,
) -> None:
    """
    Draws a set of chat messages - either replaying existing messages
    or streaming new ones.

    This function has additional logic to handle streaming tokens and tool calls.
    - Use a placeholder container to render streaming tokens as they arrive.
    - Use a status container to render tool calls. Track the tool inputs and outputs
      and update the status container accordingly.

    The function also needs to track the last message container in session state
    since later messages can draw to the same container. This is also used for
    drawing the feedback widget in the latest chat message.

    Args:
        messages_aiter: An async iterator over messages to draw.
        is_new: Whether the messages are new or not.
    """

    # Keep track of the last message container
    last_message_type = None
    st.session_state.last_message = None

    # Placeholder for intermediate streaming tokens
    streaming_content = ""
    streaming_placeholder = None

    # Iterate over the messages and draw them
    while msg := await anext(messages_agen, None):
        # str message represents an intermediate token being streamed
        if isinstance(msg, str):
            # If placeholder is empty, this is the first token of a new message
            # being streamed. We need to do setup.
            if not streaming_placeholder:
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")
                with st.session_state.last_message:
                    streaming_placeholder = st.empty()

            streaming_content += msg
            streaming_placeholder.write(streaming_content)
            continue
        if not isinstance(msg, ChatMessage):
            st.error(f"Unexpected message type: {type(msg)}")
            st.write(msg)
            st.stop()

        match msg.type:
            # A message from the user, the easiest case
            case "human":
                last_message_type = "human"
                st.chat_message("human").write(msg.content)

            # A message from the agent is the most complex case, since we need to
            # handle streaming tokens and tool calls.
            case "ai":
                # If we're rendering new messages, store the message in session state
                if is_new:
                    st.session_state.messages.append(msg)

                # If the last message type was not AI, create a new chat message
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")

                with st.session_state.last_message:
                    # If the message has content, write it out.
                    # Reset the streaming variables to prepare for the next message.
                    if msg.content:
                        if streaming_placeholder:
                            streaming_placeholder.write(msg.content)
                            streaming_content = ""
                            streaming_placeholder = None
                        else:
                            st.write(msg.content)

                    if msg.tool_calls:
                        # Create a status container for each tool call and store the
                        # status container by ID to ensure results are mapped to the
                        # correct status container.
                        call_results = {}
                        for tool_call in msg.tool_calls:
                            status = st.status(
                                f"""Tool Call: {tool_call["name"]}""",
                                state="running" if is_new else "complete",
                            )
                            call_results[tool_call["id"]] = status
                            status.write("Input:")
                            status.write(tool_call["args"])

                        # Expect one ToolMessage for each tool call.
                        for _ in range(len(call_results)):
                            tool_result: ChatMessage = await anext(messages_agen)

                            if tool_result.type != "tool":
                                st.error(f"Unexpected ChatMessage type: {tool_result.type}")
                                st.write(tool_result)
                                st.stop()

                            # Record the message if it's new, and update the correct
                            # status container with the result
                            if is_new:
                                st.session_state.messages.append(tool_result)
                            if tool_result.tool_call_id:
                                status = call_results[tool_result.tool_call_id]
                            status.write("Output:")
                            status.write(tool_result.content)
                            status.update(state="complete")

            case "custom":
                # CustomData example used by the bg-task-agent
                # See:
                # - src/agents/utils.py CustomData
                # - src/agents/bg_task_agent/task.py
                try:
                    task_data: TaskData = TaskData.model_validate(msg.custom_data)
                except ValidationError:
                    st.error("Unexpected CustomData message received from agent")
                    st.write(msg.custom_data)
                    st.stop()

                if is_new:
                    st.session_state.messages.append(msg)

                if last_message_type != "task":
                    last_message_type = "task"
                    st.session_state.last_message = st.chat_message(
                        name="task", avatar=":material/manufacturing:"
                    )
                    with st.session_state.last_message:
                        status = TaskDataStatus()

                status.add_and_draw_task_data(task_data)

            # In case of an unexpected message type, log an error and stop
            case _:
                st.error(f"Unexpected ChatMessage type: {msg.type}")
                st.write(msg)
                st.stop()


async def handle_feedback() -> None:
    """Draws a feedback widget and records feedback from the user."""

    # Keep track of last feedback sent to avoid sending duplicates
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = (None, None)

    latest_run_id = st.session_state.messages[-1].run_id
    feedback = st.feedback("stars", key=latest_run_id)

    # If the feedback value or run ID has changed, send a new feedback record
    if feedback is not None and (latest_run_id, feedback) != st.session_state.last_feedback:
        # Normalize the feedback value (an index) to a score between 0 and 1
        normalized_score = (feedback + 1) / 5.0

        agent_client: AgentClient = st.session_state.agent_client
        try:
            await agent_client.acreate_feedback(
                run_id=latest_run_id,
                key="human-feedback-stars",
                score=normalized_score,
                kwargs={"comment": "In-line human feedback"},
            )
        except AgentClientError as e:
            st.error(f"Error recording feedback: {e}")
            st.stop()
        st.session_state.last_feedback = (latest_run_id, feedback)
        st.toast("Feedback recorded", icon=":material/reviews:")


if __name__ == "__main__":
    asyncio.run(main())
