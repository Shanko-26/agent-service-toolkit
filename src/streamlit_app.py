import asyncio
import os
import urllib.parse
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError
import pandas as pd

from client import AgentClient, AgentClientError, FileClient, FileClientError
from schema import ChatHistory, ChatMessage
from schema.task_data import TaskData, TaskDataStatus
from schema.automotive.models import MeasurementFile

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
                st.session_state.agent_client = AgentClient(base_url=agent_url)
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
                st.toast("Metadata loaded successfully!")
        except FileClientError as e:
            st.error(f"Error loading file metadata: {e}")
            st.session_state.selected_file_metadata = None

    # Config options
    with st.sidebar:
        st.header(f"{APP_ICON} {APP_TITLE}")

        ""
        "Interact with automotive measurement data using natural language"
        ""

        if st.button(":material/chat: New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()
        
        # File upload section
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
                            # Refresh the file list
                            st.session_state.files = await file_client.list_files()
                            # Auto-select the newly uploaded file
                            st.session_state.selected_file_id = response.file_id
                            # Force metadata refresh
                            st.session_state.selected_file_metadata = None
                            st.toast(f"File list updated, found {len(st.session_state.files)} files")
                        else:
                            st.warning(f"Upload issue: {response.message}")
                except FileClientError as e:
                    st.error(f"Error uploading file: {e}")
        
        # File list and selection
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
                        # Force metadata refresh
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
                                st.success(f"Deleted {file.filename}")
                                st.rerun()
                        except FileClientError as e:
                            st.error(f"Error deleting file: {e}")
            
            # Refresh files button
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
            agent_idx = agent_list.index(agent_client.info.default_agent)
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
            # if it's not localhost, switch to https by default
            if not st_base_url.startswith("https") and "localhost" not in st_base_url:
                st_base_url = st_base_url.replace("http", "https")
            chat_url = f"{st_base_url}?thread_id={st.session_state.thread_id}"
            st.markdown(f"**Chat URL:**\n```text\n{chat_url}\n```")
            st.info("Copy the above URL to share or revisit this chat")

        if st.button(":material/upload: Share/resume chat", use_container_width=True):
            share_chat_dialog()

        "[View the source code](https://github.com/JoshuaC215/agent-service-toolkit)"
        st.caption("Based on agent-service-toolkit by Joshua")

    # Main chat area
    # Draw existing messages
    messages: list[ChatMessage] = st.session_state.messages

    # Display file metadata if a file is selected
    if st.session_state.selected_file_metadata:
        metadata = st.session_state.selected_file_metadata
        with st.expander(f"📊 File Metadata: {metadata.filename}", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("File Information")
                st.write(f"**File ID:** {metadata.file_id}")
                st.write(f"**Format:** {metadata.file_type}")
                st.write(f"**Size:** {metadata.file_size_bytes / (1024*1024):.2f} MB")
                st.write(f"**Description:** {metadata.description or 'N/A'}")
            
            with col2:
                st.subheader("Time Information")
                # Format the start and end times
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
            
            # Display signal/channel information
            st.subheader(f"Channels ({len(metadata.channels)})")
            
            # Create a DataFrame to display channels in tabular form
            if metadata.channels:
                channels_data = []
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
                
                # Create a DataFrame and display it
                channels_df = pd.DataFrame(channels_data)
                st.dataframe(channels_df, use_container_width=True)
                
                # Option to filter signals
                signal_filter = st.text_input("Filter signals", placeholder="Enter search term...")
                if signal_filter:
                    filtered_channels = [
                        channel for channel in channels_data 
                        if signal_filter.lower() in channel["Name"].lower() or
                           signal_filter.lower() in (channel["Description"] or "").lower()
                    ]
                    if filtered_channels:
                        st.dataframe(pd.DataFrame(filtered_channels), use_container_width=True)
                    else:
                        st.info("No channels match your filter criteria.")

    if len(messages) == 0:
        agent_message = "Hello! I'm your automotive data copilot. Upload MDF files and ask me questions about the data."
        if st.session_state.selected_file_id:
            file_metadata = next((f for f in st.session_state.files if f.file_id == st.session_state.selected_file_id), None)
            if file_metadata:
                agent_message += f"\n\nI see you've selected the file '{file_metadata.filename}'. What would you like to know about this data?"
        
        with st.chat_message("ai"):
            st.write(agent_message)

    # draw_messages() expects an async iterator over messages
    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            yield m

    await draw_messages(amessage_iter())

    # Generate new message if the user provided new input
    if user_input := st.chat_input():
        messages.append(ChatMessage(type="human", content=user_input))
        st.chat_message("human").write(user_input)
        try:
            # Add context about selected file to the message
            agent_config = {}
            if st.session_state.selected_file_id:
                file_metadata = next((f for f in st.session_state.files if f.file_id == st.session_state.selected_file_id), None)
                if file_metadata:
                    agent_config["current_file"] = {
                        "file_id": file_metadata.file_id,
                        "filename": file_metadata.filename,
                        "channel_count": file_metadata.channel_count,
                        "duration": file_metadata.duration,
                    }
            
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
            st.rerun()  # Clear stale containers
        except AgentClientError as e:
            st.error(f"Error generating response: {e}")
            st.stop()

    # If messages have been generated, show feedback widget
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
