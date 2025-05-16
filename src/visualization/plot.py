"""
Visualization module for plotting automotive measurement data.

This module provides functions for:
1. Managing plot state in Streamlit session
2. Updating plot configuration (channels, time range, etc.)
3. Fetching signal data from the backend
4. Rendering plots using Plotly

The module is designed to handle both user-driven and LLM-driven plot updates.
"""

import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, TypedDict, Literal
import logging
import asyncio
from datetime import datetime
from plotly.subplots import make_subplots

# Configure logging
logger = logging.getLogger(__name__)

# Type aliases
PlotConfig = Dict[str, Any]
SignalData = Dict[str, Dict[str, List[float]]]

class OverlayConfig(TypedDict, total=False):
    """Configuration for plot overlays like moving averages."""
    type: str  # e.g., "moving_average"
    channel: str  # channel to apply overlay to
    window: int  # window size for moving average

class AnnotationConfig(TypedDict, total=False):
    """Configuration for plot annotations like threshold lines."""
    type: str  # e.g., "threshold"
    value: float  # value for threshold
    label: str  # label for annotation
    color: str  # color for annotation
    width: int  # line width
    dash: str  # line dash style

class AxisConfig(TypedDict, total=False):
    """Configuration for a Y-axis."""
    title: str  # axis title
    unit: str  # unit to display (e.g., "km/h", "°C")
    channels: List[str]  # channels to plot on this axis
    position: Literal["left", "right"]  # position of the axis
    range: List[float]  # optional fixed range for the axis

class SubplotConfig(TypedDict, total=False):
    """Configuration for a subplot."""
    title: str  # subplot title
    channels: List[str]  # channels to plot in this subplot
    row: int  # row position in subplot grid (1-based)
    col: int  # column position in subplot grid (1-based)
    axis_config: AxisConfig  # axis configuration for this subplot

def init_plot_state() -> None:
    """
    Initialize plot configuration in session state if not exists.
    
    This sets up the default plot configuration with empty values.
    """
    st.session_state.setdefault("plot_config", {
        "channels": [],         # List of channel names to plot
        "start_time": None,     # Start time in seconds
        "end_time": None,       # End time in seconds
        "overlays": [],         # List of overlay configurations
        "annotations": [],      # List of annotations
        "layout": {},           # Plotly layout customizations
        "last_update": None,    # Timestamp of last update
        "action_source": None,  # "user" or "llm"
    })

def update_plot_config(
    channels: Optional[List[str]] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    overlays: Optional[List[Dict[str, Any]]] = None,
    annotations: Optional[List[Dict[str, Any]]] = None,
    layout: Optional[Dict[str, Any]] = None,
    action_source: str = "user"
) -> None:
    """
    Update plot configuration in session state.
    
    Args:
        channels: List of channel names to plot
        start_time: Start time in seconds
        end_time: End time in seconds
        overlays: List of overlay configurations
        annotations: List of annotations
        layout: Plotly layout customizations
        action_source: Source of the update ("user" or "llm")
    """
    # Initialize if not exists
    init_plot_state()
    
    # Get current config
    cfg = st.session_state["plot_config"].copy()
    
    # Update with new values
    if channels is not None:
        cfg["channels"] = channels
    if start_time is not None:
        cfg["start_time"] = start_time
    if end_time is not None:
        cfg["end_time"] = end_time
    if overlays is not None:
        cfg["overlays"] = overlays
    if annotations is not None:
        cfg["annotations"] = annotations
    if layout is not None:
        cfg["layout"] = layout
    
    # Set metadata
    cfg["last_update"] = datetime.now().isoformat()
    cfg["action_source"] = action_source
    
    # Update session state
    st.session_state["plot_config"] = cfg
    
    # Log the update
    logger.info(f"Plot config updated by {action_source}: {len(cfg['channels'])} channels")

async def fetch_signal_data(
    file_client,
    file_id: str, 
    channels: List[str],
    start_time: Optional[float] = None,
    end_time: Optional[float] = None
) -> SignalData:
    """
    Fetch signal data from the backend API.
    
    Args:
        file_client: The FileClient instance
        file_id: ID of the file to fetch data from
        channels: List of channel names to fetch
        start_time: Optional start time for data slice (seconds)
        end_time: Optional end time for data slice (seconds)
        
    Returns:
        Dictionary with channel data (timestamps and values)
    """
    if not channels:
        logger.warning("fetch_signal_data called with empty channels list")
        return {}
    
    if not file_id:
        logger.warning("fetch_signal_data called with no file_id")
        return {}
    
    # Create query params
    params = {
        "channels": ",".join(channels)
    }
    
    if start_time is not None:
        params["start_time"] = start_time
    
    if end_time is not None:
        params["end_time"] = end_time
    
    try:
        # Fetch data from API
        logger.info(f"Fetching data for file {file_id}, channels: {channels}")
        data = await file_client.get_channel_data(file_id, params)
        
        if not data:
            logger.warning("No data returned from API")
            return {}
        
        # Return the data directly (API already formats it correctly)
        return data
        
    except Exception as e:
        logger.error(f"Error fetching signal data: {str(e)}")
        return {}

def prepare_plot_data(signal_data: SignalData) -> pd.DataFrame:
    """
    Prepare signal data for plotting by converting to a pandas DataFrame.
    
    Args:
        signal_data: Dictionary with channel data from the API
        
    Returns:
        DataFrame with time as index and channels as columns
    """
    if not signal_data:
        return pd.DataFrame()
    
    # First, extract all unique timestamps across all channels
    all_timestamps = set()
    for channel_name, channel_data in signal_data.items():
        all_timestamps.update(channel_data.get("timestamps", []))
    
    all_timestamps = sorted(all_timestamps)
    
    # Create a DataFrame with all timestamps as index
    df = pd.DataFrame(index=all_timestamps)
    df.index.name = "time"
    
    # Add each channel as a column
    for channel_name, channel_data in signal_data.items():
        timestamps = channel_data.get("timestamps", [])
        values = channel_data.get("values", [])
        
        if len(timestamps) != len(values):
            logger.warning(f"Channel {channel_name} has mismatched timestamps and values")
            continue
        
        # Create a temporary Series for this channel
        channel_series = pd.Series(values, index=timestamps)
        
        # Add to the main DataFrame (will align on timestamps)
        df[channel_name] = channel_series
    
    # Sort by timestamp
    df.sort_index(inplace=True)
    
    return df

def render_plot(data: pd.DataFrame, config: Optional[PlotConfig] = None) -> None:
    """
    Render a plot using Plotly.
    
    Args:
        data: DataFrame with time as index and channels as columns
        config: Optional plot configuration (uses session state if None)
    """
    if config is None:
        config = st.session_state.get("plot_config", {})
    
    if data.empty:
        st.warning("No data available to plot")
        return
    
    # Define grid visibility state in session if not exists
    if "show_grid" not in st.session_state:
        st.session_state.show_grid = True
    
    # Create control button for grid visibility 
    show_grid = st.toggle("Show grid", value=st.session_state.show_grid)
    st.session_state.show_grid = show_grid
    
    # Always use dark mode colors
    colors = px.colors.qualitative.Vivid
    plot_bg_color = "rgb(20, 20, 30)"
    paper_bg_color = "rgb(17, 17, 27)"
    font_color = "rgb(240, 240, 250)"
    grid_color = "rgba(150, 150, 150, 0.2)"
    
    # Standard single-axis plot
    fig = go.Figure()
    
    # Add traces for each channel
    for i, channel in enumerate(config.get("channels", [])):
        if channel in data.columns:
            color_idx = i % len(colors)
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[channel],
                mode="lines",
                name=channel,
                line=dict(color=colors[color_idx], width=2),
                hovertemplate=f"{channel}: %{{y:.4f}}<br>Time: %{{x:.4f}} s<extra></extra>"
            ))
    
    # Handle overlays (like moving averages)
    for i, overlay in enumerate(config.get("overlays", [])):
        overlay_type = overlay.get("type")
        channel = overlay.get("channel")
        
        if overlay_type == "moving_average" and channel in data.columns:
            window = overlay.get("window", 10)
            if not data[channel].isna().all():
                # Calculate moving average
                ma = data[channel].rolling(window=window).mean()
                color_idx = (i + len(config.get("channels", []))) % len(colors)
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=ma,
                    mode="lines",
                    line=dict(color=colors[color_idx], dash="dash", width=1.5),
                    name=f"{channel} (MA-{window})",
                    hovertemplate=f"{channel} MA-{window}: %{{y:.4f}}<br>Time: %{{x:.4f}} s<extra></extra>"
                ))
    
    # Add annotations (like threshold lines)
    for i, annotation in enumerate(config.get("annotations", [])):
        if annotation.get("type") == "threshold":
            value = annotation.get("value")
            label = annotation.get("label", f"Threshold: {value}")
            color = annotation.get("color", colors[0])
            
            # Add to main plot
            fig.add_shape(
                type="line",
                xref="paper",
                x0=0,
                x1=1,
                y0=value,
                y1=value,
                line=dict(
                    color=color,
                    width=annotation.get("width", 2),
                    dash=annotation.get("dash", "dash"),
                )
            )
            
            fig.add_annotation(
                xref="paper",
                x=1,
                y=value,
                text=label,
                showarrow=False,
                xanchor="right",
                font=dict(color=color)
            )
    
    # Configure layout with improved appearance and theme options
    layout = {
        "title": config.get("title", "Signal Data Plot"),
        "xaxis": {
            "title": "Time (s)",
            "gridcolor": grid_color if show_grid else "rgba(0,0,0,0)",
            "showgrid": show_grid,
            "showline": True,
            "linewidth": 1,
            "linecolor": "rgba(150, 150, 150, 0.5)",
            "mirror": True,
            "zeroline": False,
        },
        "height": 550,
        "legend": {
            "orientation": "h", 
            "yanchor": "bottom", 
            "y": -0.2,
            "font": {"color": font_color}
        },
        "margin": {"l": 60, "r": 60, "t": 60, "b": 80},
        "hovermode": "x unified",
        "plot_bgcolor": plot_bg_color,
        "paper_bgcolor": paper_bg_color,
        "font": {"color": font_color},
        "dragmode": "zoom",
        # Add custom layout settings from config
        **config.get("layout", {})
    }
    
    # Create a meaningful y-axis title
    channels = config.get("channels", [])
    if len(channels) == 1:
        # If there's only one channel, use its name as the y-axis title
        y_axis_title = channels[0]
    elif len(channels) > 1:
        # For multiple channels, use a more generic title with count
        if len(channels) <= 3:
            # For 2-3 channels, list them all
            y_axis_title = f"Values: {', '.join(channels)}"
        else:
            # For more channels, just give the count
            y_axis_title = f"Values for {len(channels)} channels"
    else:
        y_axis_title = "Value"
        
    layout["yaxis"] = {
        "title": y_axis_title,
        "gridcolor": grid_color if show_grid else "rgba(0,0,0,0)",
        "showgrid": show_grid,
        "showline": True,
        "linewidth": 1, 
        "linecolor": "rgba(150, 150, 150, 0.5)",
        "mirror": True,
        "zeroline": False,
    }
    
    fig.update_layout(**layout)
    
    # Add buttons for zoom and pan functionality
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                buttons=[
                    dict(
                        args=[{"yaxis.autorange": True, "xaxis.autorange": True}],
                        label="Reset Zoom",
                        method="relayout"
                    ),
                ],
                pad={"r": 10, "t": 10},
                showactive=False,
                x=0.0,
                xanchor="left",
                y=1.1,
                yanchor="top",
                font={"color": font_color}
            )
        ]
    )
    
    # Add range selector for quick time window adjustment
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeslider_thickness=0.05,
    )
    
    # Render the plot
    st.plotly_chart(fig, use_container_width=True, config={
        'displayModeBar': True,
        'scrollZoom': True,
        'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawcircle', 'drawrect', 'eraseshape']
    })

def download_data_as_csv(data: pd.DataFrame, filename="signal_data.csv"):
    """
    Create a download button for the signal data as CSV.
    
    Args:
        data: DataFrame with the signal data
        filename: Name for the downloaded file
    """
    if data.empty:
        return
        
    # Reset index to make time a column
    download_df = data.reset_index()
    
    # Use to_csv to convert the DataFrame to a CSV string
    csv = download_df.to_csv(index=False)
    
    # Create a download button
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv,
        file_name=filename,
        mime="text/csv",
        help="Download the plotted signal data as a CSV file",
        use_container_width=True
    )

async def handle_plot_request(file_client, file_id: str) -> None:
    """
    Handle a plot request and render the plot.
    
    This function:
    1. Gets the plot configuration from session state
    2. Fetches the necessary data
    3. Renders the plot
    
    Args:
        file_client: FileClient instance
        file_id: ID of the file to plot
    """
    # Initialize if needed
    init_plot_state()
    
    # Get configuration
    config = st.session_state.get("plot_config", {})
    
    # Validate
    if not config.get("channels"):
        st.warning("Please select channels to plot")
        return
        
    if not file_id:
        st.warning("Please select a file to plot")
        return
    
    try:
        # Fetch data
        signal_data = await fetch_signal_data(
            file_client=file_client,
            file_id=file_id,
            channels=config["channels"],
            start_time=config.get("start_time"),
            end_time=config.get("end_time")
        )
        
        if not signal_data:
            st.error("No data returned from the server. The channels might not exist or contain no data.")
            return
        
        # Prepare data for plotting
        plot_data = prepare_plot_data(signal_data)
        
        if plot_data.empty:
            st.warning("No data available for the selected channels in the specified time range.")
            return
        
        # Render the plot
        render_plot(plot_data, config)
        
        # Add download button for the data
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success(f"📊 Plotted {len(config['channels'])} channels with {len(plot_data)} data points")
        with col2:
            # Generate a meaningful filename based on selected channels
            channels_str = "_".join(config["channels"][:2])
            if len(config["channels"]) > 2:
                channels_str += f"_and_{len(config['channels'])-2}_more"
            filename = f"{channels_str}_data.csv"
            download_data_as_csv(plot_data, filename)
        
    except Exception as e:
        st.error(f"Error plotting data: {str(e)}")
        logger.exception("Plot error")
        # Show detailed error for debugging
        with st.expander("Debug Information"):
            st.write(f"Error Type: {type(e).__name__}")
            st.write(f"Error Details: {str(e)}")
            st.write(f"File ID: {file_id}")
            st.write(f"Channels: {config.get('channels', [])}")
            st.write(f"Time Range: {config.get('start_time', 'None')} to {config.get('end_time', 'None')}")

def handle_llm_tool_call(tool_args: Dict[str, Any]) -> None:
    """
    Handle a tool call from the LLM to plot data.
    
    Args:
        tool_args: Arguments from the LLM tool call
    """
    # Extract arguments
    channels = tool_args.get("channels", [])
    start_time = tool_args.get("start_time")
    end_time = tool_args.get("end_time")
    overlays = tool_args.get("overlays", [])
    annotations = tool_args.get("annotations", [])
    layout = tool_args.get("layout", {})
    
    # Update plot config
    update_plot_config(
        channels=channels,
        start_time=start_time,
        end_time=end_time,
        overlays=overlays,
        annotations=annotations,
        layout=layout,
        action_source="llm"
    )
    
    # Log the tool call
    logger.info(f"LLM tool call: Plot updated with {len(channels)} channels")
    
    # Display a notification to the user
    if channels:
        channel_list = ", ".join(channels[:3])
        if len(channels) > 3:
            channel_list += f" and {len(channels) - 3} more"
        st.info(f"🤖 AI updated the plot with channels: {channel_list}")

def render_time_controls(metadata=None) -> None:
    """
    Render time range controls for the plot.
    
    Args:
        metadata: Optional file metadata to get min/max time
    """
    # Initialize time range in plot config if not set
    init_plot_state()
    cfg = st.session_state["plot_config"]
    
    # Default time range (0 to duration or 120s if metadata not available)
    if metadata:
        default_start = 0
        default_end = float(metadata.duration) if hasattr(metadata, 'duration') else 120.0
    else:
        default_start = 0
        default_end = 120.0
    
    # Use existing values if set
    current_start = cfg.get("start_time", default_start)
    current_end = cfg.get("end_time", default_end)
    
    # Ensure current values are within range
    current_start = max(0, current_start if current_start is not None else default_start)
    current_end = min(default_end, current_end if current_end is not None else default_end)
    
    # Create the time range control
    st.subheader("Time Range")
    time_range = st.slider(
        "Select time window (seconds)",
        min_value=float(default_start),
        max_value=float(default_end),
        value=(float(current_start or default_start), float(current_end or default_end)),
        key="plot_time_range"
    )
    
    # Update plot config if time range changed
    if time_range and (time_range[0] != current_start or time_range[1] != current_end):
        update_plot_config(
            start_time=time_range[0],
            end_time=time_range[1],
            action_source="user"
        )
