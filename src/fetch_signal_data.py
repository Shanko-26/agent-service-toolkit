import asyncio
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import time
import logging

# Configure logging
logger = logging.getLogger(__name__)

async def fetch_signal_data(file_client, file_id, channel_names, start_time=None, end_time=None):
    """
    Fetch actual signal data from the file client API
    
    Parameters:
    - file_client: FileClient instance
    - file_id: ID of the file to fetch data from
    - channel_names: List of channel names to fetch
    - start_time: Optional start time for data slice
    - end_time: Optional end time for data slice
    
    Returns:
    - Dictionary mapping channel names to signal data objects with pandas DataFrames
    """
    if not channel_names:
        st.warning("No channels selected for plotting")
        logger.warning("fetch_signal_data called with empty channel_names")
        return {}
    
    if not file_id:
        st.warning("No file selected")
        logger.warning("fetch_signal_data called with no file_id")
        return {}
    
    # Create the query parameters
    params = {
        "channels": ",".join(channel_names)
    }
    
    if start_time is not None:
        params["start_time"] = start_time
    
    if end_time is not None:
        params["end_time"] = end_time
    
    try:
        logger.info(f"Fetching data for file ID {file_id}, channels: {channel_names}")
        # Fetch the data from the API
        response = await file_client.get_channel_data(file_id, params)
        
        if not response:
            logger.warning(f"No data returned for file ID {file_id}, channels: {channel_names}")
            return {}
            
        # Process the response into DataFrame objects
        signal_data = {}
        
        # For each channel in the response, create a DataFrame
        for channel_name, channel_data in response.items():
            timestamps = channel_data.get("timestamps", [])
            values = channel_data.get("values", [])
            
            if not timestamps or not values:
                logger.warning(f"Channel {channel_name} has empty data")
                continue
                
            if len(timestamps) != len(values):
                logger.warning(f"Channel {channel_name} has mismatched timestamps and values: {len(timestamps)} vs {len(values)}")
                continue
                
            # Create a pandas DataFrame
            df = pd.DataFrame({
                "timestamp": timestamps,
                "value": values
            })
            
            signal_data[channel_name] = df
            logger.info(f"Loaded {len(df)} data points for channel {channel_name}")
        
        return signal_data
        
    except Exception as e:
        logger.error(f"Error fetching signal data: {str(e)}")
        st.error(f"Error fetching signal data: {str(e)}")
        return {}


async def fetch_file_channels(file_client, file_id):
    """
    Fetch list of channels from a file
    
    Parameters:
    - file_client: FileClient instance
    - file_id: ID of the file to fetch from
    
    Returns:
    - List of channel names
    """
    if not file_id:
        return []
    
    try:
        # Fetch file metadata which includes channel info
        metadata = await file_client.get_file_metadata(file_id)
        
        if not metadata or not metadata.channels:
            logger.warning(f"No channels found in file metadata for file ID {file_id}")
            return []
        
        # Extract channel names from metadata
        channels = list(metadata.channels.keys())
        logger.info(f"Found {len(channels)} channels in file ID {file_id}")
        return channels
        
    except Exception as e:
        logger.error(f"Error fetching file channels: {str(e)}")
        st.error(f"Error fetching file channels: {str(e)}")
        return [] 