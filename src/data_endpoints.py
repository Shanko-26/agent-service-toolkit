"""
API endpoints for accessing measurement data.

This module adds endpoints for reading channel data from measurement files,
extending the basic file management capabilities.
"""

import os
import sys
import time
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Query, Path, status
from fastapi.responses import JSONResponse

from core import settings
from storage.file_storage import file_storage
from data.enhanced_parser import parse_mdf, HAS_ASAMMDF, Signal

# Configure logging
logger = logging.getLogger(__name__)

# Create the router
router = APIRouter(
    prefix="/files",
    tags=["data"]
)

@router.get(
    "/{file_id}/data",
    summary="Get channel data",
    description="""
    Get data for specified channels from a measurement file.
    
    Retrieves timestamp and value pairs for the requested channels.
    Optionally limit the time range with start_time and end_time parameters.
    """,
    responses={
        status.HTTP_200_OK: {
            "description": "Channel data retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "ECM.ECM_EngineSpeed": {
                            "timestamps": [0.0, 0.1, 0.2],
                            "values": [1000.0, 1050.0, 1100.0]
                        }
                    }
                }
            }
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "File or channel not found"
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Error retrieving channel data"
        }
    }
)
async def get_channel_data(
    file_id: str = Path(..., description="The ID of the file to get data from"),
    channels: str = Query(..., description="Comma-separated list of channel names"),
    start_time: Optional[float] = Query(None, description="Start time in seconds from beginning of recording"),
    end_time: Optional[float] = Query(None, description="End time in seconds from beginning of recording")
):
    """
    Get data for specified channels from a file.
    
    Args:
        file_id: The ID of the file
        channels: Comma-separated list of channel names
        start_time: Optional start time in seconds from beginning of recording
        end_time: Optional end time in seconds from beginning of recording
        
    Returns:
        Dictionary of channel data with timestamps and values
    """
    # Check if asammdf is available
    if not HAS_ASAMMDF:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Data access is not available because asammdf is not installed"
        )
    
    # Get file path
    file_path = file_storage.get_file_path(file_id)
    if not file_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File with ID {file_id} not found"
        )
    
    # Parse the channels parameter
    channel_list = [ch.strip() for ch in channels.split(",") if ch.strip()]
    if not channel_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid channel names provided"
        )
    
    try:
        # Get file metadata to confirm channels exist
        metadata = file_storage.get_file_metadata(file_id)
        if not metadata:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Metadata for file ID {file_id} not found"
            )
        
        # Check if channels exist in metadata
        missing_channels = []
        for channel in channel_list:
            if metadata.channels and channel not in metadata.channels:
                missing_channels.append(channel)
        
        if missing_channels:
            logger.warning(f"Requested channels not found: {missing_channels}")
            # Filter out missing channels, but continue if we have at least one valid channel
            channel_list = [ch for ch in channel_list if ch not in missing_channels]
            
            if not channel_list:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"None of the requested channels found in file {file_id}"
                )
        
        # For now, let's implement a temporary solution - parse the MDF file directly
        # In a production system, this would likely use cached data or a dedicated service
        time_range = (start_time, end_time) if start_time is not None or end_time is not None else None
        
        # Use the parse_mdf function to get the channel data
        # We want just the signal objects, not a DataFrame
        signals, _ = parse_mdf(
            file_path=file_path,
            channels=channel_list,
            time_range=time_range,
            return_format='signals'
        )
        
        # Format the data for the response
        result = {}
        for channel_name, signal in signals.items():
            if channel_name in channel_list:
                # Convert numpy arrays to Python lists for JSON serialization
                result[channel_name] = {
                    "timestamps": signal.timestamps.tolist() if hasattr(signal.timestamps, 'tolist') else signal.timestamps,
                    "values": signal.samples.tolist() if hasattr(signal.samples, 'tolist') else signal.samples
                }
        
        # If we have at least one channel's data, return it
        if result:
            return result
        else:
            # This shouldn't happen if we've validated channels properly, but just in case
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No data found for the requested channels in file {file_id}"
            )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error getting channel data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting channel data: {str(e)}"
        )

@router.get(
    "/{file_id}/statistics",
    summary="Get channel statistics",
    description="""
    Get statistics for specified channels from a measurement file.
    
    Retrieves min, max, mean, and other statistical values for the requested channels.
    """,
    responses={
        status.HTTP_200_OK: {
            "description": "Channel statistics retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "ECM.ECM_EngineSpeed": {
                            "min": 800.0,
                            "max": 5500.0,
                            "mean": 2100.0,
                            "std": 950.0,
                            "count": 1000
                        }
                    }
                }
            }
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "File or channel not found"
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Error retrieving channel statistics"
        }
    }
)
async def get_channel_statistics(
    file_id: str = Path(..., description="The ID of the file to get statistics from"),
    channels: str = Query(..., description="Comma-separated list of channel names")
):
    """
    Get statistics for specified channels from a file.
    
    Args:
        file_id: The ID of the file
        channels: Comma-separated list of channel names
        
    Returns:
        Dictionary of channel statistics
    """
    # Check if asammdf is available
    if not HAS_ASAMMDF:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Data access is not available because asammdf is not installed"
        )
    
    # Get file path
    file_path = file_storage.get_file_path(file_id)
    if not file_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File with ID {file_id} not found"
        )
    
    # Parse the channels parameter
    channel_list = [ch.strip() for ch in channels.split(",") if ch.strip()]
    if not channel_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid channel names provided"
        )
    
    try:
        # Get file metadata to confirm channels exist
        metadata = file_storage.get_file_metadata(file_id)
        if not metadata:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Metadata for file ID {file_id} not found"
            )
        
        # Check if channels exist in metadata
        for channel in channel_list:
            if metadata.channels and channel not in metadata.channels:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Channel {channel} not found in file {file_id}"
                )
        
        # Parse the MDF file to get the channel data
        signals, _ = parse_mdf(
            file_path=file_path,
            channels=channel_list,
            return_format='signals'
        )
        
        # Calculate statistics for each channel
        result = {}
        for channel_name, signal in signals.items():
            if channel_name in channel_list and len(signal.samples) > 0:
                # Calculate basic statistics
                stats = {
                    "min": float(np.min(signal.samples)),
                    "max": float(np.max(signal.samples)),
                    "mean": float(np.mean(signal.samples)),
                    "std": float(np.std(signal.samples)),
                    "count": int(len(signal.samples))
                }
                result[channel_name] = stats
        
        # Return the channel statistics
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error getting channel statistics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting channel statistics: {str(e)}"
        )

# Function to register these endpoints with the main API
def register_data_endpoints(app):
    """Register the data endpoints with the main FastAPI application."""
    app.include_router(router)
    print("Data endpoints registered successfully") 