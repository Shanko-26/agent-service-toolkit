"""
FastAPI router for file management operations.

This module defines API endpoints for uploading, listing, downloading,
and deleting automotive measurement files.
"""

import os
import tempfile
from typing import List, Optional, Dict
from datetime import datetime
from pathlib import Path as FilePath
import uuid

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, Path, status
from fastapi.responses import JSONResponse, FileResponse

from service.utils import verify_bearer
from storage.file_storage import file_storage
from schema.automotive.models import MeasurementFile, FileUploadResponse, FileFormat, SignalMetadata
from data.enhanced_parser import parse_mdf, HAS_ASAMMDF

# Create the router
router = APIRouter(
    prefix="/files",
    tags=["files"]
)


@router.post(
    "/upload", 
    response_model=FileUploadResponse,
    summary="Upload a measurement file"
)
async def upload_file(
    file: UploadFile = File(...),
    description: str = Form(""),
):
    """Upload a measurement file and save it to storage."""
    # Print debug information
    print(f"File received: {file.filename}, size: {file.size if hasattr(file, 'size') else 'unknown'}")
    print(f"Content type: {file.content_type}")
    print(f"Description: {description}")
    
    try:
        # Save the file and get a file ID
        file_id = await file_storage.save_uploaded_file(file)
        
        # Determine file type
        file_type = FileFormat.MF4
        if file.filename:
            if file.filename.lower().endswith('.mdf'):
                file_type = FileFormat.MDF3
            elif file.filename.lower().endswith('.dat'):
                file_type = FileFormat.MDF3
        
        # Get file path to parse
        file_path = file_storage.get_file_path(file_id)
        if not file_path:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="File saved but path not found"
            )
            
        # Verify the file exists and has content
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"File not found on disk: {file_path}"
            )
            
        size_bytes = os.path.getsize(file_path)
        if size_bytes == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes)"
            )
        
        # Try to extract metadata using asammdf if available
        channels = {}
        sample_count = 0
        start_time = datetime.now()
        end_time = datetime.now()
        
        # Parse the file to extract metadata if asammdf is available
        if HAS_ASAMMDF:
            try:
                print(f"Parsing MDF file {file_path}")
                # parse_mdf returns (dataframe, metadata) tuple
                _, parsed_metadata = parse_mdf(file_path)
                
                if parsed_metadata:
                    print(f"Successfully parsed MDF file. Metadata keys: {list(parsed_metadata.keys())}")
                    
                    # Get channel information
                    if "channels" in parsed_metadata:
                        channels = parsed_metadata["channels"]
                    elif "channels_db" in parsed_metadata:
                        channels = parsed_metadata["channels_db"]
                    
                    # Add signal_id to each channel
                    enhanced_channels = {}
                    for channel_name, channel_data in channels.items():
                        # Create a copy of the channel data with signal_id added
                        channel_with_id = dict(channel_data)
                        # Create a deterministic ID based on file_id and channel_name
                        signal_id = f"{file_id[:8]}_{channel_name.replace('.', '_').replace(' ', '_')[:20]}"
                        channel_with_id["signal_id"] = signal_id
                        enhanced_channels[channel_name] = channel_with_id
                    
                    # Replace channels with enhanced version
                    channels = enhanced_channels
                    
                    # Get sample count
                    if "sample_count" in parsed_metadata:
                        sample_count = parsed_metadata.get("sample_count", 0)
                    elif "signal_count" in parsed_metadata:
                        sample_count = parsed_metadata.get("signal_count", 0)
                    
                    # Get time information
                    if "start_time" in parsed_metadata:
                        start_time = parsed_metadata["start_time"]
                    if "end_time" in parsed_metadata:
                        end_time = parsed_metadata["end_time"]
                    
                    print(f"Parsed MDF file: {len(channels)} channels found")
                else:
                    print("No metadata returned from parse_mdf")
            except Exception as e:
                print(f"Error parsing MDF file: {str(e)}")
                # Continue with default metadata if parsing fails
        else:
            print("asammdf not available, skipping file parsing")
                
        # Create metadata
        metadata = MeasurementFile(
            file_id=file_id,
            filename=file.filename or "unknown.mf4",
            file_type=file_type,
            start_time=start_time,
            end_time=end_time,
            channels=channels,
            sample_count=sample_count,
            file_size_bytes=size_bytes,
            description=description
        )
        
        # Save the metadata
        file_storage.save_file_metadata(metadata)
        
        # Create the response
        return FileUploadResponse(
            file_id=file_id,
            filename=file.filename or "unknown.mf4",
            file_type=file_type,
            channel_count=len(channels),
            start_time=start_time,
            end_time=end_time,
            duration=metadata.duration,
            size_bytes=size_bytes,
            message=f"File saved and parsed, found {len(channels)} channels",
            status="ok"
        )
    except Exception as e:
        print(f"Error in upload endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload error: {str(e)}"
        )


@router.get(
    "/list", 
    response_model=List[MeasurementFile],
    summary="List all measurement files",
    description="""
    Retrieve a list of all uploaded measurement files with their metadata.
    
    Returns detailed metadata for all files that have been uploaded, including
    file information, time ranges, and channel details.
    """,
    responses={
        status.HTTP_200_OK: {
            "description": "List of all available files with metadata",
            "model": List[MeasurementFile]
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Server error while listing files"
        }
    }
)
async def list_files():
    """
    List all available measurement files.
    
    Returns metadata for all files that have been uploaded.
    """
    try:
        return file_storage.list_files()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing files: {str(e)}"
        )


@router.get(
    "/{file_id}", 
    response_model=MeasurementFile,
    summary="Get file metadata",
    description="""
    Retrieve detailed metadata for a specific measurement file.
    
    Returns comprehensive information about the file, including channels,
    time ranges, and other extracted metadata.
    """,
    responses={
        status.HTTP_200_OK: {
            "description": "File metadata retrieved successfully",
            "model": MeasurementFile
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "File not found"
        }
    }
)
async def get_file_metadata(
    file_id: str = Path(..., description="The unique ID of the file to retrieve metadata for")
):
    """
    Get metadata for a specific file.
    
    Returns detailed metadata for the requested file ID.
    """
    metadata = file_storage.get_file_metadata(file_id)
    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File with ID {file_id} not found"
        )
    
    return metadata


@router.get(
    "/{file_id}/download",
    summary="Download measurement file",
    description="""
    Download the original measurement file.
    
    Returns the binary content of the requested file with appropriate
    content-type headers for the browser to handle as a download.
    """,
    responses={
        status.HTTP_200_OK: {
            "description": "File content streamed as download",
            "content": {
                "application/octet-stream": {}
            }
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "File not found"
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Server error while retrieving file"
        }
    }
)
async def download_file(
    file_id: str = Path(..., description="The unique ID of the file to download")
):
    """
    Download a measurement file.
    
    Returns the file content for the requested file ID.
    """
    # Get file path
    file_path = file_storage.get_file_path(file_id)
    if not file_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File with ID {file_id} not found"
        )
    
    # Get metadata for the filename
    metadata = file_storage.get_file_metadata(file_id)
    filename = metadata.filename if metadata else f"{file_id}{file_path.suffix}"
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/octet-stream"
    )


@router.delete(
    "/{file_id}",
    summary="Delete measurement file",
    description="""
    Delete a measurement file and its metadata.
    
    Permanently removes the file from storage along with all associated metadata.
    This operation cannot be undone.
    """,
    responses={
        status.HTTP_200_OK: {
            "description": "File deleted successfully",
            "content": {
                "application/json": {
                    "example": {"message": "File {file_id} deleted successfully"}
                }
            }
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "File not found"
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Server error during file deletion"
        }
    }
)
async def delete_file(
    file_id: str = Path(..., description="The unique ID of the file to delete")
):
    """
    Delete a measurement file.
    
    Removes both the file and its metadata.
    """
    # Check if file exists
    metadata = file_storage.get_file_metadata(file_id)
    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File with ID {file_id} not found"
        )
    
    # Delete the file
    if file_storage.delete_file(file_id):
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": f"File {file_id} deleted successfully"}
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting file {file_id}"
        )


@router.post("/test-upload")
async def test_upload(
    file: UploadFile = File(...),
):
    """Absolute minimal test endpoint."""
    print(f"Test upload received: {file.filename}")
    return {"filename": file.filename, "status": "ok"} 