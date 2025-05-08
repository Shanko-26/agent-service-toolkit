"""
File storage management for automotive measurement files.

This module provides utilities for storing, retrieving and managing
measurement files and their metadata.
"""

import os
import uuid
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Union, BinaryIO

from fastapi import UploadFile

from schema.automotive.models import MeasurementFile, FileFormat
from core import settings

# Default storage paths
DEFAULT_STORAGE_DIR = "file_storage"
DEFAULT_METADATA_DIR = "file_metadata"

class FileStorageManager:
    """Manages the storage and retrieval of measurement files and metadata."""
    
    def __init__(
        self,
        storage_dir: str = DEFAULT_STORAGE_DIR,
        metadata_dir: str = DEFAULT_METADATA_DIR
    ):
        """Initialize the file storage manager.
        
        Args:
            storage_dir: Directory to store the measurement files
            metadata_dir: Directory to store the metadata files
        """
        self.storage_base = Path(storage_dir)
        self.metadata_base = Path(metadata_dir)
        
        # Create directories if they don't exist
        self.storage_base.mkdir(parents=True, exist_ok=True)
        self.metadata_base.mkdir(parents=True, exist_ok=True)
    
    async def save_uploaded_file(self, file: UploadFile) -> str:
        """Save an uploaded file to storage.
        
        Args:
            file: The uploaded file
            
        Returns:
            The unique file ID assigned to the file
        """
        # Generate a unique ID for the file
        file_id = str(uuid.uuid4())
        
        # Determine file extension from original filename
        _, ext = os.path.splitext(file.filename or "unknown.mf4")
        if not ext:
            ext = ".mf4"  # Default extension
        
        # Create the destination path
        dest_path = self.storage_base / f"{file_id}{ext}"
        
        try:
            # Make sure we're at the beginning of the file
            await file.seek(0)
            
            # Read the entire file content
            content = await file.read()
            
            if not content:
                print(f"Warning: Uploaded file {file.filename} has no content (0 bytes)")
            else:
                print(f"Reading file content: {len(content)} bytes")
            
            # Write content to disk
            with open(dest_path, "wb") as f:
                f.write(content)
                
            # Verify file was written successfully
            if os.path.exists(dest_path):
                file_size = os.path.getsize(dest_path)
                print(f"File saved successfully at {dest_path}, size: {file_size} bytes")
                if file_size == 0:
                    print("Warning: File was saved but has 0 bytes")
            else:
                print(f"Error: File was not saved at {dest_path}")
                
            # Reset the file for any subsequent operations
            await file.seek(0)
            
            return file_id
            
        except Exception as e:
            print(f"Error saving uploaded file: {str(e)}")
            # Still return the ID even if there was an error, to avoid crashing
            return file_id
    
    def save_file_metadata(self, metadata: MeasurementFile) -> None:
        """Save the metadata for a file.
        
        Args:
            metadata: The file metadata to save
        """
        metadata_path = self.metadata_base / f"{metadata.file_id}.json"
        
        with open(metadata_path, "w") as f:
            # Convert to dict and then to JSON
            json.dump(metadata.model_dump(), f, indent=2, default=str)
    
    def get_file_metadata(self, file_id: str) -> Optional[MeasurementFile]:
        """Get the metadata for a file.
        
        Args:
            file_id: The ID of the file
            
        Returns:
            The file metadata or None if not found
        """
        metadata_path = self.metadata_base / f"{file_id}.json"
        
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, "r") as f:
            data = json.load(f)
            return MeasurementFile.model_validate(data)
    
    def get_file_path(self, file_id: str) -> Optional[Path]:
        """Get the path to a stored file.
        
        Args:
            file_id: The ID of the file
            
        Returns:
            The path to the file or None if not found
        """
        # Check for files with this ID with any extension
        for item in self.storage_base.glob(f"{file_id}.*"):
            if item.is_file():
                return item
        
        return None
    
    def list_files(self) -> List[MeasurementFile]:
        """List all stored files.
        
        Returns:
            A list of file metadata
        """
        files = []
        
        for metadata_file in self.metadata_base.glob("*.json"):
            try:
                with open(metadata_file, "r") as f:
                    data = json.load(f)
                    files.append(MeasurementFile.model_validate(data))
            except (json.JSONDecodeError, Exception) as e:
                print(f"Error reading metadata file {metadata_file}: {e}")
        
        return files
    
    def delete_file(self, file_id: str) -> bool:
        """Delete a file and its metadata.
        
        Args:
            file_id: The ID of the file
            
        Returns:
            True if the file was deleted, False otherwise
        """
        # Delete the metadata file
        metadata_path = self.metadata_base / f"{file_id}.json"
        file_deleted = False
        metadata_deleted = False
        
        # Delete the data file
        file_path = self.get_file_path(file_id)
        if file_path:
            try:
                file_path.unlink()
                file_deleted = True
            except Exception:
                pass
        
        # Delete the metadata
        if metadata_path.exists():
            try:
                metadata_path.unlink()
                metadata_deleted = True
            except Exception:
                pass
        
        return file_deleted or metadata_deleted

# Create a singleton instance
file_storage = FileStorageManager() 