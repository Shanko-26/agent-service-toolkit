"""
Tests for the file storage functionality.

This module tests the FileStorageManager class directly without involving API routes.
"""

import os
import sys
import unittest
import asyncio
from pathlib import Path
import tempfile
import json
import uuid
from unittest.mock import MagicMock, patch

# Add the parent directory to sys.path to find the src module
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import pytest
from src.storage.file_storage import FileStorageManager
from src.schema.automotive.models import MeasurementFile, FileFormat, SignalMetadata


class TestFileStorage(unittest.TestCase):
    """Test case for the FileStorageManager class."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for the file storage
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_dir = Path(self.temp_dir.name) / "files"
        self.metadata_dir = Path(self.temp_dir.name) / "metadata"
        
        # Create the directories
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a test storage manager with the temporary directories
        self.file_storage = FileStorageManager(
            storage_dir=str(self.storage_dir),
            metadata_dir=str(self.metadata_dir)
        )
        
        # Create sample file metadata for testing
        self.sample_metadata = MeasurementFile(
            file_id="test-file-id",
            filename="test_file.mf4",
            file_path=str(self.storage_dir / "test-file-id.mf4"),
            file_type=FileFormat.MF4,
            start_time=0.0,
            end_time=10.0,
            duration=10.0,
            sample_count=1000,
            channels={
                "Engine_Speed": SignalMetadata(
                    name="Engine_Speed",
                    qualified_name="ECM.Engine_Speed",
                    display_name="Engine Speed",
                    unit="rpm",
                    min_value=0.0,
                    max_value=8000.0,
                    description="Engine rotational speed",
                    sampling_rate=100.0,
                    signal_id="test-file-id_Engine_Speed",
                    ecu="ECM",
                    data_type="float32"
                )
            },
            channel_count=1,
            file_size_bytes=1024,
            description="Test file"
        )
    
    def tearDown(self):
        """Clean up after tests."""
        # Clean up the temporary directory
        self.temp_dir.cleanup()
    
    def test_save_metadata(self):
        """Test saving and retrieving file metadata."""
        # Save the metadata
        self.file_storage.save_file_metadata(self.sample_metadata)
        
        # Check that the metadata file was created
        metadata_path = self.metadata_dir / f"{self.sample_metadata.file_id}.json"
        self.assertTrue(metadata_path.exists())
        
        # Retrieve the metadata
        retrieved_metadata = self.file_storage.get_file_metadata(self.sample_metadata.file_id)
        
        # Check that the retrieved metadata matches the original
        self.assertEqual(retrieved_metadata.file_id, self.sample_metadata.file_id)
        self.assertEqual(retrieved_metadata.filename, self.sample_metadata.filename)
        self.assertEqual(retrieved_metadata.channel_count, self.sample_metadata.channel_count)
    
    def test_get_file_path(self):
        """Test getting the file path."""
        # Create a test file
        file_id = "test-file-id"
        file_path = self.storage_dir / f"{file_id}.mf4"
        with open(file_path, "wb") as f:
            f.write(b"Test file content")
        
        # Get the file path
        retrieved_path = self.file_storage.get_file_path(file_id)
        
        # Check that the retrieved path matches the original
        self.assertEqual(retrieved_path, file_path)
    
    def test_get_file_path_not_found(self):
        """Test getting the file path for a non-existent file."""
        # Get the file path for a non-existent file
        retrieved_path = self.file_storage.get_file_path("non-existent-file")
        
        # Check that None is returned
        self.assertIsNone(retrieved_path)
    
    def test_list_files(self):
        """Test listing files."""
        # Save the metadata
        self.file_storage.save_file_metadata(self.sample_metadata)
        
        # List the files
        files = self.file_storage.list_files()
        
        # Check that the list contains the sample metadata
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].file_id, self.sample_metadata.file_id)
        self.assertEqual(files[0].filename, self.sample_metadata.filename)
    
    def test_delete_file(self):
        """Test deleting a file."""
        # Create a test file
        file_id = "test-file-id"
        file_path = self.storage_dir / f"{file_id}.mf4"
        with open(file_path, "wb") as f:
            f.write(b"Test file content")
        
        # Save the metadata
        self.file_storage.save_file_metadata(self.sample_metadata)
        
        # Check that the file and metadata exist
        self.assertTrue(file_path.exists())
        metadata_path = self.metadata_dir / f"{file_id}.json"
        self.assertTrue(metadata_path.exists())
        
        # Delete the file
        result = self.file_storage.delete_file(file_id)
        
        # Check that the result is True
        self.assertTrue(result)
        
        # Check that the file and metadata were deleted
        self.assertFalse(file_path.exists())
        self.assertFalse(metadata_path.exists())
    
    def test_delete_file_not_found(self):
        """Test deleting a non-existent file."""
        # Delete a non-existent file
        result = self.file_storage.delete_file("non-existent-file")
        
        # Check that the result is False
        self.assertFalse(result)

    @pytest.mark.asyncio  
    async def test_save_uploaded_file_async(self):
        """Test saving an uploaded file asynchronously."""
        # Use pytest.mark.asyncio decorator instead of running async function directly
        with patch("uuid.uuid4") as mock_uuid4:
            # Mock the uuid4 function to return a known value
            mock_uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")
            mock_uuid4.return_value = mock_uuid
            
            # Create a mock uploaded file
            mock_file = MagicMock()
            mock_file.filename = "test_file.mf4"
            mock_file.read = MagicMock(return_value=b"Test file content")
            
            # Save the uploaded file
            file_id = await self.file_storage.save_uploaded_file(mock_file)
            
            # Check that the file ID is correct
            assert file_id == str(mock_uuid)
            
            # Check that the file was saved
            file_path = self.storage_dir / f"{file_id}.mf4"
            assert file_path.exists()
            
            # Check the file content
            with open(file_path, "rb") as f:
                content = f.read()
                assert content == b"Test file content"


# Run the tests
if __name__ == "__main__":
    unittest.main() 