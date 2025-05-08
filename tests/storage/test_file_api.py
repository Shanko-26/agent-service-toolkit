"""
Tests for the file management API endpoints.

This module contains tests for the file upload, listing, download,
and deletion endpoints.
"""

import os
import sys
import unittest
from pathlib import Path
import tempfile
import json
from unittest.mock import MagicMock, patch

# Add the parent directory to sys.path to find the src module
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import pytest
from fastapi import FastAPI, status, Depends
from fastapi.testclient import TestClient
from fastapi.encoders import jsonable_encoder
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# Import only what we need instead of the entire module
from src.storage.routes import router as files_router
from src.storage.file_storage import FileStorageManager
from src.schema.automotive.models import MeasurementFile, FileFormat, SignalMetadata
from src.data.enhanced_parser import parse_mdf, HAS_ASAMMDF

# Path to sample test data
SAMPLES_DIR = Path(__file__).parent.parent / "data" / "samples"

# Define our own verify_bearer function to avoid imports from src.service.service
async def verify_bearer():
    """Test version of verify_bearer."""
    return True

class TestFileAPI(unittest.TestCase):
    """Test case for file management API endpoints."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a test app with the files_router
        self.app = FastAPI()
        
        # Add our routes but override the security dependency
        self.app.include_router(
            files_router,
            dependencies=[]  # Remove the security dependency
        )
        
        # Mock the authentication dependency at the route level
        from src.storage import routes
        routes.verify_bearer = verify_bearer
        
        # Create a test client
        self.client = TestClient(self.app)
        
        # Create a temporary directory for the file storage
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_dir = Path(self.temp_dir.name) / "files"
        self.metadata_dir = Path(self.temp_dir.name) / "metadata"
        
        # Create a test storage manager with the temporary directories
        self.file_storage = FileStorageManager(
            storage_dir=str(self.storage_dir),
            metadata_dir=str(self.metadata_dir)
        )
        
        # Mock the file_storage singleton in the routes module
        from src.storage import routes
        self.original_file_storage = routes.file_storage
        routes.file_storage = self.file_storage
        
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
        # Restore the original file_storage
        from src.storage import routes
        routes.file_storage = self.original_file_storage
        
        # Clean up the temporary directory
        self.temp_dir.cleanup()
    
    def create_test_file(self, file_id="test-file-id", content=b"Test file content"):
        """Create a test file in the storage directory."""
        # Create the storage directory if it doesn't exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Create the file
        file_path = self.storage_dir / f"{file_id}.mf4"
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Save the metadata
        metadata_path = self.metadata_dir / f"{file_id}.json"
        with open(metadata_path, "w") as f:
            # Convert to dict and then to JSON
            json.dump(self.sample_metadata.model_dump(), f, indent=2, default=str)
        
        return file_path
    
    @patch("src.data.enhanced_parser.parse_mdf")
    def test_upload_file(self, mock_parse_mdf):
        """Test the file upload endpoint."""
        # Skip if asammdf is not available
        if not HAS_ASAMMDF:
            self.skipTest("asammdf is not available")
        
        # Mock the parse_mdf function
        mock_signals = {
            "Engine_Speed": MagicMock(
                name="Engine_Speed",
                unit="rpm",
                min=0.0,
                max=8000.0,
                comment="Engine rotational speed",
                timestamps=range(100),
                samples=MagicMock(dtype="float32")
            )
        }
        mock_parser_metadata = {
            "ecu_map": {"Engine_Speed": "ECM"},
            "signal_aliases": {},
            "redundant_signals": {}
        }
        mock_parse_mdf.return_value = (mock_signals, mock_parser_metadata)
        
        # Create a test file to upload
        test_content = b"This is a test MF4 file"
        
        # Send the upload request
        with tempfile.NamedTemporaryFile(suffix=".mf4") as temp_file:
            # Write the test content to the file
            temp_file.write(test_content)
            temp_file.flush()
            temp_file.seek(0)
            
            # Send the upload request
            files = {"file": ("test_file.mf4", temp_file, "application/octet-stream")}
            data = {"description": "Test upload"}
            response = self.client.post("/files/upload", files=files, data=data)
        
        # Check the response
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        response_data = response.json()
        self.assertEqual(response_data["status"], "ok")
        self.assertEqual(response_data["filename"], "test_file.mf4")
        self.assertIn("file_id", response_data)
        
        # Check that the file was saved
        file_id = response_data["file_id"]
        file_path = self.storage_dir / f"{file_id}.mf4"
        self.assertTrue(file_path.exists())
        
        # Check that the metadata was saved
        metadata_path = self.metadata_dir / f"{file_id}.json"
        self.assertTrue(metadata_path.exists())
    
    def test_list_files(self):
        """Test the file listing endpoint."""
        # Create a test file
        self.create_test_file()
        
        # Send the list request
        response = self.client.get("/files/list")
        
        # Check the response
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        response_data = response.json()
        self.assertEqual(len(response_data), 1)
        self.assertEqual(response_data[0]["file_id"], "test-file-id")
        self.assertEqual(response_data[0]["filename"], "test_file.mf4")
    
    def test_get_file_metadata(self):
        """Test the get file metadata endpoint."""
        # Create a test file
        self.create_test_file()
        
        # Send the get request
        response = self.client.get("/files/test-file-id")
        
        # Check the response
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        response_data = response.json()
        self.assertEqual(response_data["file_id"], "test-file-id")
        self.assertEqual(response_data["filename"], "test_file.mf4")
        self.assertEqual(response_data["channel_count"], 1)
        self.assertIn("Engine_Speed", response_data["channels"])
    
    def test_get_file_metadata_not_found(self):
        """Test the get file metadata endpoint with a non-existent file."""
        # Send the get request with a non-existent file ID
        response = self.client.get("/files/non-existent-file")
        
        # Check the response
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertIn("not found", response.json()["detail"])
    
    def test_download_file(self):
        """Test the file download endpoint."""
        # Create a test file
        test_content = b"Test file content"
        self.create_test_file(content=test_content)
        
        # Send the download request
        response = self.client.get("/files/test-file-id/download")
        
        # Check the response
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.content, test_content)
        
        # Check the headers
        self.assertEqual(response.headers["content-type"], "application/octet-stream")
        self.assertIn("test_file.mf4", response.headers["content-disposition"])
    
    def test_download_file_not_found(self):
        """Test the download endpoint with a non-existent file."""
        # Send the download request with a non-existent file ID
        response = self.client.get("/files/non-existent-file/download")
        
        # Check the response
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertIn("not found", response.json()["detail"])
    
    def test_delete_file(self):
        """Test the file deletion endpoint."""
        # Create a test file
        file_path = self.create_test_file()
        metadata_path = self.metadata_dir / "test-file-id.json"
        
        # Verify that the file and metadata exist
        self.assertTrue(file_path.exists())
        self.assertTrue(metadata_path.exists())
        
        # Send the delete request
        response = self.client.delete("/files/test-file-id")
        
        # Check the response
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("deleted successfully", response.json()["message"])
        
        # Verify that the file and metadata were deleted
        self.assertFalse(file_path.exists())
        self.assertFalse(metadata_path.exists())
    
    def test_delete_file_not_found(self):
        """Test the delete endpoint with a non-existent file."""
        # Send the delete request with a non-existent file ID
        response = self.client.delete("/files/non-existent-file")
        
        # Check the response
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertIn("not found", response.json()["detail"])


# Run the tests
if __name__ == "__main__":
    unittest.main() 