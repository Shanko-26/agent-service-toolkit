"""
Client for interacting with the file management API.
"""

import os
from typing import List, Optional
import httpx
from pydantic import ValidationError
from schema.automotive.models import MeasurementFile, FileUploadResponse

class FileClientError(Exception):
    """Exception raised when there is an error with the file client."""
    pass

class FileClient:
    """Client for interacting with the file management API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        timeout: float | None = None,
        auth_header: dict[str, str] | None = None,
    ) -> None:
        """
        Initialize the file client.

        Args:
            base_url (str): The base URL of the file management service.
            timeout (float, optional): The timeout for requests.
            auth_header (dict[str, str], optional): Additional headers to include in requests,
                such as Authorization header. Default: None
        """
        self.base_url = base_url
        self.auth_secret = os.getenv("AUTH_SECRET")
        self.timeout = timeout
        self.auth_header = auth_header or {}

    @property
    def _headers(self) -> dict[str, str]:
        headers = {}
        if self.auth_secret:
            headers["Authorization"] = f"Bearer {self.auth_secret}"
        # Add any additional headers
        headers.update(self.auth_header)
        return headers

    async def list_files(self) -> List[MeasurementFile]:
        """
        List all available files.

        Returns:
            List[MeasurementFile]: A list of available files and their metadata.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/files/list",
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                file_list = [MeasurementFile.model_validate(file) for file in response.json()]
                return file_list
        except (httpx.HTTPError, ValidationError) as e:
            raise FileClientError(f"Error listing files: {e}")

    async def get_file_metadata(self, file_id: str) -> MeasurementFile:
        """
        Get metadata for a specific file.

        Args:
            file_id (str): The ID of the file to get metadata for.

        Returns:
            MeasurementFile: Metadata for the requested file.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/files/{file_id}",
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                metadata = MeasurementFile.model_validate(response.json())
                return metadata
        except (httpx.HTTPError, ValidationError) as e:
            raise FileClientError(f"Error getting file metadata: {e}")

    async def upload_file(self, file_content: bytes, filename: str, description: Optional[str] = None) -> FileUploadResponse:
        """
        Upload a file to the server.

        Args:
            file_content (bytes): The content of the file to upload.
            filename (str): The name of the file.
            description (str, optional): A description of the file.

        Returns:
            FileUploadResponse: The response from the server.
        """
        try:
            # Create a temporary file to use with httpx
            temp_file_path = f"temp_{filename}"
            
            # Ensure content is not empty
            if not file_content or len(file_content) == 0:
                raise FileClientError(f"File content is empty for {filename}")
                
            print(f"Writing {len(file_content)} bytes to temporary file {temp_file_path}")
            
            # Write content to temporary file
            with open(temp_file_path, "wb") as f:
                f.write(file_content)
                f.flush()  # Ensure all data is written to disk
                
            # Verify the file was written correctly
            if not os.path.exists(temp_file_path):
                raise FileClientError(f"Failed to create temporary file {temp_file_path}")
                
            file_size = os.path.getsize(temp_file_path)
            if file_size == 0:
                raise FileClientError(f"Temporary file {temp_file_path} is empty (0 bytes)")
                
            print(f"Temporary file created successfully: {temp_file_path}, size: {file_size} bytes")
            
            # Use httpx with properly managed file handle
            async with httpx.AsyncClient() as client:
                with open(temp_file_path, "rb") as file_handle:
                    # Create a tuple with (filename, file_handle, content_type)
                    files = {"file": (filename, file_handle, "application/octet-stream")}
                    data = {}
                    if description:
                        data["description"] = description
                    
                    response = await client.post(
                        f"{self.base_url}/files/upload",
                        files=files,
                        data=data,
                        headers=self._headers,
                        timeout=self.timeout,
                    )
                # File handle is automatically closed here when exiting the with block
                
            response.raise_for_status()
            result = FileUploadResponse.model_validate(response.json())
            
            # Now safe to remove the file as all handles are closed
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except (PermissionError, OSError) as e:
                    print(f"Warning: Could not remove temporary file {temp_file_path}: {e}")
                    
            return result
            
        except (httpx.HTTPError, ValidationError) as e:
            # Attempt to clean up temporary file in case of error
            try:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
            except:
                pass
            raise FileClientError(f"Error uploading file: {e}")

    async def delete_file(self, file_id: str) -> dict:
        """
        Delete a file.

        Args:
            file_id (str): The ID of the file to delete.

        Returns:
            dict: The response from the server.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{self.base_url}/files/{file_id}",
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            raise FileClientError(f"Error deleting file: {e}") 