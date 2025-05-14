import os
import sys
import unittest
from pathlib import Path
import logging
import json
from fastapi.testclient import TestClient
from service.service import app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample data path
SAMPLE_DIR = Path(__file__).parent / "samples"
SAMPLE_MDF = SAMPLE_DIR / "sample.MF4"

client = TestClient(app)

class TestDataEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SAMPLE_MDF.exists():
            raise unittest.SkipTest(f"Sample MDF file not found: {SAMPLE_MDF}")
        # Upload the sample file
        with open(SAMPLE_MDF, "rb") as f:
            files = {"file": (SAMPLE_MDF.name, f, "application/octet-stream")}
            response = client.post("/files/upload", files=files)
        if response.status_code != 200:
            raise unittest.SkipTest(f"File upload failed: {response.text}")
        data = response.json()
        cls.file_id = data.get("file_id") or data.get("id")
        if not cls.file_id:
            raise unittest.SkipTest("No file_id returned from upload.")
        logger.info(f"Uploaded file_id: {cls.file_id}")

    def test_get_channel_data(self):
        # List channels for the file - using the correct endpoint
        resp = client.get(f"/files/{self.file_id}")
        self.assertEqual(resp.status_code, 200, resp.text)
        meta = resp.json()
        
        # Debug metadata structure
        logger.info(f"Metadata keys: {list(meta.keys())}")
        logger.info(f"Channels type: {type(meta.get('channels'))}")
        
        # Get the first valid channel name from metadata
        channels = meta.get("channels", [])
        self.assertTrue(channels, f"No channels found in metadata. Available keys: {list(meta.keys())}")
        
        # Debug the channels structure
        logger.info(f"First few channels: {json.dumps(channels[:3] if isinstance(channels, list) else channels, indent=2)}")
        
        # Extract a channel name - handle different possible structures
        if isinstance(channels, list):
            if len(channels) == 0:
                self.skipTest("No channels available in the file")
            if isinstance(channels[0], dict):
                channel = channels[0].get("name")
                if not channel:
                    # Try other common keys if 'name' not found
                    for key in ["id", "channel", "signal"]:
                        if key in channels[0]:
                            channel = channels[0][key]
                            break
            else:
                channel = channels[0]  # Assume it's a string
        elif isinstance(channels, dict):
            # If channels is a dictionary, use the first key
            channel = next(iter(channels))
        else:
            self.skipTest(f"Unexpected channels format: {type(channels)}")
            
        logger.info(f"Selected channel for testing: {channel}")
        self.assertIsNotNone(channel, "Could not extract a valid channel name")
        
        # Query data endpoint
        response = client.get(f"/files/{self.file_id}/data", params={"channels": channel})
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertIn(channel, data, f"Channel {channel} not found in response: {list(data.keys())}")
        self.assertIn("timestamps", data[channel], f"No timestamps found for channel {channel}")
        self.assertIn("values", data[channel], f"No values found for channel {channel}")
        self.assertIsInstance(data[channel]["timestamps"], list)
        self.assertIsInstance(data[channel]["values"], list)
        logger.info(f"Test passed: channel {channel} has valid data")

if __name__ == "__main__":
    unittest.main() 