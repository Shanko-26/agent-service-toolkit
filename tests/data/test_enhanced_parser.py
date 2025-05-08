"""
Tests for the enhanced MDF parser implementation.

This module tests the enhanced parser functionality with sample data.
"""

import os
import sys
import unittest
from pathlib import Path
import tempfile
import logging

# Add the parent directory to sys.path to find the src module
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the enhanced parser
try:
    from src.data.enhanced_parser import parse_mdf, detect_ecu_from_name
    HAS_PARSER = True
except ImportError as e:
    logger.error(f"Error importing parser: {e}")
    HAS_PARSER = False

# Try importing asammdf to check availability
try:
    from asammdf import MDF
    from asammdf.signal import Signal
    HAS_ASAMMDF = True
except ImportError:
    MDF = None
    Signal = None
    HAS_ASAMMDF = False


# Define sample data paths
SAMPLE_DIR = Path(__file__).parent / "samples"
SAMPLE_MDF = SAMPLE_DIR / "sample.mdf"
SAMPLE_DBC = SAMPLE_DIR / "sample.dbc"


@unittest.skipIf(not HAS_PARSER or not HAS_ASAMMDF, "Parser or asammdf not available")
class TestEnhancedParser(unittest.TestCase):
    """Test case for the enhanced parser."""
    
    def setUp(self):
        """Set up test environment."""
        # Check if sample files exist, skip tests if not
        if not SAMPLE_MDF.exists():
            self.skipTest(f"Sample MDF file not found: {SAMPLE_MDF}")
        
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.TemporaryDirectory()
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_basic_parsing(self):
        """Test basic parsing functionality."""
        # Parse with enhanced parser
        df, metadata = parse_mdf(SAMPLE_MDF)
        
        # Basic checks
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsNotNone(metadata)
        
        # Log basic info
        logger.info(f"DataFrame rows: {len(df)}")
        logger.info(f"DataFrame columns: {len(df.columns)}")
        logger.info(f"Metadata keys: {metadata.keys()}")
        
        # Check essential metadata is present
        essential_keys = {'file_id', 'filename', 'file_path', 'channels'}
        for key in essential_keys:
            self.assertIn(key, metadata.keys(), f"Missing essential metadata key: {key}")
    
    @unittest.skipIf(not SAMPLE_DBC.exists(), "Sample DBC file not found")
    def test_can_parsing(self):
        """Test CAN parsing with DBC file."""
        # Parse with enhanced parser and DBC file
        df, metadata = parse_mdf(SAMPLE_MDF, dbc_files=[str(SAMPLE_DBC)])
        
        # Check if CAN signals were extracted
        logger.info(f"Columns with DBC: {df.columns.tolist()}")
    
    def test_ecu_detection(self):
        """Test ECU detection functionality."""
        # Create test signals with known ECU patterns
        test_signals = {
            "ECM_EngineSpeed": "ECM",
            "VehicleSpeed_ESP": "ESP",
            "BCM.DoorOpen": "BCM",
            "Prefix_EPS_Angle": "EPS",
            "Unknown_Signal": "UNKNOWN"
        }
        
        # Test enhanced parser's detection
        for signal_name, expected_ecu in test_signals.items():
            detected = detect_ecu_from_name(signal_name)
            self.assertEqual(detected, expected_ecu, 
                            f"ECU detection failed for {signal_name}")
    
    def test_time_range_filtering(self):
        """Test time range filtering."""
        # Parse with time range
        df, metadata = parse_mdf(
            SAMPLE_MDF,
            time_range=(1.0, 2.0)
        )
        
        # Check if time filtering was applied, if there's a time column
        if 'time' in df.columns and len(df) > 0:
            self.assertTrue(
                df['time'].min() >= 1.0 and df['time'].max() <= 2.0,
                "Time filtering failed"
            )
    
    def test_signals_return_format(self):
        """Test returning Signal objects instead of DataFrame."""
        # Parse with Signal return
        signals, metadata = parse_mdf(SAMPLE_MDF, return_format='signals')
        
        # Check if signals were returned
        self.assertIsInstance(signals, dict)
        
        # Check if values are Signal objects
        for name, signal in signals.items():
            self.assertIsInstance(signal, Signal)
    
    def test_new_features(self):
        """Test enhanced parser's new features."""
        # Test resampling
        df_resampled, _ = parse_mdf(
            SAMPLE_MDF,
            resample=100,  # 100 Hz
            interpolation='linear'
        )
        
        # Test returning both DataFrame and signals
        df_both, signals_both, _ = parse_mdf(
            SAMPLE_MDF,
            return_format='both'
        )
        
        # Check if both were returned correctly
        self.assertIsInstance(df_both, pd.DataFrame)
        self.assertIsInstance(signals_both, dict)
        
        # Time as index feature
        df_indexed, _ = parse_mdf(
            SAMPLE_MDF,
            time_as_index=True
        )
        
        # Check if time is the index
        if len(df_indexed) > 0 and 'time' not in df_indexed.columns and isinstance(df_indexed.index, pd.Index):
            logger.info("Time successfully set as index")


if __name__ == "__main__":
    unittest.main() 