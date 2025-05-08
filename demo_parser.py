"""
Demo script to show the usage of the enhanced MDF parser.

This script demonstrates how to use the enhanced parser with various options.
"""

import os
import sys
import logging
from pathlib import Path

# Add parent dir to path if running as script
if __name__ == "__main__":
    parent_dir = str(Path(__file__).parent)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the enhanced parser
from src.data.enhanced_parser import parse_mdf

def demo_basic_parsing(mdf_path):
    """Demonstrate basic parsing functionality."""
    logger.info("===== Basic Parsing =====")
    df, metadata = parse_mdf(mdf_path)
    
    logger.info(f"DataFrame shape: {df.shape}")
    logger.info(f"Number of channels: {metadata['channel_count']}")
    logger.info(f"File version: {metadata['mdf_version']}")
    
    # Show first few rows and columns
    if not df.empty:
        logger.info("First 5 rows of DataFrame:")
        logger.info(df.head(5).to_string())
    
    return df, metadata

def demo_signal_selection(mdf_path, channels):
    """Demonstrate selecting specific channels."""
    logger.info("===== Signal Selection =====")
    df, metadata = parse_mdf(mdf_path, channels=channels)
    
    logger.info(f"Selected columns: {df.columns.tolist()}")
    if not df.empty:
        logger.info("First 5 rows of selected channels:")
        logger.info(df.head(5).to_string())
    
    return df, metadata

def demo_time_range(mdf_path, time_from, time_to):
    """Demonstrate time range filtering."""
    logger.info(f"===== Time Range Filtering ({time_from}s - {time_to}s) =====")
    df, metadata = parse_mdf(mdf_path, time_range=(time_from, time_to))
    
    if 'time' in df.columns and not df.empty:
        logger.info(f"Time range: {df['time'].min()} - {df['time'].max()}")
    
    logger.info(f"DataFrame shape: {df.shape}")
    
    return df, metadata

def demo_resampling(mdf_path, resample_rate):
    """Demonstrate resampling to a consistent frequency."""
    logger.info(f"===== Resampling ({resample_rate} Hz) =====")
    df, metadata = parse_mdf(mdf_path, resample=resample_rate)
    
    if not df.empty:
        logger.info(f"Resampled shape: {df.shape}")
        
        # If time column exists, verify resampling
        if 'time' in df.columns and len(df) > 1:
            time_diffs = df['time'].diff().dropna()
            avg_diff = time_diffs.mean()
            expected_diff = 1.0 / resample_rate
            
            logger.info(f"Average time difference: {avg_diff:.5f}s")
            logger.info(f"Expected time difference: {expected_diff:.5f}s")
            logger.info(f"Difference: {abs(avg_diff - expected_diff):.5f}s")
    
    return df, metadata

def demo_signal_objects(mdf_path):
    """Demonstrate working with Signal objects."""
    logger.info("===== Signal Objects =====")
    signals, metadata = parse_mdf(mdf_path, return_format='signals')
    
    logger.info(f"Number of signals: {len(signals)}")
    
    # Show information about the first few signals
    for i, (name, signal) in enumerate(list(signals.items())[:5]):
        logger.info(f"Signal {i+1}: {name}")
        logger.info(f"  Unit: {signal.unit}")
        logger.info(f"  Samples: {len(signal.samples)}")
        if len(signal.samples) > 0:
            logger.info(f"  Min: {signal.samples.min() if hasattr(signal.samples, 'min') else 'N/A'}")
            logger.info(f"  Max: {signal.samples.max() if hasattr(signal.samples, 'max') else 'N/A'}")
        logger.info(f"  Comment: {signal.comment[:50] + '...' if len(signal.comment) > 50 else signal.comment}")
    
    return signals, metadata

def run_all_demos(mdf_path):
    """Run all demonstration functions."""
    logger.info(f"Demonstrating parser with file: {mdf_path}")
    
    # Basic parsing
    df, metadata = demo_basic_parsing(mdf_path)
    
    # Get available column names for signal selection
    columns = df.columns.tolist()
    if 'time' in columns:
        columns.remove('time')
    
    # Signal selection (use first 3 columns if available)
    selected_channels = columns[:3] if len(columns) >= 3 else columns
    demo_signal_selection(mdf_path, selected_channels)
    
    # Time range filtering - use the middle portion of the data
    if 'time' in df.columns and not df.empty:
        time_min = df['time'].min()
        time_max = df['time'].max()
        time_from = time_min + (time_max - time_min) * 0.25
        time_to = time_min + (time_max - time_min) * 0.75
        demo_time_range(mdf_path, time_from, time_to)
    
    # Resampling demonstration
    demo_resampling(mdf_path, 100)  # 100 Hz
    
    # Signal objects
    demo_signal_objects(mdf_path)

if __name__ == "__main__":
    # Find and use the sample MDF file
    sample_dir = Path("tests/data/samples")
    sample_file = sample_dir / "sample.mdf"
    
    if not sample_file.exists():
        logger.error(f"Sample file not found: {sample_file}")
        sys.exit(1)
    
    run_all_demos(sample_file) 