"""
Simple test script to verify asammdf can be imported and used.
"""

import os
import sys
import site
import importlib

# Print Python and environment info
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path}")
print(f"User site packages: {site.getusersitepackages()}")

# Try importing asammdf
try:
    # Attempt import
    import asammdf
    print(f"asammdf successfully imported, version: {asammdf.__version__}")
    
    # Print asammdf path
    print(f"asammdf path: {importlib.util.find_spec('asammdf').origin}")
    
    # Test basic MDF functionality with our sample file
    from asammdf import MDF
    from pathlib import Path
    
    sample_file = Path("tests/data/samples/sample.mdf")
    if sample_file.exists():
        print(f"Sample file exists: {sample_file}")
        mdf = MDF(sample_file)
        print(f"MDF version: {mdf.version}")
        print(f"Number of channels: {len(mdf.channels_db)}")
        print(f"Channel names: {list(mdf.channels_db.keys())[:10]}")  # Print first 10 channel names
    else:
        print(f"Sample file not found: {sample_file}")
        
except ImportError as e:
    print(f"Failed to import asammdf: {e}")
except Exception as e:
    print(f"Error working with asammdf: {e}") 