import os
from unittest.mock import patch
import pytest
import tempfile
import sys
from pathlib import Path

# Add the project root directory to Python's module search path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Path to test data directory containing sample MDF files
TEST_DATA_DIR = Path(__file__).parent / "data" / "samples"

def pytest_addoption(parser):
    parser.addoption(
        "--run-docker", action="store_true", default=False, help="run docker integration tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "docker: mark test as requiring docker containers")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-docker"):
        skip_docker = pytest.mark.skip(reason="need --run-docker option to run")
        for item in items:
            if "docker" in item.keywords:
                item.add_marker(skip_docker)


@pytest.fixture
def mock_env():
    """Fixture to ensure environment is clean for each test."""
    with patch.dict(os.environ, {}, clear=True):
        yield

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test file operations."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname

@pytest.fixture
def sample_mdf_path():
    """Return path to a sample MDF file for testing.
    
    Note: You will need to add a small sample MDF file to tests/data/samples/ directory
    """
    # Ensure the samples directory exists
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    
    # Path to a sample MDF file (to be added)
    sample_path = TEST_DATA_DIR / "sample.mdf"
    
    if not sample_path.exists():
        pytest.skip(f"Sample MDF file not found at {sample_path}. Tests requiring MDF files will be skipped.")
    
    return str(sample_path)

@pytest.fixture
def sample_dbc_path():
    """Return path to a sample DBC file for testing."""
    # Ensure the samples directory exists
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    
    # Path to a sample DBC file (to be added)
    sample_path = TEST_DATA_DIR / "sample.dbc"
    
    if not sample_path.exists():
        pytest.skip(f"Sample DBC file not found at {sample_path}. Tests requiring DBC files will be skipped.")
    
    return str(sample_path)
