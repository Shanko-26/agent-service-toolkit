"""
Advanced MDF file parser for automotive measurement data.

This module provides enhanced utilities for parsing MDF (Measurement Data Format) files
utilizing the full capabilities of asammdf. It maintains support for ECU redundancy 
detection and signal disambiguation while adding optimized processing for various
signal types and complex data structures.

Usage:
    from src.data.enhanced_parser import parse_mdf
    
    # Basic usage
    df, metadata = parse_mdf('path/to/file.mdf')
    
    # With CAN database
    df, metadata = parse_mdf('path/to/file.mf4', dbc_files=['vehicle.dbc'])
    
    # Return as Signal objects
    signals, metadata = parse_mdf('path/to/file.mdf', return_format='signals')
    
    # With advanced options
    df, metadata = parse_mdf(
        'path/to/file.mdf',
        channels=['Engine_Speed', 'Vehicle_Speed'],
        time_range=(10.5, 20.5),
        resample=100,  # Hz
        interpolation='linear'
    )
"""

import os
import re
import uuid
import sys
import site
import logging
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Set

# Setup Python path to find asammdf in different environments
def setup_environment():
    """Setup the environment to find asammdf whether in venv or user site-packages."""
    # Add user site-packages to path
    user_site = site.getusersitepackages()
    if user_site not in sys.path:
        sys.path.append(user_site)
    
    # Try to find asammdf in various places
    potential_locations = [
        # Current directory (if installed in venv)
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'),
        # Global site-packages 
        site.getsitepackages()[0] if site.getsitepackages() else None,
        # User site-packages
        user_site,
        # Append parent directory (for direct imports)
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ]
    
    # Add potential locations to path if they exist and aren't already in path
    for location in potential_locations:
        if location and os.path.exists(location) and location not in sys.path:
            sys.path.append(location)

# Setup environment
setup_environment()

import numpy as np
import pandas as pd

# Use try/except with detailed error handling to provide better diagnostics
try:
    import asammdf
    from asammdf import MDF
    from asammdf.signal import Signal
    try:
        from asammdf.blocks.utils import extract_cncomment_xml
    except ImportError:
        extract_cncomment_xml = None
        
    HAS_ASAMMDF = True
    logger = logging.getLogger(__name__)
    logger.info(f"Successfully imported asammdf version {asammdf.__version__}")
    logger.info(f"asammdf path: {importlib.util.find_spec('asammdf').origin}")
except ImportError as e:
    logger.error(f"Error importing asammdf: {e}")
    logger.error("asammdf package is not installed. Please install it using: pip install asammdf")
    MDF = None
    Signal = None
    extract_cncomment_xml = None
    HAS_ASAMMDF = False
except Exception as e:
    logger.error(f"Unexpected error importing asammdf: {e}")
    MDF = None
    Signal = None
    extract_cncomment_xml = None
    HAS_ASAMMDF = False

# Configure logging
logger = logging.getLogger(__name__)

# Regular expressions for ECU identification in signal names
ECU_PATTERNS = [
    r'^([A-Z]{2,4})_',           # Pattern: "ECU_SignalName"
    r'^([A-Za-z0-9]+)\.',        # Pattern: "ECU.SignalName"
    r'_([A-Z]{2,4})$',           # Pattern: "SignalName_ECU"
    r'_([A-Z]{2,4})_[A-Za-z]',   # Pattern: "Prefix_ECU_SignalName"
]

# Known redundant ECU pairs (primary: backup)
REDUNDANT_ECU_PAIRS = {
    'EPS': 'EPSR',   # Electric Power Steering
    'ESC': 'ESCR',   # Electronic Stability Control
    'BCM': 'BCMR',   # Body Control Module
    'PCM': 'PCMR',   # Powertrain Control Module
    'ACM': 'ACMR',   # Airbag Control Module
}

# Common ECU names to look for in signal names
COMMON_ECUS = ['ECM', 'TCM', 'ABS', 'ESP', 'EPS', 'BCM', 'RCM', 'PCM', 'HCM', 'BMS', 'ACM']


def detect_ecu_from_name(name: str) -> str:
    """
    Detect ECU name from a signal name using pattern matching.
    
    Args:
        name: Signal name to analyze
        
    Returns:
        Detected ECU name or 'UNKNOWN'
    """
    # Check for ECUs at end of name first (to handle 'SignalName_ECU' pattern)
    match = re.search(r'_([A-Z]{2,4})$', name)
    if match:
        return match.group(1).upper()
    
    # Try other patterns
    for pattern in ECU_PATTERNS:
        match = re.search(pattern, name)
        if match:
            return match.group(1).upper()
    
    # Check for common ECU names embedded in the signal name
    for ecu in COMMON_ECUS:
        if ecu in name.upper():
            return ecu
    
    return 'UNKNOWN'


def get_qualified_name(ecu: str, name: str) -> str:
    """
    Create a qualified name combining ECU and signal name.
    
    Args:
        ecu: ECU identifier
        name: Original signal name
        
    Returns:
        Qualified name in format 'ECU.SignalName' or original if ECU unknown
    """
    if ecu == 'UNKNOWN':
        return name
    return f"{ecu}.{name}"


def get_signal_group_info(mdf: MDF) -> dict:
    """
    Get information about channel groups and their parent groups.
    
    Args:
        mdf: MDF object
        
    Returns:
        Dictionary with signal group information
    """
    groups = {}
    
    try:
        for i, group in enumerate(mdf.groups):
            if not group.channels:
                continue
                
            groups[i] = {
                'name': group.name if hasattr(group, 'name') else f"Group_{i}",
                'comment': group.comment if hasattr(group, 'comment') else "",
                'channel_count': len(group.channels),
                'first_sample': group.channel_group.first_sample_position if hasattr(group.channel_group, 'first_sample_position') else 0,
                'channel_names': [ch.name for ch in group.channels]
            }
    except Exception as e:
        logger.warning(f"Error extracting group info: {e}")
    
    return groups


def extract_mdf_metadata(mdf: MDF, file_path: Path) -> dict:
    """
    Extract comprehensive metadata from an MDF file.
    
    Args:
        mdf: MDF object
        file_path: Path to the MDF file
        
    Returns:
        Dictionary containing file and channel metadata
    """
    file_id = str(uuid.uuid4())
    
    # Basic file metadata
    metadata = {
        "file_id": file_id,
        "filename": file_path.name,
        "file_path": str(file_path),
        "file_size_bytes": file_path.stat().st_size,
        "mdf_version": mdf.version,
        "start_time": mdf.header.start_time.timestamp() if hasattr(mdf.header, 'start_time') else None,
        "channel_count": len(mdf.channels_db),
        "channel_group_count": len(mdf.groups),
        "author": mdf.header.author if hasattr(mdf.header, 'author') else None,
        "department": mdf.header.department if hasattr(mdf.header, 'department') else None,
        "project": mdf.header.project if hasattr(mdf.header, 'project') else None,
        "subject": mdf.header.subject if hasattr(mdf.header, 'subject') else None,
    }
    
    # Add comments if available
    if hasattr(mdf, 'comment'):
        metadata["file_comment"] = mdf.comment
    
    # Add channel group information
    metadata["channel_groups"] = get_signal_group_info(mdf)
    
    # Extract attachments information for MDF4
    if mdf.version >= '4.00' and hasattr(mdf, 'attachments'):
        try:
            attachments = []
            for attachment in mdf.attachments:
                attachments.append({
                    "filename": attachment.file_name,
                    "mime": attachment.mime,
                    "comment": attachment.comment,
                    "size": len(attachment.data) if hasattr(attachment, 'data') else 0
                })
            metadata["attachments"] = attachments
        except Exception as e:
            logger.warning(f"Error extracting attachments: {e}")
    
    return metadata


def extract_channel_metadata(signals: Dict[str, Signal]) -> Dict[str, Dict[str, Any]]:
    """
    Extract metadata for each signal/channel.
    
    Args:
        signals: Dictionary mapping signal names to Signal objects
        
    Returns:
        Dictionary with metadata for each signal
    """
    metadata = {}
    
    for name, signal in signals.items():
        # Basic signal metadata
        signal_meta = {
            "name": signal.name,
            "unit": signal.unit,
            "comment": signal.comment,
            "samples": len(signal.samples),
            "data_type": str(signal.samples.dtype),
        }
        
        # Detect ECU from name
        ecu = detect_ecu_from_name(signal.name)
        signal_meta["ecu"] = ecu
        
        # Add min/max for numeric types
        if np.issubdtype(signal.samples.dtype, np.number) and len(signal.samples) > 0:
            signal_meta["min_value"] = float(np.min(signal.samples))
            signal_meta["max_value"] = float(np.max(signal.samples))
            
            # Add additional statistics for numeric signals
            signal_meta["mean"] = float(np.mean(signal.samples))
            signal_meta["std_dev"] = float(np.std(signal.samples))
        
        # Add any XML metadata if available
        if extract_cncomment_xml and signal.comment and '<CNcomment' in signal.comment:
            try:
                xml_info = extract_cncomment_xml(signal.comment)
                if xml_info:
                    signal_meta["xml_metadata"] = xml_info
            except Exception as e:
                logger.debug(f"Error parsing XML comment for {name}: {e}")
        
        metadata[name] = signal_meta
    
    return metadata


def process_can_signals(mdf: MDF, dbc_files: List[str] = None) -> Dict[str, Signal]:
    """
    Process CAN signals using DBC files if provided.
    
    Args:
        mdf: MDF object
        dbc_files: List of DBC file paths
        
    Returns:
        Dictionary of extracted CAN signals
    """
    signals = {}
    
    # If no DBC files, return empty dict
    if not dbc_files:
        return signals
    
    # Process each DBC file
    for dbc_file in dbc_files:
        if not os.path.exists(dbc_file):
            logger.warning(f"DBC file not found: {dbc_file}")
            continue
            
        try:
            logger.info(f"Extracting CAN signals using DBC: {dbc_file}")
            # Use asammdf's built-in extraction
            extracted = mdf.extract_can_logging(dbc_file)
            
            # Add extracted signals to our collection
            if extracted:
                # Add ECU information to signal names
                for name, signal in extracted.items():
                    # For CAN signals from DBC files, we can often extract the ECU
                    # This might be available in the signal comment or via other means
                    ecu = detect_ecu_from_name(name)
                    qualified_name = get_qualified_name(ecu, name)
                    
                    # Update signal name
                    signal.name = qualified_name
                    signals[qualified_name] = signal
                
                logger.info(f"Extracted {len(extracted)} CAN signals from {dbc_file}")
        except Exception as e:
            logger.error(f"Error extracting CAN signals from {dbc_file}: {e}")
    
    return signals


def handle_structured_signal(signal: Signal) -> List[Signal]:
    """
    Process structured array signals into individual signals.
    
    Args:
        signal: Signal object with potentially structured data
        
    Returns:
        List of Signal objects
    """
    # If not a structured array, return the original signal
    if signal.samples.dtype.names is None:
        return [signal]
    
    result = []
    
    # Process each field in the structured array
    for field in signal.samples.dtype.names:
        try:
            # Extract field data
            field_data = signal.samples[field]
            
            # Handle multi-dimensional arrays
            if field_data.ndim > 1:
                # For byte arrays in CAN frames, convert to hex strings
                if field.lower() in ('databytes', 'data', 'bytes', 'payload'):
                    field_data = np.array([bytes(row).hex() for row in field_data])
                else:
                    # Skip other multi-dimensional fields for now
                    continue
            
            # Create a new signal for this field
            field_signal = Signal(
                samples=field_data,
                timestamps=signal.timestamps.copy(),
                name=f"{signal.name}.{field}",
                unit=signal.unit,
                comment=f"Field {field} from {signal.name}"
            )
            
            result.append(field_signal)
            
        except Exception as e:
            logger.warning(f"Error processing field {field} in {signal.name}: {e}")
    
    return result


def extract_channels(
    mdf: MDF, 
    channels: List[str] = None,
    include_master: bool = True
) -> Dict[str, Signal]:
    """
    Extract channels from MDF file with ECU detection.
    
    Args:
        mdf: MDF object
        channels: List of channel names to extract (all if None)
        include_master: Whether to include master/time channels
        
    Returns:
        Dictionary of signal name to Signal object
    """
    extracted = {}
    
    # Get all available channels if none specified
    if not channels:
        available_channels = [ch for ch in mdf.channels_db.keys() if not ch.startswith('$')]
    else:
        # Find exact matches and partial matches
        available_channels = []
        channel_set = set(channels)
        db_channels = set(mdf.channels_db.keys())
        
        # Exact matches
        exact_matches = channel_set.intersection(db_channels)
        available_channels.extend(exact_matches)
        
        # Partial matches for remaining channels
        for req_ch in channel_set - exact_matches:
            partial_matches = [ch for ch in db_channels if req_ch.lower() in ch.lower()]
            available_channels.extend(partial_matches)
    
    # Process each channel
    for channel_name in available_channels:
        # Skip non-data/hidden channels
        if channel_name.startswith('$') and not include_master:
            continue
            
        try:
            # Extract signal using asammdf's API
            signal = None
            
            # Handle multiple occurrences
            if channel_name in mdf.channels_db:
                occurrences = mdf.channels_db[channel_name]
                if len(occurrences) > 1:
                    # Use first occurrence by default
                    group, index = occurrences[0]
                    signal = mdf.get(channel_name, group=group, index=index)
                else:
                    signal = mdf.get(channel_name)
            else:
                continue
                
            if signal is None:
                continue
                
            # Process structured arrays (like CAN frames)
            if hasattr(signal.samples, 'dtype') and (
                signal.samples.dtype.names is not None or signal.samples.ndim > 1
            ):
                processed_signals = handle_structured_signal(signal)
                
                # Add each field signal to results
                for field_signal in processed_signals:
                    # Detect ECU and create qualified name
                    ecu = detect_ecu_from_name(field_signal.name)
                    qualified_name = get_qualified_name(ecu, field_signal.name)
                    
                    # Update signal name
                    field_signal.name = qualified_name
                    extracted[qualified_name] = field_signal
            else:
                # Regular signal processing
                ecu = detect_ecu_from_name(channel_name)
                qualified_name = get_qualified_name(ecu, channel_name)
                
                # Update signal name
                signal.name = qualified_name
                extracted[qualified_name] = signal
                
        except Exception as e:
            logger.warning(f"Error extracting channel {channel_name}: {e}")
    
    return extracted


def resample_signals(
    signals: Dict[str, Signal], 
    raster: Union[float, np.ndarray],
    method: str = 'linear'
) -> Dict[str, Signal]:
    """
    Resample all signals to a common time base using asammdf's built-in methods.
    
    Args:
        signals: Dictionary of signals
        raster: Either sample rate in Hz or array of timestamps
        method: Interpolation method ('linear', 'previous', 'next')
        
    Returns:
        Dictionary of resampled signals
    """
    if not signals:
        return {}
        
    # Create timestamp raster if numeric frequency provided
    if isinstance(raster, (int, float)):
        # Find signal with longest time range
        start_time = float('inf')
        end_time = 0
        
        for signal in signals.values():
            if len(signal.timestamps) > 0:
                start_time = min(start_time, signal.timestamps[0])
                end_time = max(end_time, signal.timestamps[-1])
        
        # Create evenly spaced timestamps
        frequency = float(raster)
        timestamps = np.arange(start_time, end_time, 1/frequency)
    else:
        # Use provided timestamps
        timestamps = raster
    
    # Resample each signal
    resampled = {}
    for name, signal in signals.items():
        try:
            # Check Signal.interp signature to see if it accepts 'method' parameter
            # This accommodates changes in asammdf API between versions
            import inspect
            interp_params = inspect.signature(signal.interp).parameters
            
            if 'method' in interp_params:
                # Newer asammdf versions accept 'method' parameter
                resampled_signal = signal.interp(timestamps, method=method)
            else:
                # Older versions don't have 'method' parameter
                # Try using the 'kind' parameter for older versions or default behavior
                try:
                    resampled_signal = signal.interp(timestamps, kind=method)
                except TypeError:
                    # Fallback to default with no interpolation method
                    resampled_signal = signal.interp(timestamps)
                    
            resampled[name] = resampled_signal
        except Exception as e:
            logger.warning(f"Error resampling signal {name}: {e}")
            # Keep original signal if resampling fails
            resampled[name] = signal
    
    return resampled


def signals_to_dataframe(
    signals: Dict[str, Signal], 
    include_time: bool = True,
    time_as_index: bool = False
) -> pd.DataFrame:
    """
    Convert Signal objects to a pandas DataFrame.
    
    Args:
        signals: Dictionary of signals
        include_time: Whether to include time column
        time_as_index: Whether to use time as DataFrame index
        
    Returns:
        DataFrame with all signals
    """
    if not signals:
        logger.info("No signals provided to convert to DataFrame")
        return pd.DataFrame()
    
    # Find common timebase
    signal_values = list(signals.values())
    if not signal_values:
        logger.info("Empty signals dictionary")
        return pd.DataFrame()
        
    # Use timestamps from first signal as reference
    reference_timestamps = signal_values[0].timestamps
    reference_length = len(reference_timestamps)
    logger.info(f"Reference timestamps length: {reference_length}")
    
    # Create dataframe
    data = {}
    if include_time:
        data['time'] = reference_timestamps
    
    # Check signal lengths to ensure consistent lengths
    signal_lengths = {name: len(signal.samples) for name, signal in signals.items()}
    consistent_length = all(length == reference_length for length in signal_lengths.values())
    
    if not consistent_length:
        # Log signal lengths with issues
        logger.warning(f"Inconsistent signal lengths detected:")
        for name, length in signal_lengths.items():
            if length != reference_length:
                logger.warning(f"  Signal {name}: {length} samples (expected {reference_length})")
        
        # Handle signals with inconsistent lengths
        logger.info("Adjusting signals to have consistent lengths")
        for name, signal in signals.items():
            if len(signal.samples) != reference_length:
                # Skip signals with inconsistent lengths
                logger.warning(f"Skipping signal {name} due to length mismatch")
                continue
    
    # Add each signal, ensuring each is 1-dimensional and has matching length
    valid_signals = 0
    for name, signal in signals.items():
        # Skip time signals
        if name.lower() in ('time', 'timestamp'):
            continue
        
        # Skip signals with inconsistent lengths
        if len(signal.samples) != reference_length:
            continue
            
        # Add signal data, but handle multi-dimensional arrays
        try:
            if isinstance(signal.samples, np.ndarray):
                if signal.samples.ndim > 1:
                    # For multi-dimensional arrays, convert to string representation or flatten
                    if signal.samples.shape[1] <= 10:  # Small dimensions - convert to string
                        data[name] = np.array([str(row) for row in signal.samples])
                    else:
                        # For large arrays, use first element or other summarization technique
                        logger.info(f"Flattening multi-dimensional array for {name} with shape {signal.samples.shape}")
                        data[name] = signal.samples[:, 0]  # Use first column
                else:
                    # Regular 1D data
                    data[name] = signal.samples
            else:
                # Non-numpy data, convert to string
                data[name] = np.array([str(x) for x in signal.samples])
            
            valid_signals += 1
        except Exception as e:
            logger.warning(f"Error processing signal {name}: {e}")
    
    logger.info(f"Successfully added {valid_signals} signals to DataFrame")
    
    # Check all arrays are same length as a final validation
    array_lengths = {k: len(v) for k, v in data.items() if hasattr(v, '__len__')}
    if len(set(array_lengths.values())) > 1:
        logger.error(f"Array length mismatch: {array_lengths}")
        # Keep only arrays with the most common length
        if array_lengths:
            from collections import Counter
            length_counts = Counter(array_lengths.values())
            most_common_length = length_counts.most_common(1)[0][0]
            logger.info(f"Keeping only arrays with length {most_common_length}")
            data = {k: v for k, v in data.items() if not hasattr(v, '__len__') or len(v) == most_common_length}
    
    # Create the DataFrame from valid data only
    try:
        if data:
            df = pd.DataFrame(data)
            logger.info(f"Created DataFrame with shape {df.shape}")
        else:
            logger.warning("No valid data for DataFrame, returning empty DataFrame")
            df = pd.DataFrame()
    except ValueError as e:
        logger.error(f"Error creating DataFrame: {e}")
        # Try creating DataFrame with only the time column if it exists
        if 'time' in data:
            try:
                logger.info("Attempting to create DataFrame with only time column")
                df = pd.DataFrame({'time': data['time']})
            except Exception:
                df = pd.DataFrame()
        else:
            # Fallback to empty DataFrame if conversion fails
            df = pd.DataFrame()
    
    # Set time as index if requested
    if time_as_index and 'time' in df.columns and not df.empty:
        df.set_index('time', inplace=True)
    
    return df


def detect_redundant_signals(signals: Dict[str, Signal]) -> Dict[str, List[Dict[str, str]]]:
    """
    Detect redundant signals across ECUs.
    
    Args:
        signals: Dictionary of signals
        
    Returns:
        Dictionary mapping base signal names to lists of redundant instances
    """
    # Organize signals by ECU
    ecu_signals = {}
    for name, signal in signals.items():
        ecu = detect_ecu_from_name(name)
        if ecu not in ecu_signals:
            ecu_signals[ecu] = []
        ecu_signals[ecu].append(name)
    
    # Look for similar signal names across ECUs
    redundant_groups = {}
    
    # Extract base names (removing ECU prefixes/suffixes)
    signal_base_names = {}
    for name in signals.keys():
        # Remove ECU prefixes like "ECU."
        base_name = re.sub(r'^[A-Z]{2,4}\.', '', name)
        # Remove ECU suffixes like "_ECU"
        base_name = re.sub(r'_[A-Z]{2,4}$', '', base_name)
        
        if base_name not in signal_base_names:
            signal_base_names[base_name] = []
        signal_base_names[base_name].append(name)
    
    # Find redundant signals (same base name from different ECUs)
    for base_name, signal_names in signal_base_names.items():
        if len(signal_names) > 1:
            # Check if signals come from different ECUs
            ecus = set(detect_ecu_from_name(name) for name in signal_names)
            if len(ecus) > 1:
                redundant_groups[base_name] = [
                    {
                        "ecu": detect_ecu_from_name(name),
                        "name": name,
                        "is_primary": detect_ecu_from_name(name) not in REDUNDANT_ECU_PAIRS.values()
                    }
                    for name in signal_names
                ]
    
    return redundant_groups


def handle_time_range(
    signals: Dict[str, Signal],
    time_from: Optional[float] = None,
    time_to: Optional[float] = None
) -> Dict[str, Signal]:
    """
    Filter signals to specified time range.
    
    Args:
        signals: Dictionary of signals
        time_from: Start time (seconds)
        time_to: End time (seconds)
        
    Returns:
        Dictionary of filtered signals
    """
    if not signals or (time_from is None and time_to is None):
        return signals
    
    filtered = {}
    
    for name, signal in signals.items():
        try:
            # Create a time range mask
            timestamps = signal.timestamps
            mask = np.ones_like(timestamps, dtype=bool)
            
            if time_from is not None:
                mask = np.logical_and(mask, timestamps >= time_from)
                
            if time_to is not None:
                mask = np.logical_and(mask, timestamps <= time_to)
            
            # Skip if no samples remain
            if not np.any(mask):
                logger.warning(f"No samples remain for {name} after time filtering")
                continue
                
            # Create filtered signal
            filtered_signal = Signal(
                samples=signal.samples[mask],
                timestamps=timestamps[mask],
                name=signal.name,
                unit=signal.unit,
                comment=signal.comment
            )
            
            filtered[name] = filtered_signal
            
        except Exception as e:
            logger.warning(f"Error filtering time range for {name}: {e}")
            # Keep original if filtering fails
            filtered[name] = signal
    
    return filtered


def parse_mdf(
    file_path: Union[str, Path],
    channels: Optional[List[str]] = None,
    time_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
    dbc_files: Optional[List[str]] = None,
    resample: Optional[Union[float, np.ndarray]] = None,
    interpolation: str = 'linear',
    return_format: str = 'dataframe',
    time_as_index: bool = False
) -> Union[
    Tuple[pd.DataFrame, Dict[str, Any]],
    Tuple[Dict[str, Signal], Dict[str, Any]],
    Tuple[pd.DataFrame, Dict[str, Signal], Dict[str, Any]]
]:
    """
    Advanced MDF parser with comprehensive options and optimized processing.
    
    This function parses MDF files with full support for:
    - ECU detection and disambiguation
    - Redundant signal identification
    - CAN signal extraction using DBC files
    - Structured array handling
    - Signal resampling and interpolation
    - Automatic timebase alignment
    - Rich metadata extraction
    
    Args:
        file_path: Path to MDF file
        channels: List of specific channels to extract (all data channels if None)
        time_range: Optional tuple of (start_time, end_time) in seconds
        dbc_files: List of DBC files for CAN signal extraction
        resample: Target sample rate in Hz or array of timestamps for resampling
        interpolation: Method for signal interpolation ('linear', 'previous', 'next')
        return_format: Format for return values ('dataframe', 'signals', or 'both')
        time_as_index: Whether to use time as DataFrame index
        
    Returns:
        Based on return_format:
        - 'dataframe': (DataFrame, metadata dict)
        - 'signals': (Signal dict, metadata dict)
        - 'both': (DataFrame, Signal dict, metadata dict)
    """
    if not HAS_ASAMMDF:
        logger.error("asammdf package is not installed. Please install it using: pip install asammdf")
        raise ImportError("asammdf is required for parsing MDF files")
    
    # Ensure file_path is a Path object
    file_path = Path(file_path) if not isinstance(file_path, Path) else file_path
    if not file_path.exists():
        raise FileNotFoundError(f"MDF file not found: {file_path}")
    
    # Process time range if provided
    time_from = None
    time_to = None
    if time_range:
        time_from, time_to = time_range
    
    # Load the MDF file
    mdf = MDF(file_path)
    
    # Extract file metadata
    metadata = extract_mdf_metadata(mdf, file_path)
    
    # Initialize signals dictionary
    all_signals = {}
    
    # Process CAN signals if DBC files provided
    if dbc_files:
        can_signals = process_can_signals(mdf, dbc_files)
        all_signals.update(can_signals)
    
    # Extract requested channels
    extracted_signals = extract_channels(mdf, channels)
    all_signals.update(extracted_signals)
    
    # If no signals were found but channels were requested, try a different approach
    if not all_signals and channels:
        logger.info("No signals found with exact matching, trying fuzzy matching")
        # Try extracting all channels and then filter
        all_channels = extract_channels(mdf)
        
        # Filter to channels that contain requested names
        for req_ch in channels:
            matching = {name: signal for name, signal in all_channels.items() 
                      if req_ch.lower() in name.lower()}
            all_signals.update(matching)
    
    # Apply time range filtering if specified
    if time_from is not None or time_to is not None:
        all_signals = handle_time_range(all_signals, time_from, time_to)
    
    # Detect redundant signals
    redundant_signals = detect_redundant_signals(all_signals)
    
    # Extract channel metadata
    channel_metadata = extract_channel_metadata(all_signals)
    
    # Update metadata with channel info and redundancy info
    metadata.update({
        "channels": channel_metadata,
        "redundant_signals": redundant_signals,
        "signal_count": len(all_signals),
        "time_range": {
            "start": float(min(s.timestamps[0] for s in all_signals.values() if len(s.timestamps) > 0))
            if all_signals else None,
            "end": float(max(s.timestamps[-1] for s in all_signals.values() if len(s.timestamps) > 0))
            if all_signals else None
        }
    })
    
    # Apply resampling if requested
    if resample is not None and all_signals:
        all_signals = resample_signals(all_signals, resample, interpolation)
    
    # Return based on requested format
    if return_format == 'signals':
        return all_signals, metadata
    
    # Convert to DataFrame
    df = signals_to_dataframe(all_signals, time_as_index=time_as_index)
    
    if return_format == 'both':
        return df, all_signals, metadata
    
    # Default 'dataframe' format
    return df, metadata 