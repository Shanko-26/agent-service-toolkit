"""
Pydantic models for automotive measurement data.

These models define the data structures for representing automotive measurement
data, including signal metadata and measurement file information.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from enum import Enum

from pydantic import BaseModel, Field, validator

class FileFormat(str, Enum):
    """Enumeration of supported measurement file formats."""
    MDF3 = "MDF3"
    MDF4 = "MDF4"
    MF4 = "MF4"  # Alias for MDF4
    MAT = "MAT"  # MATLAB format


class SignalMetadata(BaseModel):
    """Metadata for a signal/channel in a measurement file."""
    name: str = Field(..., description="Original signal name from the measurement file")
    qualified_name: Optional[str] = Field(None, description="Qualified name including ECU identifier")
    display_name: Optional[str] = Field(None, description="Human-readable name for display purposes")
    unit: Optional[str] = Field("", description="Physical unit of the signal")
    min_value: Optional[float] = Field(None, description="Minimum value in the signal data")
    max_value: Optional[float] = Field(None, description="Maximum value in the signal data")
    description: Optional[str] = Field("", description="Description of what the signal represents")
    sampling_rate: Optional[float] = Field(None, description="Samples per second")
    signal_id: str = Field(..., description="Unique identifier for the signal")
    ecu: Optional[str] = Field("UNKNOWN", description="ECU that produced this signal")
    data_type: Optional[str] = Field(None, description="Data type of the signal values")
    
    @validator('display_name', pre=True, always=True)
    def set_display_name(cls, v, values):
        """Set display_name to name if not provided."""
        if not v and 'name' in values:
            return values['name']
        return v


class SignalAlias(BaseModel):
    """Maps a simplified alias to a qualified signal name."""
    alias: str = Field(..., description="Simplified name/alias")
    qualified_name: str = Field(..., description="Full qualified signal name")
    description: Optional[str] = Field(None, description="Description of why this alias exists")


class RedundantSignalGroup(BaseModel):
    """Group of redundant signals from different ECUs."""
    base_name: str = Field(..., description="Base signal name without ECU prefix")
    signals: List[Dict[str, str]] = Field(..., description="List of ECU, name, qualified_name tuples")


class MeasurementFile(BaseModel):
    """Metadata for a measurement file."""
    file_id: str = Field(..., description="Unique identifier for the file")
    filename: str = Field(..., description="Original filename")
    file_path: Optional[str] = Field(None, description="Path to the file")
    file_type: FileFormat = Field(..., description="File format")
    start_time: Union[float, datetime] = Field(..., description="Start time of the measurement")
    end_time: Union[float, datetime] = Field(..., description="End time of the measurement")
    duration: Optional[float] = Field(None, description="Duration in seconds")
    channels: Dict[str, SignalMetadata] = Field(..., description="Dictionary of channel metadata")
    sample_count: int = Field(..., description="Number of samples in the file")
    channel_count: Optional[int] = Field(None, description="Number of channels in the file")
    file_size_bytes: int = Field(..., description="Size of the file in bytes")
    ecu_map: Optional[Dict[str, List[str]]] = Field(None, description="Map of ECUs to their signals")
    signal_aliases: Optional[Dict[str, str]] = Field(None, description="Map of aliases to qualified names")
    redundant_signals: Optional[Dict[str, List[Dict[str, str]]]] = Field(None, description="Groups of redundant signals")
    description: Optional[str] = Field(None, description="Description of the measurement file")
    
    @validator('channel_count', pre=True, always=True)
    def set_channel_count(cls, v, values):
        """Calculate channel_count from channels if not provided."""
        if not v and 'channels' in values:
            return len(values['channels'])
        return v
    
    @validator('duration', pre=True, always=True)
    def calculate_duration(cls, v, values):
        """Calculate duration from start_time and end_time if not provided."""
        if not v and 'start_time' in values and 'end_time' in values:
            start = values['start_time']
            end = values['end_time']
            
            # Convert datetime to float if needed
            if isinstance(start, datetime) and isinstance(end, datetime):
                return (end - start).total_seconds()
            elif isinstance(start, (int, float)) and isinstance(end, (int, float)):
                return end - start
        return v


class SignalSelectionRequest(BaseModel):
    """Request model for selecting signals from a measurement file."""
    file_id: str = Field(..., description="ID of the measurement file")
    channels: List[str] = Field(..., description="List of channel names or patterns to select")
    time_from: Optional[float] = Field(None, description="Start time for selection (seconds)")
    time_to: Optional[float] = Field(None, description="End time for selection (seconds)")
    resample: Optional[bool] = Field(False, description="Whether to resample the data")
    resample_rate: Optional[float] = Field(None, description="Resampling rate in Hz")


class FileUploadResponse(BaseModel):
    """Response model for file upload endpoint."""
    file_id: str = Field(..., description="Unique identifier for the uploaded file")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="Detected file type")
    channel_count: int = Field(..., description="Number of channels found")
    start_time: Union[float, datetime] = Field(..., description="Start time of the measurement")
    end_time: Optional[Union[float, datetime]] = Field(None, description="End time of the measurement")
    duration: Optional[float] = Field(None, description="Duration in seconds")
    size_bytes: int = Field(..., description="Size of the file in bytes")
    message: str = Field("File uploaded successfully", description="Status message")
    status: str = Field("ok", description="Status code")
    
    @validator('duration', pre=True, always=True)
    def calculate_duration(cls, v, values):
        """Calculate duration from start_time and end_time if not provided."""
        if not v and 'start_time' in values and 'end_time' in values:
            start = values['start_time']
            end = values['end_time']
            
            # Convert datetime to float if needed
            if isinstance(start, datetime) and isinstance(end, datetime):
                return (end - start).total_seconds()
            elif isinstance(start, (int, float)) and isinstance(end, (int, float)):
                return end - start
        return v 