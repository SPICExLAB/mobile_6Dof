"""
Input_Utils package - Utilities for handling input from various devices.

This package provides utilities for:
1. Socket communication with devices (socket_utils)
2. Sensor data transformations and parsing (sensor_utils)
3. Calibration of IMU devices (sensor_calibrate)
"""

from .socket_utils import SocketReceiver, ApiServer
from .sensor_utils import (
    # Data types
    IMUData,
    
    # Transformation functions
    preprocess_headphone_data,
    preprocess_rokid_data,
    apply_gravity_compensation,
    calculate_euler_from_quaternion,
    
    # Parsing functions
    parse_ios_data,
    parse_phone_data,
    parse_headphone_data,
    parse_watch_data,
    parse_rokid_glasses_data
)

from .sensor_calibrate import (
    # Calibration class
    IMUCalibrator,
    
)

__all__ = [
    # Socket utilities
    'SocketReceiver',
    'ApiServer',
    
    # Data types
    'IMUData',
    
    # Sensor transformation utilities
    'preprocess_headphone_data',
    'preprocess_rokid_data',
    'apply_gravity_compensation',
    'calculate_euler_from_quaternion',
    
    # Parsing functions
    'parse_ios_data',
    'parse_phone_data',
    'parse_headphone_data',
    'parse_watch_data',
    'parse_rokid_glasses_data',
    
    # Calibration class
    'IMUCalibrator',
    
    # Calibration functions
    'apply_calibration_transform',
    'align_global_identity'
]