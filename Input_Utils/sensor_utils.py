"""
Input_Utils/sensor_utils.py - Coordinate transformations and parsing

This module provides utilities for transforming IMU data and parsing data from
various device types.

Device coordinate systems:
- Phone/Watch:   X: Right, Y: Up, Z: Toward user (out from screen, Backward)
- Headphone:     X: Right, Y: Forward, Z: Up
- Rokid Glasses: X: Right, Y: Up, Z: Forward

Global coordinate system:
- X: Left
- Y: Up
- Z: Forward
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class IMUData:
    """IMU data structure for iOS devices and AR glasses"""
    timestamp: float
    device_id: str
    accelerometer: np.ndarray  # [ax, ay, az] in m/s²
    gyroscope: np.ndarray      # [gx, gy, gz] in rad/s
    quaternion: np.ndarray     # [x, y, z, w] orientation
    euler: np.ndarray = None   # [nod, tilt, turn] for Rokid glasses


def preprocess_phone_watch_data(quaternion, acceleration, gyroscope=None):
    """
    Transform phone/watch quaternion, acceleration and gyroscope to represent rotation in world frame.
    Phone/Watch frame: X:right, Y:up, Z:toward user (out of screen)
    World frame: X:left, Y:up, Z:forward (into screen)
    """
    # Get current device rotation
    device_rotation = R.from_quat(quaternion)
    
    # Define the transformation that aligns coordinate systems
    # Phone/Watch (X:right, Y:up, Z:toward) → World (X:left, Y:up, Z:forward)
    align_transform = R.from_matrix([
        [-1, 0, 0],   # X: right → left
        [0, 1, 0],    # Y: up → up
        [0, 0, -1]    # Z: toward → forward (negate)
    ])
    
    # Apply transformation to get rotation in world frame
    world_rotation = align_transform * device_rotation * align_transform.inv()
    
    # Extract Euler angles to correct rotation directions
    euler = world_rotation.as_euler('xyz', degrees=True)
    euler[0] = -euler[0]
    euler[1] = -euler[1]
    
    # Create new rotation from adjusted euler angles
    world_rotation = R.from_euler('xyz', euler, degrees=True)
    world_quaternion = world_rotation.as_quat()
    
    # Transform acceleration to world frame
    world_acceleration = np.array([
        -acceleration[0],  # X: right → left
        acceleration[1],   # Y: up → up
        -acceleration[2]   # Z: toward → forward (negate)
    ])
    
    # Transform gyroscope to world frame if provided
    world_gyroscope = None
    if gyroscope is not None:
        # Gyroscope represents angular velocity, which also needs transformation
        world_gyroscope = np.array([
            -gyroscope[0],  # X: right → left (negate)
            gyroscope[1],   # Y: up → up
            -gyroscope[2]   # Z: toward → forward (negate)
        ])
    
    if gyroscope is not None:
        return world_quaternion, world_acceleration, world_gyroscope
    else:
        return world_quaternion, world_acceleration


def preprocess_headphone_data(quaternion, acceleration, gyroscope=None):
    """
    Transform headphone quaternion, acceleration and gyroscope to represent rotation in world frame.
    Headphone frame: X:right, Y:forward, Z:up
    World frame: X:left, Y:up, Z:forward
    """
    # Get current device rotation
    device_rotation = R.from_quat(quaternion)
    
    # Define the transformation that aligns coordinate systems with CORRECTED rotation directions
    # Headphone (X:right, Y:forward, Z:up) → World (X:left, Y:up, Z:forward)
    align_transform = R.from_matrix([
        [-1, 0, 0],   # X: right → left (negate)
        [0, 0, 1],    # Z: up → up (map Z to Y)
        [0, 1, 0]     # Y: forward → forward (map Y to Z)
    ])
    
    # NOTE: Ensure we're correctly handling the rotation direction
    # Instead of align * device * align.inv(), use a different composition:
    world_rotation = align_transform * device_rotation * align_transform.inv()
    
    # Checking if we need to invert specific rotations
    euler = world_rotation.as_euler('xyz', degrees=True)
    euler[0] = -euler[0]
    euler[1] = -euler[1]

    # Create new rotation from adjusted euler angles
    world_rotation = R.from_euler('xyz', euler, degrees=True)
    world_quaternion = world_rotation.as_quat()
    
    # Transform acceleration consistently
    world_acceleration = np.array([
        -acceleration[0],  # X: right → left
        acceleration[2],   # Z: up → up
        acceleration[1]    # Y: forward → forward
    ])
    
    # Transform gyroscope to world frame if provided
    world_gyroscope = None
    if gyroscope is not None:
        # Gyroscope represents angular velocity, needs the same transformation as acceleration
        world_gyroscope = np.array([
            -gyroscope[0],  # X: right → left (negate)
            gyroscope[2],   # Z: up → up (map Z to Y)
            gyroscope[1]    # Y: forward → forward (map Y to Z)
        ])
    
    if gyroscope is not None:
        return world_quaternion, world_acceleration, world_gyroscope
    else:
        return world_quaternion, world_acceleration

def preprocess_rokid_data(quaternion, acceleration, gyroscope=None):
    """
    Preprocess Rokid glasses data to align DIRECTLY with global coordinate system.
    
    Rokid device frame:   X: Right, Y: Up, Z: Forward
    Global frame:         X: Left,  Y: Up, Z: Forward
    
    Args:
        quaternion: np.ndarray - Orientation quaternion [x, y, z, w]
        acceleration: np.ndarray - Acceleration [x, y, z]
        gyroscope: np.ndarray - Gyroscope [x, y, z] (optional)
        
    Returns:
        tuple - (aligned_quaternion, aligned_acceleration, [aligned_gyroscope])
    """
    # Direct transformation to global frame:
    # Simply flip X axis (right → left)
    
    # Create 180° rotation around Y axis
    transform = R.from_euler('y', 180, degrees=True)
    
    # Apply to quaternion
    device_rotation = R.from_quat(quaternion)
    aligned_rotation = transform * device_rotation
    aligned_quaternion = aligned_rotation.as_quat()
    
    # For acceleration: flip X
    aligned_acceleration = np.array([
        -acceleration[0],  # -X (right to left)
        acceleration[1],   # Y stays the same (up)
        acceleration[2]    # Z stays the same (forward)
    ])
    
    # For gyroscope: flip X
    aligned_gyroscope = None
    if gyroscope is not None:
        aligned_gyroscope = np.array([
            -gyroscope[0],  # -X (right to left)
            gyroscope[1],   # Y stays the same (up)
            gyroscope[2]    # Z stays the same (forward)
        ])
    
    if gyroscope is not None:
        return aligned_quaternion, aligned_acceleration, aligned_gyroscope
    else:
        return aligned_quaternion, aligned_acceleration

def apply_gravity_compensation(quaternion, acceleration, gravity_magnitude=9.81):
    """
    Remove gravity component from acceleration vector.
    
    Args:
        quaternion: np.ndarray - Orientation quaternion [x, y, z, w]
        acceleration: np.ndarray - Acceleration [x, y, z]
        gravity_magnitude: float - Magnitude of gravity
        
    Returns:
        np.ndarray - Linear acceleration with gravity removed
    """
    try:
        # Define gravity in world frame (pointing down)
        gravity_world = np.array([0, -gravity_magnitude, 0])
        
        # Convert to device frame using inverse of orientation
        rotation = R.from_quat(quaternion)
        gravity_device_frame = rotation.inv().apply(gravity_world)
        
        # Remove gravity from raw acceleration
        linear_acceleration = acceleration - gravity_device_frame
        
        return linear_acceleration
        
    except Exception as e:
        logger.error(f"Error in gravity compensation: {e}")
        return acceleration

# -------------------- Parsing Methods --------------------

def parse_phone_data(data_str):
    """
    Parse phone data (screen-based device).
    
    Args:
        data_str: str - Raw data string from phone
        
    Returns:
        tuple - (timestamp, device_quat, device_accel, gyro, aligned_data)
    """
    parts = data_str.split()
    if len(parts) < 11:
        raise ValueError(f"Incomplete phone data: expected at least 11 parts, got {len(parts)}")
        
    timestamp = float(parts[0])
    
    # User acceleration (m/s²)
    device_accel = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
    
    # Quaternion from iOS (x, y, z, w format)
    device_quat = np.array([float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])])
    
    # Euler angles if available
    euler = None
    if len(parts) >= 12:
        euler = np.array([float(parts[9]), float(parts[10]), float(parts[11])])
    
    # Check if there's gyroscope data (we expect it at positions 12, 13, 14)
    gyro = np.array([0.0, 0.0, 0.0])
    if len(parts) >= 15:
        try:
            gyro = np.array([float(parts[12]), float(parts[13]), float(parts[14])])
        except (ValueError, IndexError):
            # If there's an error parsing, use zeros
            pass
    
    # Pre-transform to world frame - now include gyroscope
    aligned_quat, aligned_accel, aligned_gyro = preprocess_phone_watch_data(device_quat, device_accel, gyro)
    aligned_data = (aligned_quat, aligned_accel, aligned_gyro)
    
    return timestamp, device_quat, device_accel, gyro, aligned_data

# Update parse_watch_data to use the new preprocessing
def parse_watch_data(data_str):
    """
    Parse Apple Watch data.
    
    Args:
        data_str: str - Raw data string from watch
        
    Returns:
        tuple - (timestamp, device_quat, device_accel, gyro, aligned_data)
    """
    parts = data_str.split()
    if len(parts) < 12:
        raise ValueError(f"Incomplete watch data: expected at least 12 parts, got {len(parts)}")
        
    timestamp = float(parts[0])
    device_timestamp = float(parts[1])
    
    # Parse device-frame data
    device_accel = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
    device_quat = np.array([float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])])
    gyro = np.array([float(parts[9]), float(parts[10]), float(parts[11])])
    
    # Pre-transform to world frame - now include gyroscope
    aligned_quat, aligned_accel, aligned_gyro = preprocess_phone_watch_data(device_quat, device_accel, gyro)
    aligned_data = (aligned_quat, aligned_accel, aligned_gyro)
    
    return timestamp, device_quat, device_accel, gyro, aligned_data

def parse_headphone_data(data_str):
    """
    Parse headphone data.
    
    Args:
        data_str: str - Raw data string from headphone
        
    Returns:
        tuple - (timestamp, device_quat, device_accel, gyro, aligned_data)
    """
    parts = data_str.split()
    if len(parts) < 9:
        raise ValueError(f"Incomplete headphone data: expected at least 9 parts, got {len(parts)}")
        
    timestamp = float(parts[0])
    
    # Get device-frame data
    device_accel = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
    device_quat = np.array([float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])])
    
    # Check if gyroscope data is available (we expect it at positions 9, 10, 11)
    gyro = np.array([0.0, 0.0, 0.0])
    if len(parts) >= 12:
        try:
            gyro = np.array([float(parts[9]), float(parts[10]), float(parts[11])])
        except (ValueError, IndexError):
            # If there's an error parsing, use zeros
            pass
    
    # Preprocess headphone data to DIRECTLY match global frame - now include gyroscope
    aligned_quat, aligned_accel, aligned_gyro = preprocess_headphone_data(device_quat, device_accel, gyro)
    
    aligned_data = (aligned_quat, aligned_accel, aligned_gyro)
    
    return timestamp, device_quat, device_accel, gyro, aligned_data


def parse_rokid_glasses_data(data_str):
    """
    Parse Rokid Glasses data.
    
    Args:
        data_str: str - Raw data string from Rokid glasses
        
    Returns:
        tuple - (timestamp, device_quat, device_accel, gyro, aligned_data)
    """
    parts = data_str.split()
    if len(parts) != 12:
        raise ValueError(f"Rokid Glasses data format error: expected 12 values, got {len(parts)}")
        
    # Parse Unity's format
    timestamp = float(parts[0])
    device_timestamp = float(parts[1])
    
    # Parse quaternion from Unity (x, y, z, w format)
    device_quat = np.array([float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])])
    
    # Parse sensor data
    device_accel = np.array([float(parts[6]), float(parts[7]), float(parts[8])])
    gyro = np.array([float(parts[9]), float(parts[10]), float(parts[11])])
    
    # Preprocess Rokid data to DIRECTLY match global frame - now include gyroscope
    aligned_quat, aligned_accel, aligned_gyro = preprocess_rokid_data(device_quat, device_accel, gyro)
    aligned_data = (aligned_quat, aligned_accel, aligned_gyro)
    
    return timestamp, device_quat, device_accel, gyro, aligned_data

def parse_ios_data(message):
    """
    Parse iOS device data message.
    
    Args:
        message: str - Raw message from iOS device
        
    Returns:
        tuple - (device_id, parsed_data)
            parsed_data: tuple - Parsed device data (format depends on device type)
    """
    if not ';' in message:
        raise ValueError("Invalid iOS data format: missing ';' separator")
        
    device_prefix, data_part = message.split(';', 1)
    
    if data_part.startswith('phone:'):
        return 'phone', parse_phone_data(data_part[6:])
    elif data_part.startswith('headphone:'):
        return 'headphone', parse_headphone_data(data_part[10:])
    elif data_part.startswith('watch:'):
        return 'watch', parse_watch_data(data_part[6:])
    else:
        raise ValueError(f"Unknown iOS device type in message: {message}")

def calculate_euler_from_quaternion(quaternion):
    """
    Calculate Euler angles from quaternion.
    
    Args:
        quaternion: np.ndarray - Quaternion [x, y, z, w]
        
    Returns:
        np.ndarray - Euler angles [nod, turn, tilt] in degrees
    """
    try:
        rotation = R.from_quat(quaternion)
        euler_rad = rotation.as_euler('xyz', degrees=False)
        euler_deg = euler_rad * 180.0 / np.pi
        
        # Map to head movements: nod, turn, tilt
        nod = euler_deg[0]    # X rotation = NOD (up/down)
        turn = euler_deg[1]   # Y rotation = TURN (left/right)  
        tilt = euler_deg[2]   # Z rotation = TILT (left/right tilt)
        
        return np.array([nod, turn, tilt])
    except Exception as e:
        logger.warning(f"Error calculating Euler angles: {e}")
        return np.array([0, 0, 0])