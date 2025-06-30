"""
Input_Utils/sensor_calibrate.py - IMU calibration and reference device handling

This module provides calibration functionality for IMU devices with support
for reference device selection and global frame alignment.
"""

import numpy as np
import logging
from scipy.spatial.transform import Rotation as R
import torch

logger = logging.getLogger(__name__)

def apply_calibration_transform(ori, acc, calibration_quats, device_id, reference_device=None, gyro=None):
    """
    Apply calibration transformation to pre-transformed sensor data.
    
    This function assumes the data is already pre-transformed to global frame.
    
    Args:
        ori: np.ndarray - Device orientation quaternion [x, y, z, w] in global frame
        acc: np.ndarray - Device acceleration [x, y, z] in global frame
        calibration_quats: dict - Calibration quaternions by device_id
        device_id: str - Device identifier
        reference_device: str - Current reference device (optional)
        gyro: np.ndarray - Device gyroscope [x, y, z] in global frame (optional)
        
    Returns:
        tuple of (transformed_orientation, transformed_acceleration, [transformed_gyroscope])
    """
    # Get the calibration quaternion for this device
    device_calib_quat = calibration_quats.get(device_id, np.array([0, 0, 0, 1]))

    # Convert quaternions to rotation matrices
    device_rot = R.from_quat(ori)
    calib_rot = R.from_quat(device_calib_quat)
    
    # Apply calibration to orientation - simply the inverse of calibration
    # This shows the device's movement relative to its calibration position
    # Note: For quaternions, .inv() is the conjugate/inverse
    transformed_rot = calib_rot.inv() * device_rot
    transformed_quat = transformed_rot.as_quat()
    
    # Apply same transformation to acceleration
    # First rotate acceleration to device frame
    acc_device = device_rot.apply(acc)
    # Then apply calibration rotation
    transformed_acc = calib_rot.inv().apply(acc_device)
    
    # Apply same transformation to gyroscope if provided
    transformed_gyro = None
    if gyro is not None:
        # Transform gyroscope data similar to acceleration
        gyro_device = device_rot.apply(gyro)
        transformed_gyro = calib_rot.inv().apply(gyro_device)
    
    if gyro is not None:
        return transformed_quat, transformed_acc, transformed_gyro
    else:
        return transformed_quat, transformed_acc


def align_global_identity(current_orientations, reference_device=None):
    """
    Align devices to global frame using a reference device.
    
    This function handles pre-transformed quaternions that are already aligned
    with the global coordinate system.
    
    Args:
        current_orientations: dict - Current device orientations {device_id: quaternion}
        reference_device: str - Device to use as reference (default: auto-select)
        
    Returns:
        tuple - (calibration_quats, reference_device)
            calibration_quats: dict - Updated calibration quaternions
            reference_device: str - The device used as reference
    """
    if not current_orientations:
        return {}, None
    
    # Select reference device if not specified
    if reference_device is None or reference_device not in current_orientations:
        for device in ['phone', 'glasses', 'watch', 'headphone']:
            if device in current_orientations:
                reference_device = device
                break
        if reference_device is None:
            reference_device = next(iter(current_orientations))
    
    logger.info(f"Calibrating using {reference_device} as reference device")
    
    # Get reference quaternion
    ref_quat = current_orientations[reference_device]
    ref_rotation = R.from_quat(ref_quat)
    
    # Log initial reference device orientation
    ref_euler_initial = ref_rotation.as_euler('xyz', degrees=True)
    logger.info(f"INITIAL: Reference device ({reference_device}) orientation: "
               f"X={ref_euler_initial[0]:.1f}°, Y={ref_euler_initial[1]:.1f}°, Z={ref_euler_initial[2]:.1f}°")
    
    # Store calibration quaternions
    calibration_quats = {}
    
    # For each device, calculate the calibration quaternion
    for device_id, curr_quat in current_orientations.items():
        curr_rotation = R.from_quat(curr_quat)
        
        if device_id == reference_device:
            # For reference device, we use its current orientation as the reference
            # No additional transformation needed since we're already in global frame
            calibration_quats[device_id] = ref_quat
        else:
            # For other devices, calculate the relative rotation to the reference device
            # This is the inverse of the reference rotation composed with the current device rotation
            relative_rotation = ref_rotation.inv() * curr_rotation
            calibration_quats[device_id] = relative_rotation.as_quat()
    
    # Log final calibration values
    for device_id, quat in calibration_quats.items():
        try:
            calib_euler = R.from_quat(quat).as_euler('xyz', degrees=True)
            logger.info(f"CALIBRATION: {device_id} orientation: "
                      f"X={calib_euler[0]:.1f}°, Y={calib_euler[1]:.1f}°, Z={calib_euler[2]:.1f}°")
        except:
            logger.info(f"CALIBRATION: {device_id} quaternion: [{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]")
    
    return calibration_quats, reference_device


class IMUCalibrator:
    """
    Handles calibration of IMU devices with selectable reference device.
    
    Provides two calibration stages:
    1. Initial frame alignment (reference selection and global frame alignment)
    2. T-pose calibration for MobilePoseR model inference
    """
    
    def __init__(self):
        # Store reference quaternions for each device (the "zero" position)
        self.reference_quaternions = {}
        # Currently selected reference device
        self.reference_device = None
        # Record which devices are calibrated
        self.calibrated_devices = set()
        # T-pose calibration results
        self.t_pose_calibration = {
            'smpl2imu': None,
            'device2bone': {},
            'acc_offsets': {},
            'gyro_offsets': {}  # Added gyroscope offsets
        }
    
    def set_reference_orientation(self, device_id: str, current_quaternion: np.ndarray):
        """Set current orientation as the reference/zero position for a single device"""
        self.reference_quaternions[device_id] = current_quaternion.copy()
        self.calibrated_devices.add(device_id)
        
        logger.info(f"Set reference orientation for {device_id}")
        logger.info(f"   Reference quat: [{current_quaternion[0]:.3f}, {current_quaternion[1]:.3f}, "
                   f"{current_quaternion[2]:.3f}, {current_quaternion[3]:.3f}]")
        
        # Calculate Euler angles for logging
        try:
            r = R.from_quat(current_quaternion)
            euler = r.as_euler('xyz', degrees=True)
            logger.info(f"   Reference euler: X={euler[0]:.1f}° Y={euler[1]:.1f}° Z={euler[2]:.1f}°")
        except Exception as e:
            logger.warning(f"Could not calculate Euler angles: {e}")
    
    def calibrate_all_devices(self, current_orientations, reference_device=None):
        """
        Calibrate all devices using a reference device.
        
        Args:
            current_orientations: dict - Current orientations {device_id: quaternion}
            reference_device: str - Device to use as reference (default: auto-select)
        
        Returns:
            str - The device used as reference
        """
        if not current_orientations:
            logger.warning("No device orientations provided for calibration")
            return None
        
        # Apply global identity alignment calibration
        new_calibration, ref_device = align_global_identity(
            current_orientations, 
            reference_device
        )
        
        # Update reference quaternions
        self.reference_quaternions = new_calibration
        self.reference_device = ref_device
        
        # Update calibrated devices
        self.calibrated_devices = set(new_calibration.keys())
        
        # Log success
        calibrated_devices = list(self.calibrated_devices)
        if calibrated_devices:
            logger.info(f"Calibrated devices: {', '.join(calibrated_devices)}")
            logger.info(f"Reference device: {self.reference_device}")
        else:
            logger.warning("No active devices to calibrate")
        
        return self.reference_device
    
    def apply_calibration(self, device_id, quaternion, acceleration=None, gyroscope=None):
        """
        Apply calibration to device data.
        
        Args:
            device_id: str - Device identifier
            quaternion: np.ndarray - Device orientation quaternion
            acceleration: np.ndarray - Device acceleration (optional)
            gyroscope: np.ndarray - Device gyroscope (optional)
            
        Returns:
            tuple - Calibrated data based on provided inputs
        """
        if device_id not in self.reference_quaternions:
            # Not calibrated - return as is
            if acceleration is not None and gyroscope is not None:
                return quaternion, acceleration, gyroscope
            elif acceleration is not None:
                return quaternion, acceleration
            else:
                return quaternion
        
        # Device is calibrated - apply transformation
        if acceleration is not None and gyroscope is not None:
            return apply_calibration_transform(
                quaternion, 
                acceleration, 
                self.reference_quaternions, 
                device_id,
                reference_device=self.reference_device,
                gyro=gyroscope
            )
        elif acceleration is not None:
            return apply_calibration_transform(
                quaternion, 
                acceleration, 
                self.reference_quaternions, 
                device_id,
                reference_device=self.reference_device
            )
        else:
            # Just transform quaternion
            device_rot = R.from_quat(quaternion)
            calib_rot = R.from_quat(self.reference_quaternions[device_id])
            transformed_rot = calib_rot.inv() * device_rot
            return transformed_rot.as_quat()
    
    def is_calibrated(self, device_id):
        """Check if a device is calibrated"""
        return device_id in self.calibrated_devices
    
    def get_reference_device(self):
        """Get the current reference device"""
        return self.reference_device
    
    def get_calibration_quaternions(self):
        """Get all calibration quaternions"""
        return self.reference_quaternions.copy()
        
    def calibrate_t_pose(self, device_orientations, device_accelerations=None, device_gyroscopes=None):
        """
        Perform T-pose calibration for MobilePoseR model inference
        
        This is the second calibration step after initial global frame calibration.
        
        Args:
            device_orientations: dict - Current device orientations {device_id: quaternion}
            device_accelerations: dict - Current device accelerations {device_id: acceleration}
            device_gyroscopes: dict - Current device gyroscopes {device_id: gyroscope}
            
        Returns:
            tuple - (smpl2imu, device2bone, acc_offsets, gyro_offsets)
                smpl2imu: torch.Tensor - Transformation from SMPL to IMU frame
                device2bone: dict - Device to bone transformations {device_id: matrix}
                acc_offsets: dict - Acceleration offsets {device_id: offset}
                gyro_offsets: dict - Gyroscope offsets {device_id: offset}
        """
        if not device_orientations:
            logger.warning("Cannot perform T-pose calibration: No device orientations provided")
            return None, None, None, None
        
        # Get reference device (should be already set during first calibration)
        reference_device = self.reference_device
        if not reference_device or reference_device not in device_orientations:
            # Try to find a suitable reference
            for device in ['phone', 'watch', 'glasses', 'headphone']:
                if device in device_orientations:
                    reference_device = device
                    break
            
            # If still no reference, use the first device
            if not reference_device:
                reference_device = next(iter(device_orientations))
        
        logger.info(f"T-pose calibration using {reference_device} as reference device")
        
        # Log quaternions used for calibration
        for device_id, quat in device_orientations.items():
            try:
                if hasattr(quat, 'numpy'):
                    # Convert torch tensor to numpy if needed
                    quat_np = quat.numpy()
                else:
                    quat_np = quat
                    
                r = R.from_quat(quat_np)
                euler = r.as_euler('xyz', degrees=True)
                logger.info(f"T-POSE ORIENTATION: {device_id}: "
                        f"Quaternion=[{quat_np[0]:.3f}, {quat_np[1]:.3f}, {quat_np[2]:.3f}, {quat_np[3]:.3f}]")
                logger.info(f"T-POSE EULER: {device_id}: "
                        f"X={euler[0]:.1f}°, Y={euler[1]:.1f}°, Z={euler[2]:.1f}°")
            except Exception as e:
                logger.warning(f"Error logging orientation for {device_id}: {e}")
        
        # Import the quaternion to rotation matrix function
        # Using a local import to avoid circular imports
        try:
            from mobileposer.articulate.math import quaternion_to_rotation_matrix
        except ImportError:
            # Fallback implementation if MobilePoseR is not available
            def quaternion_to_rotation_matrix(q):
                """Convert quaternion to rotation matrix"""
                if isinstance(q, torch.Tensor):
                    # Handle batch of quaternions
                    if q.dim() > 1:
                        rotmats = []
                        for quat in q:
                            r = R.from_quat(quat.cpu().numpy())
                            rotmats.append(torch.tensor(r.as_matrix(), dtype=q.dtype, device=q.device))
                        return torch.stack(rotmats)
                    else:
                        # Single quaternion
                        r = R.from_quat(q.cpu().numpy())
                        return torch.tensor(r.as_matrix(), dtype=q.dtype, device=q.device)
                else:
                    # Numpy input
                    r = R.from_quat(q)
                    return torch.tensor(r.as_matrix())
        
        # Get reference quaternion
        ref_quat = device_orientations[reference_device]
        
        # Convert to torch tensor if needed
        if not isinstance(ref_quat, torch.Tensor):
            ref_quat = torch.tensor(ref_quat, dtype=torch.float32)
        
        # Create SMPL to IMU transformation
        # This is an identity matrix because we're already in global frame
        # from the first calibration step
        smpl2imu = torch.eye(3)
        
        # Create device to bone transformations
        device2bone = {}
        for device_id, quat in device_orientations.items():
            # Convert to torch tensor if needed
            if not isinstance(quat, torch.Tensor):
                quat = torch.tensor(quat, dtype=torch.float32)
            
            # Need to unsqueeze for quaternion_to_rotation_matrix if it expects a batch
            if len(quat.shape) == 1:
                quat_unsqueezed = quat.unsqueeze(0)
                rot_matrix = quaternion_to_rotation_matrix(quat_unsqueezed).squeeze(0)
            else:
                rot_matrix = quaternion_to_rotation_matrix(quat)
            
            # Create transformation to align with T-pose bones
            device2bone[device_id] = smpl2imu.matmul(rot_matrix).transpose(0, 1).matmul(torch.eye(3))
        
        # Create acceleration offsets (including gravity)
        acc_offsets = {}
        
        if device_accelerations:
            # Use provided accelerations if available
            for device_id, acc in device_accelerations.items():
                if device_id in device_orientations:
                    # Convert to torch tensor if needed
                    if not isinstance(acc, torch.Tensor):
                        acc = torch.tensor(acc, dtype=torch.float32)
                    
                    # Reshape and apply transformation
                    acc_offsets[device_id] = smpl2imu.matmul(acc.unsqueeze(-1))
        else:
            # Otherwise use default gravity values
            for device_id in device_orientations:
                # Default gravity in global frame (approximately 9.81 m/s² downward)
                gravity = torch.tensor([0.0, -9.81, 0.0], dtype=torch.float32).unsqueeze(-1)
                
                # Apply transformation to global frame
                acc_offsets[device_id] = smpl2imu.matmul(gravity)
        
        # Create gyroscope offsets
        gyro_offsets = {}
        
        if device_gyroscopes:
            # Use provided gyroscopes if available
            for device_id, gyro in device_gyroscopes.items():
                if device_id in device_orientations:
                    # Convert to torch tensor if needed
                    if not isinstance(gyro, torch.Tensor):
                        gyro = torch.tensor(gyro, dtype=torch.float32)
                    
                    # Reshape and apply transformation
                    gyro_offsets[device_id] = smpl2imu.matmul(gyro.unsqueeze(-1))
        else:
            # Otherwise use zeros for gyroscope offsets
            for device_id in device_orientations:
                # Default gyroscope in global frame (no rotation)
                gyro_zero = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32).unsqueeze(-1)
                
                # Apply transformation to global frame
                gyro_offsets[device_id] = smpl2imu.matmul(gyro_zero)
        
        # Store T-pose calibration results
        self.t_pose_calibration = {
            'smpl2imu': smpl2imu,
            'device2bone': device2bone,
            'acc_offsets': acc_offsets,
            'gyro_offsets': gyro_offsets
        }
        
        logger.info("T-pose calibration completed successfully")
        return smpl2imu, device2bone, acc_offsets, gyro_offsets
    
    def get_t_pose_calibration(self):
        """Get the current T-pose calibration data"""
        return (
            self.t_pose_calibration['smpl2imu'],
            self.t_pose_calibration['device2bone'],
            self.t_pose_calibration['acc_offsets'],
            self.t_pose_calibration['gyro_offsets']
        )
    
    def has_t_pose_calibration(self):
        """Check if T-pose calibration has been performed"""
        return (
            self.t_pose_calibration['smpl2imu'] is not None and
            bool(self.t_pose_calibration['device2bone']) and
            bool(self.t_pose_calibration['acc_offsets']) and
            bool(self.t_pose_calibration['gyro_offsets'])
        )