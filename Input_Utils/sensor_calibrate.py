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

class IMUCalibrator:
    """
    Handles calibration of IMU devices with selectable reference device.
    
    Provides two calibration stages:
    1. Global Frame Alignment: Establish reference device and global coordinate frame
    2. T-Pose Alignment: Capture device orientations during T-pose for bone alignment
    """
    
    def __init__(self):
        # Store device orientations relative to reference device
        self.device_orientations = {}
        # Currently selected reference device
        self.reference_device = None
        # Record which devices are calibrated
        self.calibrated_devices = set()
        # Global frame alignment results
        self.global_alignment = {
            'smpl2imu': None  # Transformation from reference device to global frame
        }
        # T-pose alignment results
        self.tpose_alignment = {
            'device2bone': {},  # Device to bone transformations
            'acc_offsets': {},  # Acceleration offsets
            'gyro_offsets': {}  # Gyroscope offsets
        }
    
    def set_device_orientation(self, device_id: str, orientation: np.ndarray):
        """Set device orientation relative to reference device"""
        self.device_orientations[device_id] = orientation.copy()
        self.calibrated_devices.add(device_id)
        
        logger.info(f"Set orientation for {device_id}")
        logger.info(f"   Orientation quat: [{orientation[0]:.3f}, {orientation[1]:.3f}, "
                   f"{orientation[2]:.3f}, {orientation[3]:.3f}]")
        
        # Calculate Euler angles for logging
        try:
            r = R.from_quat(orientation)
            euler = r.as_euler('xyz', degrees=True)
            logger.info(f"   Orientation euler: X={euler[0]:.1f}° Y={euler[1]:.1f}° Z={euler[2]:.1f}°")
        except Exception as e:
            logger.warning(f"Could not calculate Euler angles: {e}")
    
    #
    # Step 1: Global Frame Alignment
    #
    
    def perform_global_alignment(self, current_orientations, reference_device=None):
        """
        Perform global frame alignment using a reference device.
        
        This is the first calibration step that establishes the global reference frame.
        
        Args:
            current_orientations: dict - Current device orientations {device_id: quaternion}
            reference_device: str - Device to use as reference (default: auto-select)
            
        Returns:
            tuple - (reference_device, smpl2imu)
                reference_device: str - The device used as reference
                smpl2imu: torch.Tensor - Transformation matrix from reference to global frame
        """
        if not current_orientations:
            logger.warning("No device orientations provided for calibration")
            return None, None
        
        # Select reference device if not specified
        if reference_device is None or reference_device not in current_orientations:
            for device in ['phone', 'glasses', 'watch', 'headphone']:
                if device in current_orientations:
                    reference_device = device
                    break
            if reference_device is None:
                reference_device = next(iter(current_orientations))
        
        logger.info(f"Global alignment using {reference_device} as reference device")
        
        # Get reference quaternion
        ref_quat = current_orientations[reference_device]
        ref_rotation = R.from_quat(ref_quat)
        
        # Calculate smpl2imu transformation from reference orientation
        # This transforms from reference device frame to global frame
        smpl2imu = torch.from_numpy(ref_rotation.as_matrix()).float().t()
        
        # Log initial reference device orientation
        ref_euler_initial = ref_rotation.as_euler('xyz', degrees=True)
        logger.info(f"REFERENCE: {reference_device} orientation: "
                   f"X={ref_euler_initial[0]:.1f}°, Y={ref_euler_initial[1]:.1f}°, Z={ref_euler_initial[2]:.1f}°")
        logger.info(f"Global alignment transformation calculated")
        
        # Store device orientations relative to reference
        device_orientations = {}
        
        # For each device, calculate the relative orientation
        for device_id, curr_quat in current_orientations.items():
            curr_rotation = R.from_quat(curr_quat)
            
            if device_id == reference_device:
                # For reference device, we use its current orientation as the reference
                device_orientations[device_id] = ref_quat
            else:
                # For other devices, calculate the relative rotation to the reference device
                relative_rotation = ref_rotation.inv() * curr_rotation
                device_orientations[device_id] = relative_rotation.as_quat()
        
        # Store calibration results
        self.device_orientations = device_orientations
        self.reference_device = reference_device
        self.calibrated_devices = set(device_orientations.keys())
        self.global_alignment['smpl2imu'] = smpl2imu
        
        # Log calibration results
        for device_id, quat in device_orientations.items():
            try:
                calib_euler = R.from_quat(quat).as_euler('xyz', degrees=True)
                logger.info(f"CALIBRATED: {device_id} orientation: "
                          f"X={calib_euler[0]:.1f}°, Y={calib_euler[1]:.1f}°, Z={calib_euler[2]:.1f}°")
            except:
                logger.info(f"CALIBRATED: {device_id} quaternion: [{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]")
        
        return reference_device, smpl2imu
    
    def apply_global_transformation(self, device_id, quaternion, acceleration=None, gyroscope=None):
        """
        Apply global frame transformation to device data.
        
        This uses the calibration from the global alignment step.
        
        Args:
            device_id: str - Device identifier
            quaternion: np.ndarray - Device orientation quaternion
            acceleration: np.ndarray - Device acceleration (optional)
            gyroscope: np.ndarray - Device gyroscope (optional)
            
        Returns:
            tuple - Transformed data based on provided inputs
        """
        if device_id not in self.device_orientations:
            # Not calibrated - return as is
            if acceleration is not None and gyroscope is not None:
                return quaternion, acceleration, gyroscope
            elif acceleration is not None:
                return quaternion, acceleration
            else:
                return quaternion
        
        # Get the device's calibration quaternion
        device_calib_quat = self.device_orientations.get(device_id, np.array([0, 0, 0, 1]))

        # Convert quaternions to rotation matrices
        device_rot = R.from_quat(quaternion)
        calib_rot = R.from_quat(device_calib_quat)
        
        # Apply calibration to orientation
        # This shows the device's movement relative to its calibration position
        transformed_rot = calib_rot.inv() * device_rot
        transformed_quat = transformed_rot.as_quat()
        
        # Transform acceleration if provided
        transformed_acc = None
        if acceleration is not None:
            # First rotate acceleration to device frame
            acc_device = device_rot.apply(acceleration)
            # Then apply calibration rotation
            transformed_acc = calib_rot.inv().apply(acc_device)
        
        # Transform gyroscope if provided
        transformed_gyro = None
        if gyroscope is not None:
            # Transform gyroscope data similar to acceleration
            gyro_device = device_rot.apply(gyroscope)
            transformed_gyro = calib_rot.inv().apply(gyro_device)
        
        # Return transformed data based on what was provided
        if acceleration is not None and gyroscope is not None:
            return transformed_quat, transformed_acc, transformed_gyro
        elif acceleration is not None:
            return transformed_quat, transformed_acc
        else:
            return transformed_quat
    
    #
    # Step 2: T-Pose Alignment
    #
    
    def perform_tpose_alignment(self, device_orientations, device_accelerations=None, device_gyroscopes=None):
        """
        Perform T-pose alignment for model inference.
        
        This is the second calibration step that captures device orientations during T-pose.
        
        Args:
            device_orientations: dict - Device orientations during T-pose {device_id: quaternion}
            device_accelerations: dict - Device accelerations during T-pose {device_id: acceleration}
            device_gyroscopes: dict - Device gyroscopes during T-pose {device_id: gyroscope}
            
        Returns:
            tuple - (smpl2imu, device2bone, acc_offsets, gyro_offsets)
                smpl2imu: torch.Tensor - Transformation from SMPL to IMU frame
                device2bone: dict - Device to bone transformations {device_id: matrix}
                acc_offsets: dict - Acceleration offsets {device_id: offset}
                gyro_offsets: dict - Gyroscope offsets {device_id: offset}
        """
        if not device_orientations:
            logger.warning("Cannot perform T-pose alignment: No device orientations provided")
            return None, None, None, None
        
        # Get reference device (should be already set during global alignment)
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
        
        logger.info(f"T-pose alignment using {reference_device} as reference device")
        
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
                logger.info(f"T-POSE: {device_id} orientation: "
                        f"Quaternion=[{quat_np[0]:.3f}, {quat_np[1]:.3f}, {quat_np[2]:.3f}, {quat_np[3]:.3f}]")
                logger.info(f"T-POSE: {device_id} euler: "
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
        
        # Get stored smpl2imu from global alignment
        smpl2imu = self.global_alignment.get('smpl2imu')
        if smpl2imu is None:
            logger.warning("No smpl2imu transformation found from global alignment. Using identity.")
            smpl2imu = torch.eye(3)
        else:
            logger.info("Using smpl2imu transformation from global alignment.")
        
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
        
        # Store T-pose alignment results
        self.tpose_alignment = {
            'device2bone': device2bone,
            'acc_offsets': acc_offsets,
            'gyro_offsets': gyro_offsets
        }
        
        logger.info("T-pose alignment completed successfully")
        return smpl2imu, device2bone, acc_offsets, gyro_offsets
    
    def apply_tpose_transformation(self, device_id, quaternion, acceleration, smpl2imu=None, device2bone=None, acc_offsets=None):
        """
        Apply T-pose transformation to pre-processed data.
        
        This is typically used during inference with the model.
        
        Args:
            device_id: str - Device identifier
            quaternion: torch.Tensor - Device orientation quaternion
            acceleration: torch.Tensor - Device acceleration
            smpl2imu: torch.Tensor - Optional override for smpl2imu
            device2bone: dict - Optional override for device2bone
            acc_offsets: dict - Optional override for acc_offsets
            
        Returns:
            tuple - (transformed_orientation, transformed_acceleration)
        """
        # Use provided values or fall back to stored values
        _smpl2imu = smpl2imu if smpl2imu is not None else self.global_alignment.get('smpl2imu')
        _device2bone = device2bone if device2bone is not None else self.tpose_alignment.get('device2bone', {}).get(device_id)
        _acc_offsets = acc_offsets if acc_offsets is not None else self.tpose_alignment.get('acc_offsets', {}).get(device_id)
        
        if _smpl2imu is None or _device2bone is None or _acc_offsets is None:
            logger.warning(f"Missing T-pose calibration data for device {device_id}")
            return quaternion, acceleration
        
        # Convert quaternion to rotation matrix
        if isinstance(quaternion, np.ndarray):
            quaternion = torch.from_numpy(quaternion).float()
        
        from mobileposer.articulate.math import quaternion_to_rotation_matrix
        ori_matrix = quaternion_to_rotation_matrix(quaternion)
        
        # Apply transformations
        transformed_ori = _smpl2imu.matmul(ori_matrix).matmul(_device2bone)
        transformed_acc = (_smpl2imu.matmul(acceleration.unsqueeze(-1)) - _acc_offsets).squeeze(-1)
        
        return transformed_ori, transformed_acc
    
    #
    # Utility Methods
    #
    
    def is_calibrated(self, device_id):
        """Check if a device is calibrated"""
        return device_id in self.calibrated_devices
    
    def get_reference_device(self):
        """Get the current reference device"""
        return self.reference_device
    
    def get_device_orientations(self):
        """Get all device orientations"""
        return self.device_orientations.copy()
    
    def get_global_alignment(self):
        """Get the global alignment transformation"""
        return self.global_alignment.get('smpl2imu')
    
    def get_tpose_alignment(self):
        """Get the T-pose alignment data"""
        return (
            self.global_alignment.get('smpl2imu'),
            self.tpose_alignment.get('device2bone', {}),
            self.tpose_alignment.get('acc_offsets', {}),
            self.tpose_alignment.get('gyro_offsets', {})
        )
    
    def is_fully_calibrated(self):
        """Check if both calibration steps have been completed"""
        return (
            self.global_alignment.get('smpl2imu') is not None and
            bool(self.tpose_alignment.get('device2bone', {})) and
            bool(self.tpose_alignment.get('acc_offsets', {}))
        )