"""
Input_Utils/sensor_calibrate.py - IMU calibration with improved T-pose alignment

This module provides calibration functionality for IMU devices with support
for reference device selection and global frame alignment.

The calibration system follows a two-step process:
1. Global Frame Alignment: Establish a consistent global coordinate system
2. T-Pose Alignment: Calculate transformations that map device orientations in T-pose to identity
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
    2. T-Pose Alignment: Calculate transformations that ensure device orientations in T-pose map to identity
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
        
        # DEBUG: Print smpl2imu in detail
        logger.info("DEBUG: smpl2imu matrix (reference → global):")
        for row in range(smpl2imu.shape[0]):
            row_str = " ".join([f"{val:.6f}" for val in smpl2imu[row]])
            logger.info(f"    {row_str}")
        
        # Store device orientations relative to reference
        device_orientations = {}
        
        # For each device, calculate the relative orientation
        for device_id, curr_quat in current_orientations.items():
            curr_rotation = R.from_quat(curr_quat)
            
            if device_id == reference_device:
                # For reference device, we use identity quaternion
                device_orientations[device_id] = np.array([0, 0, 0, 1])  # Identity quaternion
                logger.info(f"DEBUG: {device_id} (reference device) - identity relative transformation")
            else:
                # For other devices, calculate the relative rotation to the reference device
                relative_rotation = ref_rotation.inv() * curr_rotation
                device_orientations[device_id] = relative_rotation.as_quat()
                
                # Log the relative orientation
                rel_euler = relative_rotation.as_euler('xyz', degrees=True)
                logger.info(f"DEBUG: {device_id} relative to reference: "
                    f"X={rel_euler[0]:.1f}°, Y={rel_euler[1]:.1f}°, Z={rel_euler[2]:.1f}°")
        
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
            except Exception as e:
                logger.info(f"CALIBRATED: {device_id} quaternion: [{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]")
                logger.warning(f"Error calculating Euler angles: {e}")
        
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
        # For reference device, this should result in identity
        # For other devices, this applies their relative transformation
        transformed_rot = calib_rot.inv() * device_rot
        transformed_quat = transformed_rot.as_quat()
        
        # DEBUG: Log transformation details periodically
        if hasattr(self, '_frame_counter'):
            self._frame_counter = (self._frame_counter + 1) % 30
        else:
            self._frame_counter = 0
            
        if self._frame_counter == 0:
            try:
                # Log original and transformed orientations for debugging
                orig_euler = device_rot.as_euler('xyz', degrees=True)
                calib_euler = calib_rot.as_euler('xyz', degrees=True)
                transformed_euler = transformed_rot.as_euler('xyz', degrees=True)
                
                logger.debug(f"GLOBAL TRANSFORM: {device_id} original: "
                        f"X={orig_euler[0]:.1f}°, Y={orig_euler[1]:.1f}°, Z={orig_euler[2]:.1f}°")
                logger.debug(f"GLOBAL TRANSFORM: {device_id} calibration: "
                        f"X={calib_euler[0]:.1f}°, Y={calib_euler[1]:.1f}°, Z={calib_euler[2]:.1f}°")
                logger.debug(f"GLOBAL TRANSFORM: {device_id} transformed: "
                        f"X={transformed_euler[0]:.1f}°, Y={transformed_euler[1]:.1f}°, Z={transformed_euler[2]:.1f}°")
            except Exception as e:
                logger.debug(f"GLOBAL TRANSFORM: {device_id} logging error: {e}")
        
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
        Perform T-pose alignment for model inference with MobilePoseR-compatible approach.
        
        This is the second calibration step: T-Pose Alignment.
        Modified to exactly match MobilePoseR's device2bone calculation.
        
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
        
        # Get stored smpl2imu from global alignment
        smpl2imu = self.global_alignment.get('smpl2imu')
        if smpl2imu is None:
            logger.warning("No smpl2imu transformation found from global alignment. Using identity.")
            smpl2imu = torch.eye(3)
        else:
            logger.info("Using smpl2imu transformation from global alignment.")
            # Log the smpl2imu matrix for debugging
            logger.info("SMPL2IMU matrix:")
            for row in range(smpl2imu.shape[0]):
                row_str = " ".join([f"{val:.6f}" for val in smpl2imu[row]])
                logger.info(f"    {row_str}")
        
        # Log quaternions used for calibration with more detail
        for device_id, quat in device_orientations.items():
            try:
                if hasattr(quat, 'numpy'):
                    # Convert torch tensor to numpy if needed
                    quat_np = quat.numpy()
                else:
                    quat_np = quat
                    
                r = R.from_quat(quat_np)
                euler = r.as_euler('xyz', degrees=True)
                logger.info(f"T-POSE RAW: {device_id} orientation: "
                        f"Roll={euler[0]:.1f}°, Pitch={euler[1]:.1f}°, Yaw={euler[2]:.1f}°")
                logger.info(f"T-POSE RAW: {device_id} quaternion: "
                        f"[{quat_np[0]:.6f}, {quat_np[1]:.6f}, {quat_np[2]:.6f}, {quat_np[3]:.6f}]")
            except Exception as e:
                logger.warning(f"Error logging orientation for {device_id}: {e}")
        
        # Import the quaternion to rotation matrix function
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
        
        # Create device to bone transformations - optimized for batch processing
        device2bone = {}
        
        # Store original rotation matrices and global orientations for verification
        original_rotations = {}
        global_orientations = {}
        
        # DEBUG: Add detailed logging to understand each step
        logger.info("DEBUG: T-POSE TRANSFORMATION CALCULATION STEPS:")
        
        # Process each device
        for device_id, quat in device_orientations.items():
            # Convert to torch tensor if needed
            if not isinstance(quat, torch.Tensor):
                quat = torch.tensor(quat, dtype=torch.float32)
            
            # Convert quaternion to rotation matrix
            if len(quat.shape) == 1:
                # Single quaternion: [4] -> [3, 3]
                rot_matrix = quaternion_to_rotation_matrix(quat.unsqueeze(0)).squeeze(0)
            else:
                # Already batched
                rot_matrix = quaternion_to_rotation_matrix(quat)
            
            # Store original rotation for verification
            original_rotations[device_id] = rot_matrix.clone()
            
            # DEBUG: Log t-pose orientation matrix
            logger.info(f"DEBUG: {device_id} T-POSE orientation matrix:")
            for row in range(rot_matrix.shape[0]):
                row_str = " ".join([f"{val:.6f}" for val in rot_matrix[row]])
                logger.info(f"    {row_str}")
            
            # DEBUG: Log smpl2imu matrix before multiplication
            logger.info(f"DEBUG: smpl2imu matrix being applied:")
            for row in range(smpl2imu.shape[0]):
                row_str = " ".join([f"{val:.6f}" for val in smpl2imu[row]])
                logger.info(f"    {row_str}")
            
            # Calculate global t-pose orientation
            global_ori = smpl2imu.matmul(rot_matrix)
            
            # Store global orientation for verification
            global_orientations[device_id] = global_ori.clone()
            
            # DEBUG: Log global t-pose orientation
            logger.info(f"DEBUG: {device_id} GLOBAL T-POSE orientation (smpl2imu * rot_matrix):")
            for row in range(global_ori.shape[0]):
                row_str = " ".join([f"{val:.6f}" for val in global_ori[row]])
                logger.info(f"    {row_str}")
            
            # Calculate device2bone EXACTLY like MobilePoseR does
            device2bone[device_id] = global_ori.transpose(0, 1)
            
            # DEBUG: Log the device2bone matrix
            logger.info(f"DEBUG: {device_id} device2bone (calculated like MobilePoseR):")
            for row in range(device2bone[device_id].shape[0]):
                row_str = " ".join([f"{val:.6f}" for val in device2bone[device_id][row]])
                logger.info(f"    {row_str}")
            
            # Verify the result by checking if applying this transformation to the T-pose orientation yields identity
            verification_result = global_ori.matmul(device2bone[device_id])
            logger.info(f"DEBUG: {device_id} VERIFICATION (global_ori * device2bone):")
            for row in range(verification_result.shape[0]):
                row_str = " ".join([f"{val:.6f}" for val in verification_result[row]])
                logger.info(f"    {row_str}")
            
            # Calculate how close to identity the verification result is
            identity_error = torch.norm(verification_result - torch.eye(3))
            logger.info(f"DEBUG: {device_id} VERIFICATION error (deviation from identity): {identity_error:.10f}")
        
        # Process accelerations - Create acceleration offsets (including gravity)
        acc_offsets = {}
        
        if device_accelerations:
            # Use provided accelerations if available
            for device_id, acc in device_accelerations.items():
                if device_id in device_orientations:
                    # Convert to torch tensor if needed
                    if not isinstance(acc, torch.Tensor):
                        acc = torch.tensor(acc, dtype=torch.float32)
                    
                    # Apply transformation to global frame
                    acc_offsets[device_id] = smpl2imu.matmul(acc.unsqueeze(-1))
                    
                    # Log acceleration offsets
                    if hasattr(acc_offsets[device_id], 'squeeze'):
                        acc_offset_list = acc_offsets[device_id].squeeze().tolist()
                        logger.info(f"ACC_OFFSET: {device_id}: {acc_offset_list}")
        else:
            # Otherwise use default gravity values
            for device_id in device_orientations:
                # Default gravity in global frame (approximately 9.81 m/s² downward)
                gravity = torch.tensor([0.0, -9.81, 0.0], dtype=torch.float32).unsqueeze(-1)
                
                # Apply transformation to global frame
                acc_offsets[device_id] = smpl2imu.matmul(gravity)
                
                # Log default gravity offsets
                if hasattr(acc_offsets[device_id], 'squeeze'):
                    acc_offset_list = acc_offsets[device_id].squeeze().tolist()
                    logger.info(f"DEFAULT_GRAVITY: {device_id}: {acc_offset_list}")
        
        # Create gyroscope offsets
        gyro_offsets = {}
        
        if device_gyroscopes:
            # Use provided gyroscopes if available
            for device_id, gyro in device_gyroscopes.items():
                if device_id in device_orientations:
                    # Convert to torch tensor if needed
                    if not isinstance(gyro, torch.Tensor):
                        gyro = torch.tensor(gyro, dtype=torch.float32)
                    
                    # Apply transformation to global frame
                    gyro_offsets[device_id] = smpl2imu.matmul(gyro.unsqueeze(-1))
                    
                    # Log gyroscope offsets
                    if hasattr(gyro_offsets[device_id], 'squeeze'):
                        gyro_offset_list = gyro_offsets[device_id].squeeze().tolist()
                        logger.info(f"GYRO_OFFSET: {device_id}: {gyro_offset_list}")
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
            'gyro_offsets': gyro_offsets,
            'original_rotations': original_rotations,  # Store for verification
            'global_orientations': global_orientations  # Store for verification
        }
        
        # Re-verify after storing results
        logger.info("DEBUG: Re-verifying T-pose transformation with stored matrices:")
        for device_id in device_orientations.keys():
            if device_id not in device2bone or device_id not in global_orientations:
                continue
            
            # Get stored global orientation
            global_ori = global_orientations[device_id]
            
            # Verify with stored device2bone
            verification_result = global_ori.matmul(device2bone[device_id])
            logger.info(f"FINAL VERIFY: {device_id} final orientation (global_ori * device2bone):")
            for row in range(verification_result.shape[0]):
                row_str = " ".join([f"{val:.6f}" for val in verification_result[row]])
                logger.info(f"    {row_str}")
            
            # Calculate identity error
            identity_error = torch.norm(verification_result - torch.eye(3))
            logger.info(f"FINAL VERIFY: {device_id} identity error: {identity_error:.10f}")
        
        logger.info("T-pose alignment completed successfully")
        return smpl2imu, device2bone, acc_offsets, gyro_offsets
    

    def apply_tpose_transformation(self, device_id, quaternion, acceleration, gyroscope=None, smpl2imu=None, device2bone=None, acc_offsets=None, gyro_offsets=None):
        """
        Apply T-pose transformation to pre-processed data.
        
        This is typically used during inference with the model.
        
        Args:
            device_id: str - Device identifier
            quaternion: torch.Tensor - Device orientation quaternion
            acceleration: torch.Tensor - Device acceleration
            gyroscope: torch.Tensor - Device gyroscope (optional)
            smpl2imu: torch.Tensor - Optional override for smpl2imu
            device2bone: dict - Optional override for device2bone
            acc_offsets: dict - Optional override for acc_offsets
            gyro_offsets: dict - Optional override for gyro_offsets
            
        Returns:
            tuple - (transformed_orientation, transformed_acceleration, [transformed_gyroscope])
        """
        # Use provided values or fall back to stored values
        _smpl2imu = smpl2imu if smpl2imu is not None else self.global_alignment.get('smpl2imu')
        _device2bone = device2bone if device2bone is not None else self.tpose_alignment.get('device2bone', {}).get(device_id)
        _acc_offsets = acc_offsets if acc_offsets is not None else self.tpose_alignment.get('acc_offsets', {}).get(device_id)
        _gyro_offsets = gyro_offsets if gyro_offsets is not None else self.tpose_alignment.get('gyro_offsets', {}).get(device_id)
        
        if _smpl2imu is None or _device2bone is None or _acc_offsets is None:
            logger.warning(f"Missing T-pose calibration data for device {device_id}")
            if gyroscope is not None:
                return quaternion, acceleration, gyroscope
            else:
                return quaternion, acceleration
        
        try:
            # Convert quaternion to rotation matrix
            if isinstance(quaternion, np.ndarray):
                quaternion = torch.tensor(quaternion, dtype=torch.float32)
            
            # Log input data shape for debugging
            logger.debug(f"T-POSE TRANSFORM: {device_id} input quaternion shape: {quaternion.shape}")
            logger.debug(f"T-POSE TRANSFORM: {device_id} input acceleration shape: {acceleration.shape}")
            if gyroscope is not None:
                logger.debug(f"T-POSE TRANSFORM: {device_id} input gyroscope shape: {gyroscope.shape}")
            
            # Import MobilePoseR's quaternion_to_rotation_matrix if available
            try:
                from mobileposer.articulate.math import quaternion_to_rotation_matrix
                ori_matrix = quaternion_to_rotation_matrix(quaternion)
            except ImportError:
                # Fallback to using scipy
                if quaternion.dim() > 1:
                    # Handle batch dimension
                    r = R.from_quat(quaternion.cpu().numpy())
                    ori_matrix = torch.tensor(r.as_matrix(), dtype=torch.float32)
                else:
                    r = R.from_quat(quaternion.cpu().numpy())
                    ori_matrix = torch.tensor(r.as_matrix(), dtype=torch.float32)
            
            # DEBUG: Periodically log input and transformation details
            if hasattr(self, '_tpose_frame_counter'):
                self._tpose_frame_counter = (self._tpose_frame_counter + 1) % 30
            else:
                self._tpose_frame_counter = 0
                
            if self._tpose_frame_counter == 0:
                # Log input orientation
                try:
                    logger.debug(f"DEBUG: {device_id} RUNTIME orientation matrix:")
                    for row in range(ori_matrix.shape[0]):
                        row_str = " ".join([f"{val:.6f}" for val in ori_matrix[row]])
                        logger.debug(f"    {row_str}")
                    
                    # Log smpl2imu matrix
                    logger.debug(f"DEBUG: smpl2imu matrix:")
                    for row in range(_smpl2imu.shape[0]):
                        row_str = " ".join([f"{val:.6f}" for val in _smpl2imu[row]])
                        logger.debug(f"    {row_str}")
                    
                    # Log device2bone matrix
                    logger.debug(f"DEBUG: {device_id} device2bone matrix:")
                    for row in range(_device2bone.shape[0]):
                        row_str = " ".join([f"{val:.6f}" for val in _device2bone[row]])
                        logger.debug(f"    {row_str}")
                except Exception as e:
                    logger.debug(f"DEBUG logging error: {e}")
            
            # Log input orientation safely
            try:
                if hasattr(quaternion, 'detach'):
                    quat_np = quaternion.detach().cpu().numpy()
                elif hasattr(quaternion, 'numpy'):
                    quat_np = quaternion.numpy()
                else:
                    quat_np = quaternion
                
                # Only log for scalar values, not batches
                if isinstance(quat_np, np.ndarray) and quat_np.ndim == 1:
                    logger.debug(f"T-POSE TRANSFORM: {device_id} input quaternion: "
                                f"[{float(quat_np[0]):.6f}, {float(quat_np[1]):.6f}, "
                                f"{float(quat_np[2]):.6f}, {float(quat_np[3]):.6f}]")
            except Exception as e:
                logger.debug(f"T-POSE TRANSFORM: {device_id} input quaternion logging error: {e}")
            
            # Ensure acceleration has the right shape for transformation
            if acceleration.dim() == 1:
                # [3] -> [3, 1]
                accel_for_transform = acceleration.unsqueeze(-1)
            else:
                # Already has the right shape
                accel_for_transform = acceleration
            
            # Prepare gyroscope data if provided
            gyro_for_transform = None
            if gyroscope is not None:
                if gyroscope.dim() == 1:
                    # [3] -> [3, 1]
                    gyro_for_transform = gyroscope.unsqueeze(-1)
                else:
                    # Already has the right shape
                    gyro_for_transform = gyroscope
            
            # Apply transformations with new approach
            # For orientation: smpl2imu.matmul(ori).matmul(device2bone)
            global_ori = _smpl2imu.matmul(ori_matrix)
            transformed_ori = global_ori.matmul(_device2bone)
            
            # DEBUG: Calculate how close to identity the transformation is
            # when in T-pose, the result should be close to identity
            if self._tpose_frame_counter == 0:
                try:
                    # Calculate the difference from expected identity
                    t_pose_diff = global_ori.matmul(_device2bone)
                    
                    logger.debug(f"DEBUG: {device_id} T-POSE difference (global_ori * device2bone):")
                    for row in range(t_pose_diff.shape[0]):
                        row_str = " ".join([f"{val:.6f}" for val in t_pose_diff[row]])
                        logger.debug(f"    {row_str}")
                    
                    # Calculate how close to identity the difference is
                    identity_error = torch.norm(t_pose_diff - torch.eye(3))
                    logger.debug(f"DEBUG: {device_id} T-POSE error (deviation from identity): {identity_error:.10f}")
                except Exception as e:
                    logger.debug(f"DEBUG T-pose difference calculation error: {e}")
            
            # For acceleration: (smpl2imu.matmul(acc) - acc_offsets)
            transformed_acc = (_smpl2imu.matmul(accel_for_transform) - _acc_offsets)
            
            # For gyroscope: (smpl2imu.matmul(gyro) - gyro_offsets)
            transformed_gyro = None
            if gyro_for_transform is not None and _gyro_offsets is not None:
                transformed_gyro = (_smpl2imu.matmul(gyro_for_transform) - _gyro_offsets)
            
            # Ensure output has consistent dimensionality
            if acceleration.dim() == 1 and transformed_acc.dim() > 1:
                transformed_acc = transformed_acc.squeeze(-1)
            
            if gyroscope is not None and transformed_gyro is not None and gyroscope.dim() == 1 and transformed_gyro.dim() > 1:
                transformed_gyro = transformed_gyro.squeeze(-1)
            
            # Log orientation matrix shape for debugging
            logger.debug(f"T-POSE TRANSFORM: {device_id} transformed_ori shape: {transformed_ori.shape}")
            logger.debug(f"T-POSE TRANSFORM: {device_id} transformed_acc shape: {transformed_acc.shape}")
            if transformed_gyro is not None:
                logger.debug(f"T-POSE TRANSFORM: {device_id} transformed_gyro shape: {transformed_gyro.shape}")
            
            # Log final transformed orientation safely
            try:
                # Convert to rotation matrix to euler angles for logging if it's a single matrix
                if transformed_ori.dim() == 2 and transformed_ori.shape[0] == 3 and transformed_ori.shape[1] == 3:
                    from scipy.spatial.transform import Rotation as R
                    rot_matrix_np = transformed_ori.detach().cpu().numpy() if hasattr(transformed_ori, 'detach') else transformed_ori.cpu().numpy()
                    euler = R.from_matrix(rot_matrix_np).as_euler('xyz', degrees=True)
                    logger.debug(f"T-POSE TRANSFORM: {device_id} final orientation: "
                            f"Roll={float(euler[0]):.1f}°, Pitch={float(euler[1]):.1f}°, Yaw={float(euler[2]):.1f}°")
                else:
                    logger.debug(f"T-POSE TRANSFORM: {device_id} final orientation matrix shape: {transformed_ori.shape}")
            except Exception as e:
                logger.debug(f"T-POSE TRANSFORM: {device_id} final orientation logging error: {e}")
            
            # Return transformed data based on what was provided
            if gyroscope is not None and transformed_gyro is not None:
                return transformed_ori, transformed_acc, transformed_gyro
            else:
                return transformed_ori, transformed_acc
            
        except Exception as e:
            logger.warning(f"Error in apply_tpose_transformation: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            
            # Return original data if transformation fails
            if gyroscope is not None:
                return quaternion, acceleration, gyroscope
            else:
                return quaternion, acceleration
    
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