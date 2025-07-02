#!/usr/bin/env python3
"""
IMU_receiver.py 

Key improvements:
- Enhanced modular design with separate calibration and parsing modules
- Improved coordinate transformations for all device types
- Support for reference device selection
- Added reset functionality for calibration
- Improved gyroscope data handling
- Dynamic waveform cleanup for disconnected devices
"""

import numpy as np
import time
import torch
from collections import deque
import logging
from scipy.spatial.transform import Rotation as R

# Import our visualization module
from UI.main_visualizer import IMUVisualizer

# Import our input utilities
from Input_Utils.socket_utils import SocketReceiver, ApiServer

# Import our sensor utilities
from Input_Utils.sensor_utils import (
    # Data types
    IMUData,
    
    # Transformation functions
    apply_gravity_compensation,
    calculate_euler_from_quaternion,
    
    # Parsing functions
    parse_ios_data,
    parse_rokid_glasses_data
)

# Import calibration module and functions
from Input_Utils.sensor_calibrate import (
    IMUCalibrator
)

logging.basicConfig(level=logging.INFO,
                              format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


class IMUReceiver:
    """Core IMU receiver with reference device selection before calibration"""
    
    # Class variable to store the instance for external access
    _instance = None
    
    def __init__(self, data_port=8001, api_port=9001):
        # Original initialization code...

        # Current device orientations and raw data
        self.current_orientations = {}
        self.raw_device_data = {}
        
        # Add storage for transformed data for efficient access
        self.transformed_data = {}
        
        # Public calibration flag - can be set by external applications
        self.calibration_requested = False
        self.reset_requested = False
        
        # Core components
        self.calibrator = IMUCalibrator()
        self.visualizer = IMUVisualizer()
        self.running = False
        
        # Initialize socket receivers
        self.socket_receiver = SocketReceiver(port=data_port)
        
        # Initialize API server with callbacks
        self.api_server = ApiServer(
            port=api_port,
            callbacks={
                'get_device_data': self._get_device_data,
                'get_active_devices': self._get_active_devices,
                'calibrate': self.request_calibration,
                'reset_calibration': self.request_reset,
                'select_reference_device': self._select_reference_device,
                'calibrate_t_pose': self.calibrate_t_pose,
                'get_t_pose_calibration': self.get_t_pose_calibration,
                'test_tpose_transformation': self.test_tpose_transformation  
            }
        )
        
        # Store instance for external access
        IMUReceiver._instance = self
        
        logger.info(f"Enhanced IMU Receiver initialized with reference device selection")
        logger.info(f"Data port: {data_port}, API port: {api_port}")
        logger.info("Headphones/AR Glasses pre-transformed to global frame for easier calibration")
        logger.info("Gyroscope data now properly transformed to global frame")
    
    @classmethod
    def get_instance(cls):
        """Get the current IMUReceiver instance for external access"""
        return cls._instance
    
    def request_calibration(self):
        """Request calibration (can be called internally or externally)"""
        self.calibration_requested = True
        # Use the selected reference device from visualizer
        logger.info(f"Calibration requested")
        return True
    
    def request_reset(self):
        """Request reset of calibration settings"""
        self.reset_requested = True
        logger.info(f"Reset calibration requested")
        return True
    
    def reset_calibration(self):
        """Reset all calibration settings"""
        # Reset calibrator
        self.calibrator = IMUCalibrator()
        
        # Reset visualizer's reference device
        self.visualizer.reset_calibration()
        
        logger.info("Calibration reset: all devices marked as uncalibrated, reference cleared")
        return True
    
    def _select_reference_device(self, device_id):
        """Select a device as reference (can be called via API)"""
        if device_id in self.current_orientations:
            # Log orientation at time of selection
            curr_quat = self.current_orientations[device_id]
            euler = calculate_euler_from_quaternion(curr_quat)
            logger.info(f"REFERENCE SELECTION: {device_id} orientation at selection: "
                    f"Roll={euler[0]:.1f}°, Pitch={euler[1]:.1f}°, Yaw={euler[2]:.1f}°")
            logger.info(f"REFERENCE SELECTION: {device_id} quaternion at selection: "
                    f"[{curr_quat[0]:.3f}, {curr_quat[1]:.3f}, {curr_quat[2]:.3f}, {curr_quat[3]:.3f}]")
            
            success = self.visualizer.select_reference_device(device_id)
            if success:
                logger.info(f"Selected {device_id} as reference device")
            return success
        return False
    
    def _get_active_devices(self):
        """Get list of active devices - used by API server"""
        active = []
        current_time = time.time()
        for device_id, device_data in self.visualizer.device_data.items():
            if current_time - device_data['last_update'] < 2.0:
                active.append(device_id)
        return active
    
    def _get_device_data(self, device_id):
        """Get data for a specific device - used by API server"""
        if device_id not in self.current_orientations:
            return None
            
        # Check if transformed data is available
        if device_id in self.transformed_data:
            transformed_data = self.transformed_data[device_id]
            current_time = time.time()
            
            # Check if data is fresh (less than 2 seconds old)
            if current_time - transformed_data['timestamp'] < 2.0:
                # Get device visualization data for frequency calculation
                device_data = self.visualizer.device_data.get(device_id)
                frequency = device_data.get('frequency', 0.0) if device_data else 0.0
                
                # Get transformed data
                quaternion = transformed_data['orientation']
                
                try:
                    rotation_matrix = R.from_quat(quaternion).as_matrix()
                except Exception as e:
                    logger.warning(f"Error converting quaternion to matrix: {e}")
                    rotation_matrix = np.eye(3)
                    
                latest_acceleration = transformed_data['acceleration']
                
                # Get gyroscope if available
                gyroscope = transformed_data.get('gyroscope')
                
                # Check calibration status based on global alignment
                global_alignment_complete = self.calibrator.global_alignment.get('smpl2imu') is not None
                device_calibrated = device_id in self.calibrator.calibrated_devices
                
                # Helper function to convert NumPy values to standard Python values
                def to_python_type(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
                        return float(obj)
                    elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
                        return int(obj)
                    elif isinstance(obj, list) or isinstance(obj, tuple):
                        return [to_python_type(item) for item in obj]
                    else:
                        return obj
                
                # Convert all values to standard Python types
                result = {
                    'timestamp': to_python_type(transformed_data['timestamp']),
                    'device_name': device_id,
                    'frequency': to_python_type(frequency),
                    'acceleration': to_python_type(latest_acceleration),
                    'rotation_matrix': to_python_type(rotation_matrix),
                    'quaternion': to_python_type(quaternion), 
                    'gyroscope': to_python_type(gyroscope),
                    'is_calibrated': bool(global_alignment_complete and device_calibrated),
                    'is_reference': bool(device_id == self.calibrator.reference_device),
                    'is_tpose_calibrated': bool(self.calibrator.is_fully_calibrated())
                }
                
                return result
        
        # Fallback to original implementation if transformed data is not available
        device_data = self.visualizer.device_data.get(device_id)
        if not device_data:
            return None
            
        current_time = time.time()
        if current_time - device_data['last_update'] > 2.0:
            return None
            
        # Get device data
        quaternion = device_data.get('quaternion', np.array([0, 0, 0, 1]))
        
        try:
            rotation_matrix = R.from_quat(quaternion).as_matrix()
        except:
            rotation_matrix = np.eye(3)
            
        acceleration_history = device_data.get('accel_history', [])
        if len(acceleration_history) == 0:
            return None
            
        latest_acceleration = list(acceleration_history[-1])
        
        # Get gyroscope if available
        gyroscope = None
        gyro_history = device_data.get('gyro_history', [])
        if len(gyro_history) > 0:
            latest_gyro = list(gyro_history[-1])
            if sum(abs(x) for x in latest_gyro) > 0.001:
                gyroscope = latest_gyro

        # Check calibration status based on global alignment
        global_alignment_complete = self.calibrator.global_alignment.get('smpl2imu') is not None
        device_calibrated = device_id in self.calibrator.calibrated_devices
        
        # Helper function to convert NumPy values to standard Python values
        def to_python_type(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, list) or isinstance(obj, tuple):
                return [to_python_type(item) for item in obj]
            else:
                return obj
        
        # Convert all values to standard Python types
        result = {
            'timestamp': to_python_type(device_data['last_update']),
            'device_name': device_id,
            'frequency': to_python_type(device_data.get('frequency', 0.0)),
            'acceleration': to_python_type(latest_acceleration),
            'rotation_matrix': to_python_type(rotation_matrix),
            'quaternion': to_python_type(quaternion), 
            'gyroscope': to_python_type(gyroscope),
            'is_calibrated': bool(global_alignment_complete and device_calibrated),
            'is_reference': bool(device_id == self.calibrator.reference_device),
            'is_tpose_calibrated': bool(self.calibrator.is_fully_calibrated())
        }
        
        return result
        
    def _parse_data(self, message, addr):
        """Parse incoming data from socket"""
        try:
            # Handle iOS system messages
            if isinstance(message, str):
                if message == "client_initialized":
                    logger.info(f"iOS device connected from {addr[0]}")
                    return
                elif message == "client_disconnected":
                    logger.info(f"iOS device disconnected")
                    return
            
            # Check if it's iOS format (contains ';' and device prefixes)
            if ';' in message and ('phone:' in message or 'headphone:' in message or 'watch:' in message):
                try:
                    # Use the parsing function from sensor_utils
                    device_id, parsed_data = parse_ios_data(message)
                    self._process_device_data(device_id, parsed_data, addr)
                except Exception as e:
                    logger.warning(f"iOS parse error: {e}")
            else:
                # Assume it's Rokid Glasses Unity format
                try:
                    # Use the parsing function from sensor_utils
                    parsed_data = parse_rokid_glasses_data(message)
                    device_id = self.socket_receiver.get_device_id_for_ip(addr[0], 'glasses')
                    self._process_device_data(device_id, parsed_data, addr)
                except Exception as e:
                    logger.warning(f"Rokid Glasses parse error: {e}")
                    
        except Exception as e:
            logger.warning(f"Parse error: {e}")
    
    def calibrate_all_devices(self):
        """
        Calibrate all active devices using the selected reference device.
        
        This is the first calibration step: Global Frame Alignment.
        """
        # Get the selected reference device
        reference_device = self.visualizer.get_selected_reference_device()
        
        # If no reference device is selected, don't proceed
        if not reference_device:
            logger.warning("No reference device selected for calibration")
            return None
        
        # Log raw orientation at time of calibration
        for device_id, quaternion in self.current_orientations.items():
            try:
                euler = calculate_euler_from_quaternion(quaternion)
                logger.info(f"PRE-CALIBRATION RAW: {device_id} orientation: "
                        f"Roll={euler[0]:.1f}°, Pitch={euler[1]:.1f}°, Yaw={euler[2]:.1f}°")
                logger.info(f"PRE-CALIBRATION RAW: {device_id} quaternion: "
                        f"[{quaternion[0]:.3f}, {quaternion[1]:.3f}, {quaternion[2]:.3f}, {quaternion[3]:.3f}]")
            except:
                pass
        
        # Apply global frame alignment using the calibrator
        ref_device, smpl2imu = self.calibrator.perform_global_alignment(
            self.current_orientations,
            reference_device
        )
        
        # Update reference device in visualizer
        self.visualizer.set_reference_device(ref_device)
        
        logger.info(f"Global frame alignment complete using {ref_device} as reference device")
        
        # Log calibrated device orientations
        for device_id, quaternion in self.current_orientations.items():
            try:
                # Log raw orientation after calibration
                euler = calculate_euler_from_quaternion(quaternion)
                logger.info(f"POST-CALIBRATION RAW: {device_id} orientation: "
                        f"Roll={euler[0]:.1f}°, Pitch={euler[1]:.1f}°, Yaw={euler[2]:.1f}°")
                
                # If device is calibrated, log its calibrated orientation
                if device_id in self.calibrator.device_orientations:
                    cal_quat = self.calibrator.device_orientations[device_id]
                    cal_euler = calculate_euler_from_quaternion(cal_quat)
                    logger.info(f"CALIBRATION QUATERNION: {device_id}: "
                            f"[{cal_quat[0]:.3f}, {cal_quat[1]:.3f}, {cal_quat[2]:.3f}, {cal_quat[3]:.3f}]")
                    logger.info(f"CALIBRATION EULER: {device_id}: "
                            f"Roll={cal_euler[0]:.1f}°, Pitch={cal_euler[1]:.1f}°, Yaw={cal_euler[2]:.1f}°")
                    
                    # Apply global transformation and log result
                    try:
                        transformed_quat = self.calibrator.apply_global_transformation(
                            device_id, 
                            quaternion
                        )
                            
                        display_euler = calculate_euler_from_quaternion(transformed_quat)
                        logger.info(f"TRANSFORMED ORIENTATION: {device_id}: "
                                f"Roll={display_euler[0]:.1f}°, Pitch={display_euler[1]:.1f}°, Yaw={display_euler[2]:.1f}°")
                    except Exception as e:
                        logger.warning(f"Could not calculate transformed orientation: {e}")
                        
            except Exception as e:
                logger.warning(f"Could not log orientation for {device_id}: {e}")
        
        return ref_device

    def calibrate_t_pose(self):
        """
        Perform T-pose calibration for model inference with improved transformation
        
        This is the second calibration step: T-Pose Alignment.
        
        Returns:
            Tuple of (smpl2imu, device2bone, acc_offsets, gyro_offsets) if successful
        """
        # Get the current reference device
        reference_device = self.calibrator.reference_device
        if not reference_device:
            logger.warning("Cannot perform T-pose calibration: No reference device selected")
            return None
        
        # Ensure all devices are already calibrated with the first step
        calibrated_devices = self.calibrator.calibrated_devices
        if len(calibrated_devices) == 0:
            logger.warning("Cannot perform T-pose calibration: No devices calibrated")
            return None
        
        # Collect mean orientations and accelerations for 3 seconds
        logger.info("T-pose calibration: Collecting data for 3 seconds...")
        
        # Create accumulators
        accumulated_quaternions = {}
        accumulated_accelerations = {}
        accumulated_gyroscopes = {}  # Add gyroscope accumulation
        sample_count = 0
        
        # Collection start time
        start_time = time.time()
        collection_time = 3.0  # 3 seconds
        
        # Collect samples
        while time.time() - start_time < collection_time:
            # Get current orientations
            for device_id in calibrated_devices:
                if device_id in self.current_orientations:
                    quat = self.current_orientations[device_id]
                    if device_id not in accumulated_quaternions:
                        accumulated_quaternions[device_id] = []
                    accumulated_quaternions[device_id].append(quat)
                    
                    # Get acceleration if available
                    if device_id in self.raw_device_data and 'acceleration' in self.raw_device_data[device_id]:
                        accel = self.raw_device_data[device_id]['acceleration']
                        if device_id not in accumulated_accelerations:
                            accumulated_accelerations[device_id] = []
                        accumulated_accelerations[device_id].append(accel)
                    
                    # Get gyroscope if available
                    if device_id in self.raw_device_data and 'gyroscope' in self.raw_device_data[device_id]:
                        gyro = self.raw_device_data[device_id]['gyroscope']
                        if device_id not in accumulated_gyroscopes:
                            accumulated_gyroscopes[device_id] = []
                        accumulated_gyroscopes[device_id].append(gyro)
                        
            sample_count += 1
            time.sleep(0.01)  # 100 Hz collection
        
        logger.info(f"T-pose calibration: Collected {sample_count} samples for {len(accumulated_quaternions)} devices")
        
        # Calculate means
        device_orientations = {}
        device_accelerations = {}
        device_gyroscopes = {}
        
        for device_id, quats in accumulated_quaternions.items():
            mean_quat = np.mean(np.array(quats), axis=0)
            # Normalize quaternion
            mean_quat = mean_quat / np.linalg.norm(mean_quat)
            device_orientations[device_id] = mean_quat
            
            # Log the mean orientation for each device during T-pose
            try:
                euler = calculate_euler_from_quaternion(mean_quat)
                logger.info(f"T-POSE MEAN: {device_id} orientation: "
                        f"Roll={euler[0]:.1f}°, Pitch={euler[1]:.1f}°, Yaw={euler[2]:.1f}°")
                logger.info(f"T-POSE MEAN: {device_id} quaternion: "
                        f"[{mean_quat[0]:.6f}, {mean_quat[1]:.6f}, {mean_quat[2]:.6f}, {mean_quat[3]:.6f}]")
            except Exception as e:
                logger.warning(f"Error calculating Euler angles for {device_id}: {e}")
        
        for device_id, accels in accumulated_accelerations.items():
            device_accelerations[device_id] = np.mean(np.array(accels), axis=0)
            logger.info(f"T-POSE MEAN: {device_id} acceleration: {device_accelerations[device_id].tolist()}")
        
        for device_id, gyros in accumulated_gyroscopes.items():
            device_gyroscopes[device_id] = np.mean(np.array(gyros), axis=0)
            logger.info(f"T-POSE MEAN: {device_id} gyroscope: {device_gyroscopes[device_id].tolist()}")
        
        # Get the smpl2imu matrix for verification before calibration
        smpl2imu = self.calibrator.global_alignment.get('smpl2imu')
        if smpl2imu is not None:
            logger.info("DEBUG: Global frame transformation matrix (smpl2imu) BEFORE T-pose:")
            for row in range(smpl2imu.shape[0]):
                row_str = " ".join([f"{val:.6f}" for val in smpl2imu[row]])
                logger.info(f"    {row_str}")
        
        # Convert quaternions to rotation matrices for verification
        logger.info("DEBUG: T-pose device orientations in global frame BEFORE calibration:")
        for device_id, quat in device_orientations.items():
            rot_matrix = R.from_quat(quat).as_matrix()
            
            # Convert to torch tensor for compatibility with smpl2imu
            rot_tensor = torch.tensor(rot_matrix, dtype=torch.float32)
            
            # Calculate global orientation if smpl2imu is available
            if smpl2imu is not None:
                global_ori = smpl2imu.matmul(rot_tensor)
                logger.info(f"DEBUG: {device_id} global orientation in T-pose:")
                for row in range(global_ori.shape[0]):
                    row_str = " ".join([f"{val:.6f}" for val in global_ori[row]])
                    logger.info(f"    {row_str}")
                
                # Calculate what device2bone should be (inverse of global orientation)
                try:
                    expected_device2bone = torch.inverse(global_ori)
                    logger.info(f"DEBUG: {device_id} expected device2bone (inverse of global_ori):")
                    for row in range(expected_device2bone.shape[0]):
                        row_str = " ".join([f"{val:.6f}" for val in expected_device2bone[row]])
                        logger.info(f"    {row_str}")
                    
                    # Verify the result (should be identity)
                    verification = global_ori.matmul(expected_device2bone)
                    logger.info(f"DEBUG: {device_id} verification (global_ori * device2bone):")
                    for row in range(verification.shape[0]):
                        row_str = " ".join([f"{val:.6f}" for val in verification[row]])
                        logger.info(f"    {row_str}")
                    
                    # Calculate identity error
                    identity_error = torch.norm(verification - torch.eye(3))
                    logger.info(f"DEBUG: {device_id} identity error: {identity_error:.10f}")
                except Exception as e:
                    logger.warning(f"Error calculating inverse: {e}")
        
        # Call calibrate_t_pose from the calibrator with the updated method
        result = self.calibrator.perform_tpose_alignment(device_orientations, device_accelerations, device_gyroscopes)
        
        # Log the calculated parameters
        if result:
            smpl2imu, device2bone, acc_offsets, gyro_offsets = result
            logger.info("T-pose calibration completed successfully with parameters:")
            
            # Log smpl2imu matrix after calibration
            logger.info("DEBUG: Global frame transformation matrix (smpl2imu) AFTER T-pose:")
            for row in range(smpl2imu.shape[0]):
                row_str = " ".join([f"{val:.6f}" for val in smpl2imu[row]])
                logger.info(f"    {row_str}")
            
            # Log device2bone matrices in more detail
            for device_id, matrix in device2bone.items():
                logger.info(f"TPOSE RESULT: {device_id} device2bone computed")
                # Log the matrix details for each device
                try:
                    logger.info(f"TPOSE RESULT: {device_id} device2bone matrix:")
                    for row in range(matrix.shape[0]):
                        row_values = [f"{val:.6f}" for val in matrix[row].tolist()]
                        logger.info(f"    {' '.join(row_values)}")
                except Exception as e:
                    logger.warning(f"Error formatting device2bone matrix for {device_id}: {e}")
            
            # Perform verification checks after calibration
            logger.info("DEBUG: Performing post-calibration verification checks:")
            for device_id, quat in device_orientations.items():
                if device_id not in device2bone:
                    continue
                
                # Create tensors from the SAME quaternions used for calibration
                rot_matrix = R.from_quat(quat).as_matrix()
                rot_tensor = torch.tensor(rot_matrix, dtype=torch.float32)
                
                # Calculate global orientation using the SAME smpl2imu from calibration
                global_ori = smpl2imu.matmul(rot_tensor)
                
                # Apply device2bone to get final orientation
                final_ori = global_ori.matmul(device2bone[device_id])
                
                # Log the final orientation
                logger.info(f"DEBUG: {device_id} final orientation (global_ori * device2bone):")
                for row in range(final_ori.shape[0]):
                    row_str = " ".join([f"{val:.6f}" for val in final_ori[row]])
                    logger.info(f"    {row_str}")
                
                # Calculate identity error
                identity_error = torch.norm(final_ori - torch.eye(3))
                logger.info(f"DEBUG: {device_id} final identity error: {identity_error:.10f}")
                
                # Check if the error is acceptable
                if identity_error < 1e-5:
                    logger.info(f"✅ {device_id} T-pose verification passed")
                else:
                    logger.warning(f"⚠️ {device_id} T-pose verification FAILED: error={identity_error:.10f}")
                    
                    # Detailed analysis of the error
                    eigen_values, eigen_vectors = torch.linalg.eig(final_ori)
                    logger.info(f"DEBUG: {device_id} eigenvalues of final orientation:")
                    eigen_values_real = torch.real(eigen_values)
                    eigen_values_imag = torch.imag(eigen_values)
                    for i in range(eigen_values.shape[0]):
                        logger.info(f"    λ{i+1} = {eigen_values_real[i]:.6f} + {eigen_values_imag[i]:.6f}j")
                    
                    # Euler angle decomposition
                    try:
                        euler = R.from_matrix(final_ori.detach().cpu().numpy()).as_euler('xyz', degrees=True)
                        logger.info(f"DEBUG: {device_id} euler angles of final orientation: "
                                f"Roll={float(euler[0]):.1f}°, Pitch={float(euler[1]):.1f}°, Yaw={float(euler[2]):.1f}°")
                    except Exception as e:
                        logger.warning(f"Error calculating Euler angles: {e}")
        
        # Clear transformed data to force recalculation with new calibration
        self.transformed_data = {}
        
        return result


    def test_tpose_transformation(self, device_id):
        """
        Test the T-pose transformation for a specific device to see if it works correctly.
        
        This function applies the transformation to the current device orientation and logs
        how far it is from identity when the device is in T-pose.
        
        Args:
            device_id: str - Device identifier to test
            
        Returns:
            float - Identity error (norm of difference from identity matrix)
        """
        if not self.calibrator.is_fully_calibrated():
            logger.warning("Cannot test T-pose transformation: System not fully calibrated")
            return None
        
        if device_id not in self.current_orientations:
            logger.warning(f"Cannot test T-pose transformation: Device {device_id} not found")
            return None
        
        # Get current device orientation
        current_quat = self.current_orientations[device_id]
        
        # Log current orientation
        euler = calculate_euler_from_quaternion(current_quat)
        logger.info(f"TEST T-POSE: {device_id} current orientation: "
                f"Roll={euler[0]:.1f}°, Pitch={euler[1]:.1f}°, Yaw={euler[2]:.1f}°")
        
        # Get calibration data
        smpl2imu = self.calibrator.global_alignment.get('smpl2imu')
        device2bone = self.calibrator.tpose_alignment.get('device2bone', {}).get(device_id)
        
        if smpl2imu is None or device2bone is None:
            logger.warning(f"Cannot test T-pose transformation: Missing calibration data for {device_id}")
            return None
        
        # Convert quaternion to rotation matrix
        rot_matrix = R.from_quat(current_quat).as_matrix()
        rot_tensor = torch.tensor(rot_matrix, dtype=torch.float32)
        
        # Calculate global orientation
        global_ori = smpl2imu.matmul(rot_tensor)
        
        # Apply device2bone to get final orientation
        final_ori = global_ori.matmul(device2bone)
        
        # Log the final orientation
        logger.info(f"TEST T-POSE: {device_id} final orientation (global_ori * device2bone):")
        for row in range(final_ori.shape[0]):
            row_str = " ".join([f"{val:.6f}" for val in final_ori[row]])
            logger.info(f"    {row_str}")
        
        # Calculate identity error
        identity_error = torch.norm(final_ori - torch.eye(3)).item()
        logger.info(f"TEST T-POSE: {device_id} identity error: {identity_error:.10f}")
        
        # Check if the error is acceptable
        if identity_error < 1e-1:
            logger.info(f"✅ {device_id} Current pose is close to T-pose (error={identity_error:.10f})")
        else:
            logger.info(f"ℹ️ {device_id} Current pose is not in T-pose (error={identity_error:.10f})")
        
        # Calculate what the current pose is relative to T-pose
        try:
            euler = R.from_matrix(final_ori.detach().cpu().numpy()).as_euler('xyz', degrees=True)
            logger.info(f"TEST T-POSE: {device_id} deviation from T-pose: "
                    f"Roll={float(euler[0]):.1f}°, Pitch={float(euler[1]):.1f}°, Yaw={float(euler[2]):.1f}°")
        except Exception as e:
            logger.warning(f"Error calculating Euler angles: {e}")

        return identity_error

    def _process_device_data(self, device_id, parsed_data, addr):
        """
        Process parsed device data with improved T-pose transformation.
        
        Args:
            device_id: str - Device identifier
            parsed_data: tuple - Parsed data from device
            addr: tuple - Socket address (host, port)
        """
        try:
            timestamp, device_quat, device_accel, gyro, aligned_data = parsed_data
            
            # Log initial connection and orientation
            if device_id not in self.current_orientations:
                logger.info(f"Initial connection from {device_id} at {addr[0]}")
                
                # Log initial orientation
                euler = calculate_euler_from_quaternion(device_quat)
                logger.info(f"CONNECTED: {device_id} initial orientation: "
                        f"Roll={euler[0]:.1f}°, Pitch={euler[1]:.1f}°, Yaw={euler[2]:.1f}°")
                logger.info(f"CONNECTED: {device_id} initial quaternion: "
                        f"[{device_quat[0]:.3f}, {device_quat[1]:.3f}, {device_quat[2]:.3f}, {device_quat[3]:.3f}]")
            
            # Store raw data for future reference
            self.raw_device_data[device_id] = {
                'quaternion': device_quat,
                'acceleration': device_accel,
                'gyroscope': gyro
            }
            
            # For all devices, use the pre-aligned data if available
            if aligned_data is not None:
                # Check if aligned_data contains gyroscope
                if len(aligned_data) == 3:
                    aligned_quat, aligned_accel, aligned_gyro = aligned_data
                else:
                    aligned_quat, aligned_accel = aligned_data
                    aligned_gyro = gyro  # Use original gyro if not transformed
                
                # Store original and aligned data
                self.raw_device_data[device_id].update({
                    'aligned_quaternion': aligned_quat,
                    'aligned_acceleration': aligned_accel,
                    'aligned_gyroscope': aligned_gyro
                })
                
                # Use aligned data for calibration and transformation
                quat_for_calibration = aligned_quat
                accel_for_processing = aligned_accel
                gyro_for_processing = aligned_gyro
                
                # Store the aligned quaternion for calibration
                self.current_orientations[device_id] = aligned_quat
            else:
                # Fallback to original data if no alignment available
                quat_for_calibration = device_quat
                accel_for_processing = device_accel
                gyro_for_processing = gyro
                
                # Store the original quaternion for calibration
                self.current_orientations[device_id] = device_quat
                
                # Log current raw orientation periodically (every 30 frames)
                if device_id in self.raw_device_data and 'frame_counter' in self.raw_device_data[device_id]:
                    frame_counter = self.raw_device_data[device_id]['frame_counter'] + 1
                    self.raw_device_data[device_id]['frame_counter'] = frame_counter
                    
                    if frame_counter % 30 == 0:
                        euler = calculate_euler_from_quaternion(device_quat)
                        logger.debug(f"RAW: {device_id} orientation: "
                                f"Roll={euler[0]:.1f}°, Pitch={euler[1]:.1f}°, Yaw={euler[2]:.1f}°")
                else:
                    # Initialize frame counter
                    self.raw_device_data[device_id]['frame_counter'] = 0
            
            # Apply calibration transformation if device is calibrated
            global_alignment_complete = self.calibrator.global_alignment.get('smpl2imu') is not None
            device_calibrated = device_id in self.calibrator.calibrated_devices

            if global_alignment_complete and device_calibrated:
                # Apply global transformation with gyroscope
                if gyro_for_processing is not None:
                    global_quat, global_acc, global_gyro = self.calibrator.apply_global_transformation(
                        device_id, 
                        quat_for_calibration, 
                        accel_for_processing,
                        gyroscope=gyro_for_processing
                    )
                else:
                    global_quat, global_acc = self.calibrator.apply_global_transformation(
                        device_id, 
                        quat_for_calibration, 
                        accel_for_processing
                    )
                    global_gyro = gyro_for_processing  # Use processed gyro if not returned
                
                # Log global transformation result
                euler = calculate_euler_from_quaternion(global_quat)
                logger.debug(f"GLOBAL TRANSFORM: {device_id} orientation: "
                            f"Roll={euler[0]:.1f}°, Pitch={euler[1]:.1f}°, Yaw={euler[2]:.1f}°")
                
                # If T-pose calibration is complete, apply that transformation too
                if self.calibrator.is_fully_calibrated():
                    try:
                        # Convert to tensors for T-pose transformation
                        global_quat_tensor = torch.tensor(global_quat, dtype=torch.float32)
                        global_acc_tensor = torch.tensor(global_acc, dtype=torch.float32)
                        
                        # Convert gyro to tensor if available
                        global_gyro_tensor = torch.tensor(global_gyro, dtype=torch.float32) if global_gyro is not None else None
                        
                        # Log pre-T-pose transformation
                        try:
                            euler = calculate_euler_from_quaternion(global_quat)
                            logger.debug(f"PRE-TPOSE: {device_id} orientation: "
                                        f"Roll={float(euler[0]):.1f}°, Pitch={float(euler[1]):.1f}°, Yaw={float(euler[2]):.1f}°")
                        except Exception as e:
                            logger.debug(f"PRE-TPOSE: {device_id} logging error: {e}")
                        
                        # Apply T-pose transformation with updated method
                        if global_gyro_tensor is not None:
                            transformed_ori, transformed_acc, transformed_gyro = self.calibrator.apply_tpose_transformation(
                                device_id, global_quat_tensor, global_acc_tensor, global_gyro_tensor
                            )
                        else:
                            transformed_ori, transformed_acc = self.calibrator.apply_tpose_transformation(
                                device_id, global_quat_tensor, global_acc_tensor
                            )
                            transformed_gyro = None
                        
                        # Convert rotation matrix back to quaternion for visualization
                        if isinstance(transformed_ori, torch.Tensor):
                            from scipy.spatial.transform import Rotation as R
                            try:
                                # Ensure we have a proper 3x3 matrix before conversion
                                if transformed_ori.ndim == 2 and transformed_ori.shape == (3, 3):
                                    # Single rotation matrix [3, 3]
                                    matrix_for_conversion = transformed_ori.cpu().numpy()
                                    r = R.from_matrix(matrix_for_conversion)
                                    global_quat = r.as_quat()
                                    
                                    # Log final orientation after T-pose transformation
                                    euler = r.as_euler('xyz', degrees=True)
                                    logger.debug(f"POST-TPOSE: {device_id} orientation: "
                                            f"Roll={float(euler[0]):.1f}°, Pitch={float(euler[1]):.1f}°, Yaw={float(euler[2]):.1f}°")
                                elif transformed_ori.ndim > 2:
                                    # Batched rotation matrices - take first one for display
                                    matrix_for_conversion = transformed_ori[0].cpu().numpy()
                                    r = R.from_matrix(matrix_for_conversion)
                                    global_quat = r.as_quat()
                                    
                                    # Log final orientation
                                    euler = r.as_euler('xyz', degrees=True)
                                    logger.debug(f"POST-TPOSE (from batch): {device_id} orientation: "
                                            f"Roll={float(euler[0]):.1f}°, Pitch={float(euler[1]):.1f}°, Yaw={float(euler[2]):.1f}°")
                                else:
                                    # Unexpected shape - log and use original quaternion
                                    logger.warning(f"Unexpected transformed_ori shape for {device_id}: {transformed_ori.shape}")
                                    # Keep original global_quat
                            except Exception as e:
                                logger.warning(f"POST-TPOSE: {device_id} matrix conversion error: {e}")
                                # Keep original global_quat
                        
                        # Convert transformed acceleration to numpy
                        if isinstance(transformed_acc, torch.Tensor):
                            # Handle different dimensions
                            if transformed_acc.ndim == 1:
                                # Single vector [3]
                                global_acc = transformed_acc.cpu().numpy()
                            elif transformed_acc.ndim == 2:
                                if transformed_acc.shape[1] == 1:
                                    # Column vector [3, 1]
                                    global_acc = transformed_acc.squeeze(-1).cpu().numpy()
                                else:
                                    # Unexpected shape - log and use first row
                                    logger.debug(f"Unexpected transformed_acc shape: {transformed_acc.shape}")
                                    global_acc = transformed_acc[0].cpu().numpy()
                            else:
                                # More dimensions - take first element
                                logger.debug(f"Complex transformed_acc shape: {transformed_acc.shape}")
                                global_acc = transformed_acc.view(-1, 3)[0].cpu().numpy()
                        
                        # Convert transformed gyroscope to numpy if available
                        if transformed_gyro is not None and isinstance(transformed_gyro, torch.Tensor):
                            # Handle different dimensions
                            if transformed_gyro.ndim == 1:
                                # Single vector [3]
                                global_gyro = transformed_gyro.cpu().numpy()
                            elif transformed_gyro.ndim == 2:
                                if transformed_gyro.shape[1] == 1:
                                    # Column vector [3, 1]
                                    global_gyro = transformed_gyro.squeeze(-1).cpu().numpy()
                                else:
                                    # Unexpected shape - log and use first row
                                    logger.debug(f"Unexpected transformed_gyro shape: {transformed_gyro.shape}")
                                    global_gyro = transformed_gyro[0].cpu().numpy()
                            else:
                                # More dimensions - take first element
                                logger.debug(f"Complex transformed_gyro shape: {transformed_gyro.shape}")
                                global_gyro = transformed_gyro.view(-1, 3)[0].cpu().numpy()
                        
                        # Store the transformed data for efficient access
                        if device_id not in self.transformed_data:
                            self.transformed_data[device_id] = {}
                        
                        # Update transformed data storage
                        self.transformed_data[device_id] = {
                            'orientation': global_quat,
                            'acceleration': global_acc,
                            'gyroscope': global_gyro if transformed_gyro is not None else None,
                            'timestamp': timestamp
                        }
                        
                    except Exception as e:
                        logger.warning(f"Error applying T-pose transformation: {e}")
                        import traceback
                        logger.debug(traceback.format_exc())
                        
                        # Store the global transformed data as fallback
                        if device_id not in self.transformed_data:
                            self.transformed_data[device_id] = {}
                        
                        self.transformed_data[device_id] = {
                            'orientation': global_quat,
                            'acceleration': global_acc,
                            'gyroscope': global_gyro,
                            'timestamp': timestamp
                        }
                else:
                    # Not fully calibrated - use global transformation results
                    # Store the global transformed data
                    if device_id not in self.transformed_data:
                        self.transformed_data[device_id] = {}
                    
                    self.transformed_data[device_id] = {
                        'orientation': global_quat,
                        'acceleration': global_acc,
                        'gyroscope': global_gyro,
                        'timestamp': timestamp
                    }
            else:
                # Not calibrated - use as is
                global_quat = quat_for_calibration
                global_acc = accel_for_processing
                global_gyro = gyro_for_processing
                
                # Store the raw data
                if device_id not in self.transformed_data:
                    self.transformed_data[device_id] = {}
                
                self.transformed_data[device_id] = {
                    'orientation': global_quat,
                    'acceleration': global_acc,
                    'gyroscope': global_gyro,
                    'timestamp': timestamp
                }
            
            # Apply gravity compensation if enabled for glasses
            if self.visualizer.get_gravity_enabled() and device_id == 'glasses':
                linear_accel = apply_gravity_compensation(global_quat, global_acc)
            else:
                # Use acceleration without gravity compensation
                linear_accel = global_acc

            if isinstance(global_quat, np.ndarray) and global_quat.ndim > 1:
                global_quat_1d = global_quat.flatten()
            else:
                global_quat_1d = global_quat

            # Calculate Euler angles for display
            euler_deg = calculate_euler_from_quaternion(global_quat_1d)
            
            # Create IMU data with the transformed values
            imu_data = IMUData(
                timestamp=timestamp,
                device_id=device_id,
                accelerometer=linear_accel,
                gyroscope=global_gyro,
                quaternion=global_quat,
                euler=euler_deg
            )
            
            # Update visualization
            is_calibrated = global_alignment_complete and device_calibrated
            is_reference = device_id == self.calibrator.reference_device
            self.visualizer.update_device_data(imu_data, is_calibrated, is_reference)
                    
        except Exception as e:
            logger.warning(f"Error processing {device_id} data: {e}")
            import traceback
            logger.debug(traceback.format_exc())


    def get_t_pose_calibration(self):
        """
        Get the current T-pose calibration data
        
        Returns:
            tuple: (smpl2imu, device2bone, acc_offsets, gyro_offsets) if T-pose calibration has been performed
        """
        print("get_t_pose_calibration called")
        # Change this line to call the correct method
        result = self.calibrator.get_tpose_alignment()
        print(f"get_t_pose_calibration result: {result}")
        return result

    def process_data(self):
        """Process any pending data from the socket receiver"""
        # Get next data packet
        data_packet = self.socket_receiver.get_data()
        
        # Process if we have data
        if data_packet:
            message, addr = data_packet
            self._parse_data(message, addr)
    
    def run(self):
        """Main application loop"""
        # Start the socket receiver and API server
        self.socket_receiver.start()
        self.api_server.start()
        
        self.running = True
        
        try:
            while self.running:
                # Handle visualization events
                event = self.visualizer.handle_events()
                
                if event == "quit":
                    self.running = False
                elif event == "calibrate":
                    self.calibrate_all_devices()
                elif event == "reset_calibration":
                    self.reset_calibration()
                elif event and event.startswith("select_reference:"):
                    # Handle reference device selection
                    device_id = event.split(":", 1)[1]
                    self._select_reference_device(device_id)
                
                # Check for external calibration request
                if self.calibration_requested:
                    self.calibrate_all_devices()
                    self.calibration_requested = False
                
                # Check for external reset request
                if self.reset_requested:
                    self.reset_calibration()
                    self.reset_requested = False
                
                # Process incoming data
                self.process_data()
                
                # Render visualization
                self.visualizer.render()
        
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        
        finally:
            self.running = False
            self.socket_receiver.stop()
            self.api_server.stop()
            self.visualizer.cleanup()

def main():
    print("=============================")
    print("IMU Receiver with Enhanced Calibration")
    print("=============================")
    print("Device Positioning for Calibration:")
    print("  All devices: Place vertically with screen/front facing you")
    print()
    print("Global frame:    X:left, Y:up, Z:forward (into screen)")
    print("Visualization:   All devices shown in global frame")
    print()
    print("Calibration:")
    print("  1. Select which device sets the reference frame")
    print("  2. Position the device vertically with screen facing you")
    print("  3. Calibrate all devices relative to this reference")
    print("  4. Use RESET button to clear calibration and start over")
    print()
    print("External API: Available on port 9001")
    
    receiver = IMUReceiver(data_port=8001, api_port=9001)
    receiver.run()

if __name__ == "__main__":
    main()