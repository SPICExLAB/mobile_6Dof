"""
MobilePoseR Live Demo with IMU_receiver Integration
File: live_demo.py

Usage:
1. First run: python IMU_receiver.py
2. Make sure devices are connected and calibrated
3. Then run: python live_demo.py
"""

import os
import time
import socket
import threading
import torch
import numpy as np
from datetime import datetime
from argparse import ArgumentParser
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
from pygame.time import Clock

# Import MobilePoseR modules
from mobileposer.articulate.math import *
from mobileposer.models import *
from mobileposer.utils.model_utils import *
from mobileposer.config import *

# Import our API
from imu_data_api import get_imu_api


class IMUSetAdapter:
    """
    Adapter class to make IMU_receiver compatible with MobilePoseR's IMUSet interface
    """
    def __init__(self, api_port=9001, buffer_len=40):
        """Initialize adapter with IMU API connection"""
        self.api = get_imu_api(api_port)
        self.buffer_len = buffer_len
        
        # Initialize buffers
        self._quat_buffer = []
        self._acc_buffer = []
        self._rot_buffer = []
        
        # Check active devices
        active_devices = self.api.get_active_devices()
        print(f"IMUSetAdapter: Found {len(active_devices)} active devices: {', '.join(active_devices)}")
    
    def get_mean_measurement_of_n_second(self, num_seconds=3, buffer_len=40):
        """
        Collect data for n seconds and return the mean orientation and acceleration
        
        Returns:
            Tuple of (mean_orientation, mean_acceleration) as torch tensors
        """
        print(f"Collecting data for {num_seconds} seconds...")
        
        # Get calibrated devices
        calibrated_devices = self.api.get_calibrated_devices()
        if not calibrated_devices:
            print("No calibrated devices found!")
            return None, None
        
        # Device count in MobilePoseR
        device_count = 5
        
        # Collect data for specified duration
        ori_buffer = []
        acc_buffer = []
        
        start_time = time.time()
        while time.time() - start_time < num_seconds:
            # Get current data from all devices
            all_data = self.api.get_all_device_data()
            
            # Initialize frame tensors
            ori_frame = np.zeros((device_count, 4))
            acc_frame = np.zeros((device_count, 3))
            
            # Fill data from calibrated devices
            for i, device_id in enumerate(calibrated_devices):
                if i >= device_count:
                    break
                
                if device_id in all_data:
                    device_data = all_data[device_id]
                    if device_data.is_calibrated:
                        # Get quaternion from rotation matrix
                        from scipy.spatial.transform import Rotation as R
                        r = R.from_matrix(device_data.rotation_matrix)
                        ori_frame[i] = r.as_quat()
                        acc_frame[i] = device_data.acceleration
            
            # Add to buffers
            ori_buffer.append(ori_frame)
            acc_buffer.append(acc_frame)
            
            # Brief pause
            time.sleep(0.01)
        
        # Calculate means
        if ori_buffer:
            ori_mean = np.mean(np.array(ori_buffer), axis=0)
            acc_mean = np.mean(np.array(acc_buffer), axis=0)
            
            # Normalize quaternions
            for i in range(device_count):
                if np.linalg.norm(ori_mean[i]) > 0:
                    ori_mean[i] = ori_mean[i] / np.linalg.norm(ori_mean[i])
            
            # Convert to torch tensors
            return torch.tensor(ori_mean, dtype=torch.float32), torch.tensor(acc_mean, dtype=torch.float32)
        else:
            print("No data collected!")
            return None, None
    
    def start_reading(self):
        """Start reading from IMU devices"""
        print("Starting IMU data reading...")
        # Pre-fill buffers with placeholder data
        device_count = 5
        
        for _ in range(self.buffer_len):
            quats = np.zeros((device_count, 4))
            accs = np.zeros((device_count, 3))
            rots = np.zeros((device_count, 3, 3))
            
            # Set identity quaternions and rotation matrices
            quats[:, 3] = 1.0
            for i in range(device_count):
                rots[i] = np.eye(3)
            
            self._quat_buffer.append(quats)
            self._acc_buffer.append(accs)
            self._rot_buffer.append(rots)
    
    def stop_reading(self):
        """Stop reading from IMU devices"""
        print("Stopping IMU data reading...")
        self.api.close()
    
    def get_current_buffer(self):
        """
        Get current orientation and acceleration buffers
        
        Returns:
            Tuple of (quaternion_buffer, acceleration_buffer) as torch tensors
        """
        # Get all active devices
        active_devices = self.api.get_active_devices()
        calibrated_devices = [d for d in active_devices if self.api.is_device_calibrated(d)]
        
        # Device count in MobilePoseR
        device_count = 5
        
        # Convert buffer to torch tensors
        q = torch.tensor(np.array(self._quat_buffer), dtype=torch.float32)
        a = torch.tensor(np.array(self._acc_buffer), dtype=torch.float32)
        
        return q, a
    
    def get_latest_data(self):
        """Update buffers with latest data"""
        # Get all device data
        all_data = self.api.get_all_device_data()
        
        # Device count in MobilePoseR
        device_count = 5
        
        # Create new frame
        ori_frame = np.zeros((device_count, 4))
        acc_frame = np.zeros((device_count, 3))
        rot_frame = np.zeros((device_count, 3, 3))
        
        # Set identity quaternions and rotation matrices by default
        ori_frame[:, 3] = 1.0
        for i in range(device_count):
            rot_frame[i] = np.eye(3)
        
        # Map device names to location mapping 
        # 0: Left wrist, 1: Right wrist, 2: Left thigh, 3: Right thigh, 4: Head
        device_map = {
            'watch': 0,       # Left wrist - index 0
            'headphone': 4,   # Head - index 4
            'glasses': 4      # Alternative for head - index 4
        }
        
        # Priority for head devices (glasses > headphone)
        has_head_device = False
        
        # Fill data from calibrated devices using the mapping
        for device_id, device_data in all_data.items():
            if not device_data.is_calibrated:
                continue
                
            # Skip if we don't have a mapping for this device
            if device_id not in device_map:
                continue
            
            # Special handling for head devices (glasses take priority)
            if device_id == 'glasses' and device_map[device_id] == 4:
                has_head_device = True
            elif device_id == 'headphone' and device_map[device_id] == 4 and has_head_device:
                # Skip headphone if we already have glasses data for head
                continue
                
            # Get the MobilePoser index for this device
            mobileposer_idx = device_map[device_id]
            if mobileposer_idx >= device_count:
                continue
            
            # Get quaternion from rotation matrix if available
            from scipy.spatial.transform import Rotation as R
            if hasattr(device_data, 'rotation_matrix') and device_data.rotation_matrix is not None:
                r = R.from_matrix(device_data.rotation_matrix)
                ori_frame[mobileposer_idx] = r.as_quat()
                rot_frame[mobileposer_idx] = device_data.rotation_matrix
            else:
                ori_frame[mobileposer_idx] = device_data.quaternion
            
            acc_frame[mobileposer_idx] = device_data.acceleration
        
        # Update buffers
        self._quat_buffer.append(ori_frame)
        self._acc_buffer.append(acc_frame)
        self._rot_buffer.append(rot_frame)
        
        # Trim buffers if too long
        if len(self._quat_buffer) > self.buffer_len:
            self._quat_buffer.pop(0)
        if len(self._acc_buffer) > self.buffer_len:
            self._acc_buffer.pop(0)
        if len(self._rot_buffer) > self.buffer_len:
            self._rot_buffer.pop(0)
        
        return ori_frame, acc_frame, rot_frame

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--vis", action='store_true')
    parser.add_argument("--save", action='store_true')
    args = parser.parse_args()
    
    # Specify device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Setup IMU collection using our adapter
    imu_set = IMUSetAdapter()

    # Check if devices are already calibrated
    print("Checking if devices are calibrated in IMU_receiver...")
    active_devices = imu_set.api.get_active_devices()
    calibrated_devices = [d for d in active_devices if imu_set.api.is_device_calibrated(d)]

    if not calibrated_devices:
        print("❌ No calibrated devices found!")
        print("Please complete Global Alignment in IMU_receiver first:")
        print("1. Select a reference device (e.g., watch)")
        print("2. Press the ALIGN GLOBAL button in the IMU_receiver window")
        exit(1)

    print(f"✅ Found {len(calibrated_devices)} calibrated devices: {', '.join(calibrated_devices)}")


    # Perform T-pose calibration if not already done
    input('Now please wear all IMUs correctly and press any key to continue.')
    for i in range(3, 0, -1):
        print('\rStand straight in T-pose and be ready. The calibration will begin after %d seconds.' % i, end='')
        time.sleep(1)
    print('\nStand straight in T-pose. Keep the pose for 3 seconds ...', end='')

    # Request T-pose calibration from IMU receiver
    if imu_set.api.request_t_pose_calibration():
        print('\tT-pose calibration completed successfully.')
    else:
        print('\n❌ T-pose calibration failed!')
        print('Please make sure all devices are calibrated in IMU_receiver.')
        exit(1)

    # Load model
    model = load_model(paths.weights_file)

    # Setup Unity server for visualization
    if args.vis:
        server_for_unity = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Fix for Windows which doesn't have SO_REUSEPORT
        if hasattr(socket, 'SO_REUSEPORT'):
            server_for_unity.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        else:
            # On Windows, use SO_REUSEADDR instead
            server_for_unity.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        server_for_unity.bind(('0.0.0.0', 8889))
        server_for_unity.listen(1)
        print('Server start. Waiting for unity3d to connect.')
        conn, addr = server_for_unity.accept()

    running = True
    clock = Clock()
    is_recording = False
    record_buffer = None
    start_recording = False

    get_input_thread = threading.Thread(target=get_input)
    get_input_thread.setDaemon(True)
    get_input_thread.start()

    n_imus = 5
    accs, oris, rots = [], [], []
    raw_accs, raw_oris = [], [], []
    poses, trans = [], []

    # Start reading data
    imu_set.start_reading()
    
    # Main loop for collecting data and running inference
    model.eval()
    try:
        while running:
            # Update clock for FPS calculation
            clock.tick(datasets.fps)
            
            # Get latest IMU data - properly mapped to expected indices
            ori_calib, acc_calib, rot_calib = imu_set.get_latest_data()
            
            # Skip if no data
            if ori_calib is None or acc_calib is None:
                continue
            
            # Convert to torch tensors
            ori_calib = torch.tensor(ori_calib, dtype=torch.float32).unsqueeze(0)
            acc_calib = torch.tensor(acc_calib, dtype=torch.float32).unsqueeze(0)
            rot_calib = torch.tensor(rot_calib, dtype=torch.float32).unsqueeze(0)
            
            if args.save:
                raw_accs.append(acc_calib)
                raw_oris.append(ori_calib)
                rots.append(rot_calib)
            
            # The data is already in the correct device order, but we still need to
            # apply MobilePoser's model input processing
            
            # Initialize empty tensors for the final input
            acc = torch.zeros(1, 5, 3, device=device)
            ori = torch.zeros(1, 5, 3, 3, device=device)
            
            # Device combo - we'll use 'lw_h' which corresponds to [0, 4] after our mapping
            combo = 'lw_h'  # left wrist and head
            c = amass.combos[combo]  # This should be [0, 4] for 'lw_h'
            
            # Scale accelerations
            acc_scaled = acc_calib / amass.acc_scale
            
            # Fill only the devices we need based on the combo
            for i, idx in enumerate(c):
                if idx < acc_scaled.shape[1]:
                    acc[0, idx] = acc_scaled[0, idx]
                    ori[0, idx] = rot_calib[0, idx]
            
            # Flatten and concatenate for model input
            imu_input = torch.cat([acc.flatten(1), ori.flatten(1)], dim=1).to(device)
            
            # Predict pose and translation
            with torch.no_grad():
                output = model.forward_online(imu_input.squeeze(0), [imu_input.shape[0]])
                pred_pose = output[0]  # [24, 3, 3]
                pred_tran = output[2]  # [3]
            
            if args.save:
                poses.append(pred_pose)
                trans.append(pred_tran)
            
            # Convert rotation matrix to axis angle for Unity visualization
            pose = rotation_matrix_to_axis_angle(pred_pose.view(1, 216)).view(72)
            tran = pred_tran
            
            # Send pose to Unity if visualization is enabled
            if args.vis:
                s = ','.join(['%g' % v for v in pose]) + '#' + \
                    ','.join(['%g' % v for v in tran]) + '$'
                conn.send(s.encode('utf8'))
            
            # Handle recording if requested
            if start_recording and not is_recording:
                is_recording = True
                print("\nStarted recording")
                accs, oris, rots, raw_accs, raw_oris, poses, trans = [], [], [], [], [], [], []
            elif not start_recording and is_recording:
                is_recording = False
                print("\nStopped recording")
                
                # Save the recorded data
                if args.save and poses:
                    data = {
                        'raw_acc': torch.cat(raw_accs, dim=0),
                        'raw_ori': torch.cat(raw_oris, dim=0),
                        'rot': torch.cat(rots, dim=0),
                        'acc': torch.cat(accs, dim=0) if accs else None,
                        'ori': torch.cat(oris, dim=0) if oris else None,
                        'pose': torch.cat(poses, dim=0),
                        'tran': torch.cat(trans, dim=0)
                    }
                    save_path = paths.dev_data / f'dev_{int(time.time())}.pt'
                    torch.save(data, save_path)
                    print(f"Recording saved to {save_path}")
            
            # Debug output
            if os.getenv("DEBUG") is not None:
                print('\r', '(recording)' if is_recording else '', 'Output FPS:', clock.get_fps(), end='')
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Stop all threads and connections
        running = False
        get_input_thread.join(timeout=1.0)
        imu_set.stop_reading()
        
        # Save final data if recording
        if args.save and is_recording and poses:
            data = {
                'raw_acc': torch.cat(raw_accs, dim=0),
                'raw_ori': torch.cat(raw_oris, dim=0),
                'rot': torch.cat(rots, dim=0),
                'acc': torch.cat(accs, dim=0) if accs else None,
                'ori': torch.cat(oris, dim=0) if oris else None,
                'pose': torch.cat(poses, dim=0),
                'tran': torch.cat(trans, dim=0)
            }
            save_path = paths.dev_data / f'dev_{int(time.time())}.pt'
            torch.save(data, save_path)
            print(f"Final recording saved to {save_path}")
            
        print('Finish.')