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
        
        # Calibration data
        self.smpl2imu = None
        self.device2bone = None
        self.acc_offsets = None
        
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
                        ori_frame[i] = device_data.quaternion
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
            
            # Set identity quaternions
            quats[:, 3] = 1.0
            
            self._quat_buffer.append(quats)
            self._acc_buffer.append(accs)
    
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
    
    def update_latest_data(self):
        """Update buffers with latest data"""
        # Get all device data
        all_data = self.api.get_all_device_data()
        
        # Get calibrated devices
        calibrated_devices = [d for d in all_data if all_data[d].is_calibrated]
        
        # Device count in MobilePoseR
        device_count = 5
        
        # Create new frame
        ori_frame = np.zeros((device_count, 4))
        acc_frame = np.zeros((device_count, 3))
        
        # Set identity quaternions by default
        ori_frame[:, 3] = 1.0
        
        # Fill data from calibrated devices
        for i, device_id in enumerate(calibrated_devices):
            if i >= device_count:
                break
            
            device_data = all_data[device_id]
            ori_frame[i] = device_data.quaternion
            acc_frame[i] = device_data.acceleration
        
        # Update buffers
        self._quat_buffer.append(ori_frame)
        self._acc_buffer.append(acc_frame)
        
        # Trim buffers if too long
        if len(self._quat_buffer) > self.buffer_len:
            self._quat_buffer.pop(0)
        if len(self._acc_buffer) > self.buffer_len:
            self._acc_buffer.pop(0)
        
        return ori_frame, acc_frame
    
    def get_t_pose_calibration(self):
        """
        Get T-pose calibration data from receiver
        
        Returns:
            tuple: (smpl2imu, device2bone, acc_offsets)
        """
        # Send request to get T-pose calibration data
        response = self.api._send_request("get_t_pose_calibration")
        
        if not response or "error" in response:
            print("Error getting T-pose calibration data")
            return None, None, None
            
        # Deserialize the calibration data
        calib_data = response.get("calibration", {})
        
        # Convert numpy arrays to torch tensors
        smpl2imu = torch.tensor(np.array(calib_data.get("smpl2imu", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])), 
                                dtype=torch.float32)
        
        # Convert device2bone dictionary
        device2bone = {}
        for device_id, matrix_data in calib_data.get("device2bone", {}).items():
            device2bone[device_id] = torch.tensor(np.array(matrix_data), dtype=torch.float32)
            
        # Convert acc_offsets dictionary
        acc_offsets = {}
        for device_id, offset_data in calib_data.get("acc_offsets", {}).items():
            acc_offsets[device_id] = torch.tensor(np.array(offset_data), dtype=torch.float32)
        
        return smpl2imu, device2bone, acc_offsets


def get_input():
    global running, start_recording
    while running:
        c = input()
        if c == 'q':
            running = False
        elif c == 'r':
            start_recording = True
        elif c == 's':
            start_recording = False


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
        print("Please calibrate devices in IMU_receiver first.")
        print("Press the CALIBRATE button in the IMU_receiver window.")
        exit(1)
    
    print(f"✅ Found {len(calibrated_devices)} calibrated devices: {', '.join(calibrated_devices)}")

    # Perform T-pose calibration
    input('Now please wear all IMUs correctly and press any key to continue.')
    for i in range(3, 0, -1):
        print('\rStand straight in T-pose and be ready. The calibration will begin after %d seconds.' % i, end='')
        time.sleep(1)
    print('\nStand straight in T-pose. Keep the pose for 3 seconds ...', end='')

    # Request T-pose calibration from IMU receiver
    if imu_set.api.request_t_pose_calibration():
        print('\tT-pose calibration completed successfully.')
        
        # Get the T-pose calibration data
        smpl2imu, device2bone, acc_offsets = imu_set.get_t_pose_calibration()
        
        if smpl2imu is None or not device2bone or not acc_offsets:
            print('\n❌ Failed to get T-pose calibration data!')
            print('Please try again with a different reference device.')
            exit(1)
            
        # Store in the adapter for later use
        imu_set.smpl2imu = smpl2imu
        imu_set.device2bone = device2bone
        imu_set.acc_offsets = acc_offsets
        
        # Prepare device2bone for MobilePoseR format
        # Convert dictionary to tensor for easier matrix operations
        device_bone_tensor = torch.zeros((5, 3, 3), dtype=torch.float32)
        for i, device_id in enumerate(calibrated_devices[:5]):
            if device_id in device2bone:
                device_bone_tensor[i] = device2bone[device_id]
        
        # Prepare acc_offsets for MobilePoseR format
        acc_offsets_tensor = torch.zeros((5, 3, 1), dtype=torch.float32)
        for i, device_id in enumerate(calibrated_devices[:5]):
            if device_id in acc_offsets:
                acc_offsets_tensor[i] = acc_offsets[device_id]
    else:
        print('\n❌ T-pose calibration failed!')
        print('Please make sure all devices are calibrated in IMU_receiver.')
        exit(1)

    # The IMU receiver already has the calibration, we just need to get the latest data
    # for model inference
    print('\nReady for pose estimation. Press q to quit')

    # Load model
    model = load_model(paths.weights_file)

    # Setup Unity server for visualization
    if args.vis:
        server_for_unity = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_for_unity.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
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
    accs, oris = [], []
    raw_accs, raw_oris = [], []
    poses, trans = [], []

    # Main loop for collecting data and running inference
    model.eval()
    try:
        while running:
            # Update clock for FPS calculation
            clock.tick(datasets.fps)
            
            # Get latest IMU data
            ori_raw, acc_raw = imu_set.update_latest_data()
            
            # Skip if no data
            if ori_raw is None or acc_raw is None:
                continue
            
            # Convert to torch tensors and reshape for processing
            ori_raw = torch.tensor(ori_raw, dtype=torch.float32).unsqueeze(0)
            acc_raw = torch.tensor(acc_raw, dtype=torch.float32).unsqueeze(0)
            
            if args.save:
                raw_accs.append(acc_raw)
                raw_oris.append(ori_raw)
            
            # Convert quaternions to rotation matrices
            ori_raw = quaternion_to_rotation_matrix(ori_raw).view(-1, n_imus, 3, 3)
            
            # Apply calibration using the saved T-pose data
            glb_acc = (smpl2imu.matmul(acc_raw.view(-1, n_imus, 3, 1)) - acc_offsets_tensor).view(-1, n_imus, 3)
            glb_ori = smpl2imu.matmul(ori_raw).matmul(device_bone_tensor)
            
            if args.save:
                accs.append(glb_acc)
                oris.append(glb_ori)
            
            # Normalization 
            _acc = glb_acc.view(-1, 5, 3)[:, [1, 4, 3, 0, 2]] / amass.acc_scale
            _ori = glb_ori.view(-1, 5, 3, 3)[:, [1, 4, 3, 0, 2]]
            acc = torch.zeros_like(_acc)
            ori = torch.zeros_like(_ori)

            # Device combo
            combo = 'lw_h'
            # meaning we will use watch on left wristm headphone or glasses on head
            c = amass.combos[combo]

            # Filter and concat input
            acc[:, c] = _acc[:, c] 
            ori[:, c] = _ori[:, c]
            
            imu_input = torch.cat([acc.flatten(1), ori.flatten(1)], dim=1)
            
            # Predict pose and translation
            with torch.no_grad():
                output = model.forward_online(imu_input.squeeze(0), [imu_input.shape[0]])
                pred_pose = output[0] # [24, 3, 3]
                pred_tran = output[2] # [3]
            
            if args.save:
                poses.append(pred_pose)
                trans.append(pred_tran)
            
            # Convert rotation matrix to axis angle
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
                accs, oris, raw_accs, raw_oris, poses, trans = [], [], [], [], [], []
            elif not start_recording and is_recording:
                is_recording = False
                print("\nStopped recording")
                
                # Save the recorded data
                if args.save and accs:
                    data = {
                        'raw_acc': torch.cat(raw_accs, dim=0),
                        'raw_ori': torch.cat(raw_oris, dim=0),
                        'acc': torch.cat(accs, dim=0),
                        'ori': torch.cat(oris, dim=0),
                        'pose': torch.cat(poses, dim=0),
                        'tran': torch.cat(trans, dim=0),
                        'calibration': {
                            'smpl2imu': smpl2imu,
                            'device2bone': device_bone_tensor
                        }
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
        get_input_thread.join(timeout=1.0)
        imu_set.stop_reading()
        
        # Save final data if recording
        if args.save and is_recording and accs:
            data = {
                'raw_acc': torch.cat(raw_accs, dim=0),
                'raw_ori': torch.cat(raw_oris, dim=0),
                'acc': torch.cat(accs, dim=0),
                'ori': torch.cat(oris, dim=0),
                'pose': torch.cat(poses, dim=0),
                'tran': torch.cat(trans, dim=0),
                'calibration': {
                    'smpl2imu': smpl2imu,
                    'device2bone': device_bone_tensor
                }
            }
            save_path = paths.dev_data / f'dev_{int(time.time())}.pt'
            torch.save(data, save_path)
            print(f"Final recording saved to {save_path}")
            
        print('Finish.')