import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import deque

from mobileposer.config import *
from mobileposer.constants import *


class SensorData:
    """Store sensor data from devices."""
    def __init__(self):
        self.raw_acc_buffer = {
            _id: deque(np.zeros((BUFFER_SIZE, 3)), maxlen=BUFFER_SIZE) 
            for _id in sensor.device_ids.values()
        }
        self.raw_ori_buffer = {
            _id: deque(np.array([[0, 0, 0, 1]] * BUFFER_SIZE), maxlen=BUFFER_SIZE) 
            for _id in sensor.device_ids.values()
        }
        self.calibration_quats = {
            _id: np.array([0, 0, 0, 1]) 
            for _id in sensor.device_ids.values()
        }
        self.virtual_acc = {
            _id: np.zeros((1, 3)) 
            for _id in sensor.device_ids.values()
        }
        self.virtual_ori = {
            _id: np.array([0, 0, 0, 1]) 
            for _id in sensor.device_ids.values()
        }
        self.reference_times = {
            _id: None 
            for _id in sensor.device_ids.values()
        }

    def update(self, device_id, curr_acc, curr_ori, timestamps):
        if self.reference_times[device_id] is None:
            self.reference_times[device_id] = [timestamps[0], timestamps[1]]

        curr_timestamp = (
            self.reference_times[device_id][0] + 
            timestamps[1] - self.reference_times[device_id][1]
        )

        self.raw_acc_buffer[device_id].append(curr_acc.flatten())
        self.raw_ori_buffer[device_id].append(curr_ori.flatten())

        return curr_timestamp
    
    def calibrate(self):
        for _id, ori_buffer in self.raw_ori_buffer.items():
            if len(ori_buffer) < 30:
                print(f"Not enough data to calibrate for device {_id}.")
                continue
            # Convert deque to list and then to Rotation objects
            quaternions = np.array(ori_buffer)[-30:]
            rotations = R.from_quat(quaternions)
            # Compute the mean rotation
            mean_rotation = rotations.mean()
            self.calibration_quats[_id] = mean_rotation.as_quat()

    def get_timestamps(self, device_id):
        return self.reference_times[device_id][-1]

    def get_orientation(self, device_id):
        return self.raw_ori_buffer[device_id][-1]

    def get_acceleration(self, device_id):
        return self.raw_acc_buffer[device_id][-1]

    def update_virtual(self, device_id, glb_acc, glb_ori):
        self.virtual_acc[device_id] = glb_acc.reshape(1, 3)
        self.virtual_ori[device_id] = glb_ori


def process_data(message):
    """Process the data from the sensors (e.g., iPhone, Apple Watch, etc.)."""
    message = message.strip()
    if not message:
        return
    message = message.decode('utf-8')
    if message == STOP:
        return
    if SEP not in message:
        return 

    try:
        device_id, raw_data_str = message.split(";")
        device_type, data_str = raw_data_str.split(":")
    except Exception as e:
        print("(1) Exception encountered: ", e)
        return

    # Parse data values
    data = []
    for d in data_str.strip().split(" "):
        try: 
            data.append(float(d))
        except Exception as e:
            print("(2) Exception encountered: ", e)
            continue
    
    # Smart parsing based on device type and data length
    if len(data) < 9:  # Minimum: 2 timestamps + 3 accel + 4 quat
        print(f"Insufficient data! Got {len(data)} values, need at least 9")
        return
    
    # Extract common data (timestamps, acceleration, quaternion)
    timestamps = data[:2]
    curr_acc = np.array(data[2:5]).reshape(1, 3)
    curr_ori = np.array(data[5:9]).reshape(1, 4)
    
    # Handle additional data based on device type and data length
    if device_type.lower() == "watch" and len(data) >= 12:
        # Watch has gyroscope data: timestamps + accel + quat + gyro
        gyro_data = data[9:12]  # Extract gyroscope data if present
        print(f"Watch gyro data: {gyro_data}")  # For debugging
    elif device_type.lower() == "phone" and len(data) >= 12:
        # Phone has roll, pitch, yaw: timestamps + accel + quat + rpy
        rpy_data = data[9:12]  # Extract Euler angles if present
        print(f"Phone RPY data: {rpy_data}")  # For debugging
    
    # Map device types to config.py device_ids keys
    device_type_lower = device_type.lower()
    
    # Create the key for sensor.device_ids lookup
    config_key = f"Left_{device_type_lower}"
    
    # Check if the key exists in sensor.device_ids
    if config_key in sensor.device_ids:
        config_device_id = sensor.device_ids[config_key]
        
        # Apply Version 2 remapping: Phone=0, Watch=1, AirPods=2
        if device_type_lower == 'phone':
            numerical_device_id = 0
        elif device_type_lower == 'watch':
            numerical_device_id = 1  # Watch gets position 1
        elif device_type_lower == 'headphone':
            numerical_device_id = 2  # AirPods get position 2
        else:
            numerical_device_id = config_device_id  # Fallback to config value
    else:
        print(f"Unknown device type: {device_type} (key: {config_key})")
        return
    
    # Create send string (legacy format for compatibility)
    if len(data) >= 12:
        send_str = f"w{data[8]}wa{data[5]}ab{data[6]}bc{data[7]}c"
    else:
        send_str = f"w{data[8]}wa{data[5]}ab{data[6]}bc{data[7]}c"

    # Apply coordinate frame fixes for AirPods (Right_Headphone equivalent)
    if device_type_lower == 'headphone':
        curr_euler = R.from_quat(curr_ori).as_euler("xyz").squeeze()
        fixed_euler = np.array([[curr_euler[0] * -1, curr_euler[2], curr_euler[1]]])
        curr_ori = R.from_euler("xyz", fixed_euler).as_quat().reshape(1, 4)
        curr_acc = np.array([[curr_acc[0, 0]*-1, curr_acc[0, 2], curr_acc[0, 1]]])

    return send_str, numerical_device_id, curr_acc, curr_ori, timestamps


def sensor2global(ori, acc, calibration_quats, device_id):
    """Convert the sensor data to the global inertial frame."""
    device_mean_quat = calibration_quats[device_id]

    og_mat = R.from_quat(ori).as_matrix() # rotation matrix from quaternion
    global_inertial_frame = R.from_quat(device_mean_quat).as_matrix()
    global_mat = (global_inertial_frame.T).dot(og_mat)
    global_ori = R.from_matrix(global_mat).as_quat()

    acc_ref_frame = og_mat.dot(acc) # align acc. to sensor frame of reference
    global_acc = (global_inertial_frame.T).dot(acc_ref_frame) # align acc. to world frame

    return  global_ori, global_acc