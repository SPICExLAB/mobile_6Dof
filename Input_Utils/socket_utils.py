"""
Socket utilities for receiving IMU data from devices.

This module provides classes for:
1. Receiving UDP data from sensors
2. Handling API requests via UDP
"""

import socket
import threading
import time
import logging
import json
import queue

logger = logging.getLogger(__name__)

class SocketReceiver:
    """Handles UDP socket connections for receiving sensor data"""
    
    def __init__(self, host='0.0.0.0', port=8001, timeout=0.1):
        """Initialize the socket receiver"""
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket = None
        self.running = False
        self.data_queue = queue.Queue()
        self.receive_thread = None
        
        # Device detection - track IP addresses for device assignment
        self.device_ip_mapping = {}
        self.next_device_assignment = ['phone', 'watch', 'glasses']  # Assignment order
    
    def start(self):
        """Start the socket receiver"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind((self.host, self.port))
            self.socket.settimeout(self.timeout)
            
            logger.info(f"UDP server started on port {self.port}")
            logger.info("Waiting for data from:")
            logger.info("  - iOS devices (phone/watch/headphone)")
            logger.info("  - AR Glasses (Unity app)")
            
            self.running = True
            
            # Start receiver thread
            self.receive_thread = threading.Thread(target=self._receive_loop)
            self.receive_thread.daemon = True
            self.receive_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start socket receiver: {e}")
            return False
    
    def stop(self):
        """Stop the socket receiver"""
        self.running = False
        
        if self.receive_thread and self.receive_thread.is_alive():
            self.receive_thread.join(timeout=1.0)
            
        if self.socket:
            self.socket.close()
            self.socket = None
    
    def _receive_loop(self):
        """Main loop for receiving data"""
        while self.running:
            try:
                data, addr = self.socket.recvfrom(4096)
                message = data.decode('utf-8').strip()
                
                # Queue the data along with the address for processing
                self.data_queue.put((message, addr))
                    
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    logger.error(f"Receive error: {e}")
    
    def get_data(self):
        """Get next data packet from the queue"""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_device_id_for_ip(self, ip_address, preferred_type='glasses'):
        """Get device ID for an IP address, assigning new ones as needed"""
        if ip_address in self.device_ip_mapping:
            return self.device_ip_mapping[ip_address]
        
        # For AR glasses, always assign 'glasses' if available
        if preferred_type == 'glasses' and 'glasses' not in self.device_ip_mapping.values():
            self.device_ip_mapping[ip_address] = 'glasses'
            logger.info(f"Assigned AR Glasses to IP {ip_address}")
            return 'glasses'
        
        # Find next available device type
        for device_type in self.next_device_assignment:
            if device_type not in self.device_ip_mapping.values():
                self.device_ip_mapping[ip_address] = device_type
                logger.info(f"Assigned {device_type} to IP {ip_address}")
                return device_type
        
        # Fallback if all devices assigned
        fallback = f"device_{len(self.device_ip_mapping)}"
        self.device_ip_mapping[ip_address] = fallback
        logger.warning(f"All standard device types assigned, using {fallback} for IP {ip_address}")
        return fallback


class ApiServer:
    """Provides an API server for external access to IMU data"""
    
    def __init__(self, host='127.0.0.1', port=9001, timeout=0.1, callbacks=None):
        """
        Initialize the API server
        
        Args:
            host: Host IP to bind to
            port: Port to listen on
            timeout: Socket timeout in seconds
            callbacks: Dictionary of callback functions for handling requests
                       Should include at least:
                       - 'get_device_data': function(device_id) -> dict
                       - 'get_active_devices': function() -> list
                       - 'calibrate': function() -> bool
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket = None
        self.running = False
        self.api_thread = None
        
        # Default empty callbacks
        self.callbacks = callbacks or {}
    
    def start(self):
        """Start the API server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind((self.host, self.port))
            self.socket.settimeout(self.timeout)
            
            logger.info(f"API server started on port {self.port}")
            
            self.running = True
            
            # Start API server thread
            self.api_thread = threading.Thread(target=self._api_loop)
            self.api_thread.daemon = True
            self.api_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
            return False
    
    def stop(self):
        """Stop the API server"""
        self.running = False
        
        if self.api_thread and self.api_thread.is_alive():
            self.api_thread.join(timeout=1.0)
            
        if self.socket:
            self.socket.close()
            self.socket = None
    
    def _api_loop(self):
        """Main loop for API server"""
        while self.running:
            try:
                data, addr = self.socket.recvfrom(1024)
                request = data.decode('utf-8')
                response = self._handle_request(request)
                self.socket.sendto(response.encode('utf-8'), addr)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    logger.error(f"API server error: {e}")
    
    def _handle_request(self, request):
        """Handle API requests and return JSON responses"""
        try:
            if request == "get_all_devices":
                devices = {}
                active_devices = self.callbacks.get('get_active_devices', lambda: [])()
                for device_id in active_devices:
                    device_data = self.callbacks.get('get_device_data', lambda x: None)(device_id)
                    if device_data:
                        devices[device_id] = device_data
                return json.dumps(devices)
                
            elif request.startswith("get_device:"):
                device_id = request.split(":", 1)[1]
                device_data = self.callbacks.get('get_device_data', lambda x: None)(device_id)
                return json.dumps(device_data) if device_data else json.dumps(None)
                
            elif request == "get_active_devices":
                active = self.callbacks.get('get_active_devices', lambda: [])()
                return json.dumps(active)
                
            elif request == "calibrate":
                result = self.callbacks.get('calibrate', lambda: False)()
                return json.dumps({"status": "calibration_requested", "success": result})
            
            elif request == "calibrate_t_pose":
                result = self.callbacks.get('calibrate_t_pose', lambda: None)()
                return json.dumps({"status": "t_pose_calibration_requested", "success": result is not None})
                
            elif request == "get_t_pose_calibration":
                t_pose_data = self.callbacks.get('get_t_pose_calibration', lambda: None)()
                if t_pose_data:
                    smpl2imu, device2bone, acc_offsets = t_pose_data
                    
                    # Convert to JSON-serializable format
                    result = {
                        "calibration": {
                            "smpl2imu": smpl2imu.tolist() if hasattr(smpl2imu, 'tolist') else smpl2imu,
                            "device2bone": {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in device2bone.items()},
                            "acc_offsets": {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in acc_offsets.items()}
                        }
                    }
                    return json.dumps(result)
                else:
                    return json.dumps({"error": "T-pose calibration data not available"})
            
            elif request == "ping":
                return json.dumps({"status": "ok", "timestamp": time.time()})
                
        except Exception as e:
            logger.error(f"Error handling API request: {e}")
            return json.dumps({"error": str(e)})
        
        return json.dumps({"error": "Unknown request"})