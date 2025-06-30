"""Main IMU Visualizer - Enhanced with Reference Device Selection"""

import pygame
import numpy as np
import time
import logging
from collections import deque

from .utils.colors import Colors
from .utils.fonts import FontManager
from .components.device_panel import DevicePanel
from .components.waveform_panel import WaveformPanel
from .components.reference_panel import ReferencePanel
from .components.calibration_button import CalibrationButton
from .layouts.device_grid import DeviceGridLayout

logger = logging.getLogger(__name__)

class IMUVisualizer:
    """Enhanced IMU visualizer with reference device selection before calibration"""
    
    def __init__(self, width=1400, height=800):
        pygame.init()
        
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("IMU Receiver - iOS Devices + AR Glasses Visualization")
        
        # Initialize managers
        self.font_manager = FontManager()
        
        # Layout dimensions
        self.left_panel_width = 500
        self.right_panel_width = width - self.left_panel_width - 30
        
        # Device data storage
        self.device_data = {}
        
        # Reference device for calibration
        self.reference_device = None
        # Selected reference device (before calibration)
        self.selected_reference_device = None
        
        # Gravity toggle state (True = remove gravity, False = include gravity)
        self.gravity_removal_enabled = True
        
        # Initialize components
        self._init_components()
        
        # Device order - now includes AR glasses
        self.device_order = ['phone', 'headphone', 'watch', 'glasses']
        
        # Waveform settings
        self.waveform_history = 300
        
        logger.info("Enhanced IMU Visualizer initialized with reference device selection before calibration")
    
    def _init_components(self):
        """Initialize UI components"""
        # Reference panel (top of left panel)
        self.reference_panel = ReferencePanel(
            self.screen, 
            x=10, 
            y=20, 
            width=self.left_panel_width, 
            height=150
        )
        
        # Device grid layout manager (updated for AR glasses)
        self.device_layout = DeviceGridLayout(box_size=180, margin=20)
        
        # Calibration button (bottom of left panel)
        button_width = 180
        button_x = (self.left_panel_width - button_width) // 2 + 10
        button_y = self.height - 80
        self.calibration_button = CalibrationButton(
            self.screen, 
            button_x, 
            button_y
        )
        
        # Waveform panel (right side)
        self.waveform_panel = WaveformPanel(
            self.screen,
            x=self.left_panel_width + 20,
            y=10,
            width=self.right_panel_width,
            height=self.height - 20
        )
        
        # Device panels will be created dynamically
        self.device_panels = {}
    
    def update_device_data(self, imu_data, is_calibrated: bool):
        """Update device data for visualization"""
        device_id = imu_data.device_id
        
        if device_id not in self.device_data:
            self.device_data[device_id] = {
                'accel_history': deque(maxlen=self.waveform_history),
                'gyro_history': deque(maxlen=self.waveform_history),
                'quaternion': np.array([0, 0, 0, 1]),
                'euler': None,
                'last_update': 0,
                'sample_count': 0,
                'is_calibrated': False,
                'frequency': 0,
                'frequency_counter': 0,
                'frequency_timer': time.time(),
                'log_counter': 0  # Add counter for periodic logging
            }
        
        data = self.device_data[device_id]
        
        # Update data
        data['accel_history'].append(imu_data.accelerometer)
        data['gyro_history'].append(imu_data.gyroscope)
        data['quaternion'] = imu_data.quaternion
        data['euler'] = imu_data.euler  # Store Euler angles for glasses
        data['last_update'] = time.time()
        data['sample_count'] += 1
        data['is_calibrated'] = is_calibrated
        
        # Periodic logging of device orientation (every 120 frames)
        data['log_counter'] = (data['log_counter'] + 1) % 120
        if data['log_counter'] == 0:
            try:
                euler = calculate_euler_from_quaternion(imu_data.quaternion)
                status = "CALIBRATED" if is_calibrated else "UNCALIBRATED"
                is_ref = "REFERENCE" if device_id == self.reference_device else ""
                logger.debug(f"PERIODIC {status} {is_ref}: {device_id} orientation: "
                            f"Roll={euler[0]:.1f}°, Pitch={euler[1]:.1f}°, Yaw={euler[2]:.1f}°")
            except:
                pass
        
        # Calculate frequency
        data['frequency_counter'] += 1
        if data['frequency_counter'] % 30 == 0:
            current_time = time.time()
            time_diff = current_time - data['frequency_timer']
            if time_diff > 0:
                data['frequency'] = 30.0 / time_diff
                data['frequency_timer'] = current_time
    
    def get_active_devices(self):
        """Get list of currently active devices"""
        active = []
        current_time = time.time()
        for device_id, data in self.device_data.items():
            if current_time - data['last_update'] < 2.0:
                active.append(device_id)
        return active
    
    def get_gravity_enabled(self):
        """Get current gravity removal state"""
        return self.gravity_removal_enabled
    
    def toggle_gravity_removal(self):
        """Toggle gravity removal for AR glasses"""
        self.gravity_removal_enabled = not self.gravity_removal_enabled
        logger.info(f"Gravity removal {'enabled' if self.gravity_removal_enabled else 'disabled'} for AR glasses")
        return "toggle_gravity"
    
    def select_reference_device(self, device_id):
        """Select a device as reference before calibration"""
        if device_id in self.device_data:
            self.selected_reference_device = device_id
            logger.info(f"Selected {device_id} as reference device")
            return True
        return False
    
    def get_selected_reference_device(self):
        """Get the currently selected reference device"""
        return self.selected_reference_device
    
    def set_reference_device(self, device_id):
        """Set the active reference device after calibration"""
        if device_id in self.device_data:
            self.reference_device = device_id
            # Clear selected reference since we now have an actual reference
            self.selected_reference_device = None
            logger.info(f"Set {device_id} as active reference device")
            return True
        return False
    
    def _update_device_panels(self):
        """Update device panels based on active devices"""
        # Always create panels for all possible devices
        all_devices = self.device_order.copy()
        
        # Get positions for all devices (both active and inactive)
        active_devices = self.get_active_devices()
        positions = self.device_layout.calculate_positions(all_devices)
        
        # Mark which ones are active
        for device_name in positions:
            positions[device_name]['active'] = device_name in active_devices
        
        # Create or update device panels
        for device_name in self.device_order:
            if device_name in positions:
                if device_name not in self.device_panels:
                    self.device_panels[device_name] = DevicePanel(
                        self.screen, 
                        device_name, 
                        positions[device_name]
                    )
                else:
                    # Update position info
                    self.device_panels[device_name].position_info = positions[device_name]
                    self.device_panels[device_name].center = positions[device_name]['center']
                    self.device_panels[device_name].size = positions[device_name]['size']
                    self.device_panels[device_name].is_active = positions[device_name]['active']
    
    def handle_events(self):
        """Handle pygame events and return action"""
        mouse_pos = pygame.mouse.get_pos()
        self.calibration_button.update(mouse_pos)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Check calibration button
                if self.calibration_button.is_clicked(event):
                    return "calibrate"
                
                # Check device panel clicks
                for device_name, device_panel in self.device_panels.items():
                    if device_panel.is_active:
                        panel_action = device_panel.handle_click(event.pos)
                        if panel_action == 'toggle_gravity':
                            return self.toggle_gravity_removal()
                        elif panel_action == 'select_reference':
                            if self.select_reference_device(device_name):
                                return f"select_reference:{device_name}"
                
                # Check waveform panel clicks
                waveform_action = self.waveform_panel.handle_click(event.pos)
                if waveform_action:
                    action_type, device_name = waveform_action
                    # Handle waveform actions
        
        return None
    
    def render(self):
        """Main render function"""
        # Clear screen
        self.screen.fill(Colors.BG)
        
        # Draw left panel background
        pygame.draw.rect(self.screen, Colors.PANEL, 
                        (10, 10, self.left_panel_width, self.height - 20))
        
        # Draw components
        self.reference_panel.draw()
        
        # Update and draw device panels
        self._update_device_panels()
        for device_name in self.device_order:
            if device_name in self.device_panels:
                device_data = self.device_data.get(device_name, None)
                if device_data and device_name in self.get_active_devices():
                    is_calibrated = device_data.get('is_calibrated', False)
                    is_reference = (device_name == self.reference_device)
                    is_selected_as_reference = (device_name == self.selected_reference_device)
                    
                    # Pass all status flags to device panels
                    self.device_panels[device_name].draw(
                        device_data, 
                        is_calibrated, 
                        is_reference,
                        is_selected_as_reference,
                        self.gravity_removal_enabled
                    )
                else:
                    # Draw inactive panel
                    self.device_panels[device_name].draw(None, False, False, False, True)
        
        # Draw calibration button
        self.calibration_button.draw()
        
        # Draw waveform panel
        self.waveform_panel.draw(self.device_data)
        
        # Draw connection status and calibration info
        self._draw_connection_status()
        
        # Update display
        pygame.display.flip()
    
    def _draw_connection_status(self):
        """Draw connection status and calibration information"""
        # Show active device count and types
        active_devices = self.get_active_devices()
        
        status_text = f"Active Devices: {len(active_devices)}"
        if active_devices:
            device_names = []
            for device in active_devices:
                if device == 'glasses':
                    device_names.append("AR Glasses")
                elif device == 'phone':
                    device_names.append("Phone")
                elif device == 'watch':
                    device_names.append("Watch")
                elif device == 'headphone':
                    device_names.append("AirPods")
                else:
                    device_names.append(device.title())
            
            status_text += f" ({', '.join(device_names)})"
        
        status_surface = self.font_manager.render_text(status_text, 'small', Colors.TEXT_SECONDARY)
        self.screen.blit(status_surface, (20, self.height - 25))
        
        # Show reference device info
        if self.reference_device:
            ref_text = f"Reference Device: {self.reference_device.upper()}"
            ref_surface = self.font_manager.render_text(ref_text, 'small', Colors.REFERENCE)
            self.screen.blit(ref_surface, (300, self.height - 25))
        elif self.selected_reference_device:
            ref_text = f"Selected Reference: {self.selected_reference_device.upper()} (press CALIBRATE to confirm)"
            ref_surface = self.font_manager.render_text(ref_text, 'small', Colors.REFERENCE)
            self.screen.blit(ref_surface, (300, self.height - 25))
        
        # Show listening port
        port_text = "Listening on UDP port 8001"
        port_surface = self.font_manager.render_text(port_text, 'tiny', Colors.TEXT_TERTIARY)
        self.screen.blit(port_surface, (20, self.height - 45))
        
        # Show gravity removal status for AR glasses
        if 'glasses' in active_devices:
            gravity_status = f"AR Glasses Gravity: {'Removed' if self.gravity_removal_enabled else 'Included'}"
            gravity_color = (100, 200, 100) if self.gravity_removal_enabled else (200, 100, 100)
            gravity_surface = self.font_manager.render_text(gravity_status, 'tiny', gravity_color)
            self.screen.blit(gravity_surface, (300, self.height - 45))
        
        # Show calibration help if no reference selected
        if not self.selected_reference_device and not self.reference_device:
            help_text = "Select a reference device to define the global coordinate system"
            help_surface = self.font_manager.render_text(help_text, 'small', Colors.REFERENCE)
            help_rect = help_surface.get_rect(centerx=self.left_panel_width//2 + 10, y=self.height - 110)
            self.screen.blit(help_surface, help_rect)
        elif self.selected_reference_device and not self.reference_device:
            help_text = "Press CALIBRATE to confirm reference device and perform calibration"
            help_surface = self.font_manager.render_text(help_text, 'small', Colors.REFERENCE)
            help_rect = help_surface.get_rect(centerx=self.left_panel_width//2 + 10, y=self.height - 110)
            self.screen.blit(help_surface, help_rect)
    
    def cleanup(self):
        """Clean up resources"""
        pygame.quit()
        logger.info("Enhanced IMU visualizer with reference device selection cleaned up")