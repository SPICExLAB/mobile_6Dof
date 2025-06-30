"""Waveform panel component for IMU data visualization"""

import pygame
import numpy as np
from ..utils.colors import Colors
from ..utils.fonts import FontManager
from .device_waveform import DeviceWaveform

class WaveformPanel:
    """Displays IMU sensor waveforms"""
    
    def __init__(self, screen, x, y, width, height):
        self.screen = screen
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        
        self.font_manager = FontManager()
        
        # Device waveform components
        self.device_waveforms = {}
        
        # Zoom levels for each device
        self.zoom_levels = {}
        self.zoom_presets = [
            ((-0.5, 0.5), (-0.2, 0.2)),     # Very zoomed in
            ((-1.0, 1.0), (-0.5, 0.5)),     # Zoomed in
            ((-3.0, 3.0), (-2.0, 2.0)),     # Default
            ((-6.0, 6.0), (-4.0, 4.0)),     # Zoomed out
            ((-10.0, 10.0), (-8.0, 8.0))    # Very zoomed out
        ]
        self.default_zoom_index = 2
    
    def change_zoom(self, device_name, direction):
        """Change zoom level for a device (-1 for zoom in, 1 for zoom out)"""
        current_index = self.zoom_levels.get(device_name, self.default_zoom_index)
        new_index = max(0, min(len(self.zoom_presets) - 1, current_index + direction))
        self.zoom_levels[device_name] = new_index
    
    def get_zoom_ranges(self, device_name):
        """Get current zoom ranges for a device"""
        index = self.zoom_levels.get(device_name, self.default_zoom_index)
        return self.zoom_presets[index]
    
    def draw(self, device_data_dict):
        """Draw waveforms for all active devices"""
        # Panel background
        pygame.draw.rect(self.screen, Colors.PANEL, (self.x, self.y, self.width, self.height))
        
        # Title
        title = self.font_manager.render_text("IMU Waveforms", 'large', Colors.TEXT)
        self.screen.blit(title, (self.x + 20, self.y + 20))
        
        # Get active devices
        active_devices = []
        for device_name, data in device_data_dict.items():
            if data and len(data.get('accel_history', [])) > 0:
                active_devices.append((device_name, data))
        
        if not active_devices:
            self._draw_no_data_message()
            return
        
        # Create device waveform components if needed
        for device_name, _ in active_devices:
            if device_name not in self.device_waveforms:
                self.device_waveforms[device_name] = DeviceWaveform(self.screen, device_name)
        
        # Draw waveforms
        current_y = self.y + 70
        
        for device_name, device_data in active_devices:
            device_waveform = self.device_waveforms[device_name]
            
            # Get zoom ranges
            zoom_ranges = self.get_zoom_ranges(device_name)
            
            # Calculate height based on collapsed state and sensor availability
            if device_waveform.is_collapsed:
                height_used = device_waveform.draw_collapsed(
                    self.x + 10, current_y, self.width - 20, device_data
                )
            else:
                has_gyro = self._has_gyro_data(device_data)
                device_height = 280 if has_gyro else 200
                
                # Add zoom level indicator
                zoom_index = self.zoom_levels.get(device_name, self.default_zoom_index)
                zoom_text = self.font_manager.render_text(
                    f"Zoom: {zoom_index + 1}/{len(self.zoom_presets)}", 
                    'tiny', Colors.TEXT_SECONDARY
                )
                self.screen.blit(zoom_text, (self.x + self.width - 120, current_y + 10))
                
                height_used = device_waveform.draw(
                    self.x + 10, current_y, self.width - 20, device_height, 
                    device_data, zoom_ranges
                )
            
            current_y += height_used + 10
    
    def _has_gyro_data(self, device_data):
        """Check if device has gyroscope data"""
        gyro_history = device_data.get('gyro_history', [])
        return len(gyro_history) > 0 and any(np.linalg.norm(g) > 0.001 for g in gyro_history)
    
    def _draw_no_data_message(self):
        """Draw message when no data is available"""
        msg1 = self.font_manager.render_text("No active devices", 'medium', Colors.TEXT_SECONDARY)
        self.screen.blit(msg1, (self.x + 20, self.y + 100))
        
        msg2 = self.font_manager.render_text("Enable devices in your iOS SensorTracker app", 
                                           'small', Colors.TEXT_SECONDARY)
        self.screen.blit(msg2, (self.x + 20, self.y + 130))
        
        # Connection help
        help_lines = [
            "1. Open iOS SensorTracker app",
            "2. Enter your computer's IP address", 
            "3. Connect and enable devices",
            "4. Real-time data will appear here"
        ]
        
        for i, line in enumerate(help_lines):
            help_text = self.font_manager.render_text(line, 'small', Colors.TEXT_SECONDARY)
            self.screen.blit(help_text, (self.x + 40, self.y + 170 + i * 25))
    
    def handle_click(self, pos):
        """Handle mouse click on waveform controls"""
        for device_name, device_waveform in self.device_waveforms.items():
            button_clicked = device_waveform.handle_click(pos)
            
            if button_clicked:
                if button_clicked in ['collapse', 'expand']:
                    device_waveform.toggle_collapse()
                    return (button_clicked, device_name)
                elif button_clicked == 'zoom_in':
                    self.change_zoom(device_name, -1)
                    return ('zoom_in', device_name)
                elif button_clicked == 'zoom_out':
                    self.change_zoom(device_name, 1)
                    return ('zoom_out', device_name)
        
        return None