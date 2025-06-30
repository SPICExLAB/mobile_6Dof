"""Device waveform component - handles a complete device's waveforms"""

import pygame
import numpy as np
from ..utils.colors import Colors
from ..utils.fonts import FontManager
from .sensor_waveform import SensorWaveform

class DeviceWaveform:
    """Manages waveforms for a single device"""
    
    def __init__(self, screen, device_name):
        self.screen = screen
        self.device_name = device_name
        self.font_manager = FontManager()
        
        # Create sensor waveform components
        self.accel_waveform = SensorWaveform(screen, 'accel')
        self.gyro_waveform = SensorWaveform(screen, 'gyro')
        
        # Control buttons
        self.buttons = {}
        self.is_collapsed = False
        
    def draw(self, x, y, width, height, device_data, zoom_ranges):
        """Draw the device waveform panel"""
        # Clear buttons for this frame
        self.buttons = {}
        
        # Draw header
        header_height = 35
        self._draw_header(x, y, width, header_height, device_data)
        
        if self.is_collapsed:
            return header_height + 5
        
        # Get sensor data
        accel_history = list(device_data.get('accel_history', []))
        gyro_history = list(device_data.get('gyro_history', []))
        
        if len(accel_history) < 2:
            return height
        
        # Check if we have gyroscope data
        has_gyro = self._has_gyro_data(device_data)
        
        # Calculate waveform area
        content_y = y + header_height + 25  # Space for sensor title
        available_height = height - header_height - 30
        
        # Get current values for display
        accel_values = accel_history[-1] if accel_history else [0, 0, 0]
        gyro_values = gyro_history[-1] if gyro_history and has_gyro else [0, 0, 0]
        
        # Get zoom ranges
        accel_range, gyro_range = zoom_ranges
        
        if has_gyro:
            # Split space for two sensors
            sensor_height = available_height // 2 - 20
            
            # Draw accelerometer
            accel_title = "Accelerometer (m/s²)"
            self.accel_waveform.draw(x, content_y, width, sensor_height, 
                                    accel_history, accel_range, accel_title, accel_values)
            
            # Draw gyroscope
            gyro_y = content_y + sensor_height + 40
            gyro_title = "Gyroscope (rad/s)"
            if self.device_name == 'watch':
                gyro_title += " - Apple Watch"
            
            self.gyro_waveform.draw(x, gyro_y, width, sensor_height,
                                   gyro_history, gyro_range, gyro_title, gyro_values)
        else:
            # Only accelerometer
            accel_title = "Accelerometer (m/s²) - No gyroscope data"
            self.accel_waveform.draw(x, content_y, width, available_height - 20,
                                    accel_history, accel_range, accel_title, accel_values)
        
        return height
    
    def draw_collapsed(self, x, y, width, device_data):
        """Draw collapsed device header"""
        height = 35
        
        # Background
        pygame.draw.rect(self.screen, Colors.WAVEFORM_BG, (x, y, width, height))
        pygame.draw.rect(self.screen, Colors.get_device_color(self.device_name), (x, y, width, height), 2)
        
        # Expand button
        button_rect = pygame.Rect(x + 5, y + 5, 25, 25)
        pygame.draw.rect(self.screen, Colors.BUTTON, button_rect)
        expand_text = self.font_manager.render_text("+", 'medium', Colors.TEXT)
        expand_rect = expand_text.get_rect(center=button_rect.center)
        self.screen.blit(expand_text, expand_rect)
        self.buttons['expand'] = button_rect
        
        # Device info
        device_color = Colors.get_device_color(self.device_name)
        status_text = "CAL" if device_data.get('is_calibrated', False) else "UNCAL"
        freq_text = f"@ {device_data.get('frequency', 0):.1f}Hz"
        
        header = self.font_manager.render_text(f"{status_text} {self.device_name.upper()} {freq_text}", 
                                             'small', device_color)
        self.screen.blit(header, (x + 40, y + 10))
        
        # Current magnitude
        accel_history = device_data.get('accel_history', [])
        if len(accel_history) > 0:
            magnitude = np.linalg.norm(accel_history[-1])
            mag_text = self.font_manager.render_text(f"Magnitude: {magnitude:.3f}", 'small', Colors.TEXT_SECONDARY)
            self.screen.blit(mag_text, (x + width - 150, y + 10))
        
        return height + 5
    
    def _draw_header(self, x, y, width, height, device_data):
        """Draw device header with controls"""
        # Background
        pygame.draw.rect(self.screen, Colors.WAVEFORM_BG, (x, y, width, height))
        pygame.draw.rect(self.screen, Colors.get_device_color(self.device_name), (x, y, width, height), 2)
        
        # Collapse button
        collapse_rect = pygame.Rect(x + 5, y + 5, 25, 25)
        pygame.draw.rect(self.screen, Colors.BUTTON, collapse_rect)
        collapse_text = self.font_manager.render_text("-", 'medium', Colors.TEXT)
        collapse_text_rect = collapse_text.get_rect(center=collapse_rect.center)
        self.screen.blit(collapse_text, collapse_text_rect)
        self.buttons['collapse'] = collapse_rect
        
        # Device info
        device_color = Colors.get_device_color(self.device_name)
        status_text = "CAL" if device_data.get('is_calibrated', False) else "UNCAL"
        freq_text = f"@ {device_data.get('frequency', 0):.1f}Hz"
        
        header = self.font_manager.render_text(f"{status_text} {self.device_name.upper()} {freq_text}", 
                                             'medium', device_color)
        self.screen.blit(header, (x + 40, y + 8))
        
        # Zoom controls on the right
        zoom_out_rect = pygame.Rect(x + width - 60, y + 5, 25, 25)
        zoom_in_rect = pygame.Rect(x + width - 30, y + 5, 25, 25)
        
        # Zoom out button (-)
        pygame.draw.rect(self.screen, Colors.BUTTON, zoom_out_rect)
        zoom_out_text = self.font_manager.render_text("-", 'small', Colors.TEXT)
        zoom_out_text_rect = zoom_out_text.get_rect(center=zoom_out_rect.center)
        self.screen.blit(zoom_out_text, zoom_out_text_rect)
        self.buttons['zoom_out'] = zoom_out_rect
        
        # Zoom in button (+)
        pygame.draw.rect(self.screen, Colors.BUTTON, zoom_in_rect)
        zoom_in_text = self.font_manager.render_text("+", 'small', Colors.TEXT)
        zoom_in_text_rect = zoom_in_text.get_rect(center=zoom_in_rect.center)
        self.screen.blit(zoom_in_text, zoom_in_text_rect)
        self.buttons['zoom_in'] = zoom_in_rect
    
    def _has_gyro_data(self, device_data):
        """Check if device has gyroscope data"""
        gyro_history = device_data.get('gyro_history', [])
        return len(gyro_history) > 0 and any(np.linalg.norm(g) > 0.001 for g in gyro_history)
    
    def handle_click(self, pos):
        """Handle click events"""
        for button_type, rect in self.buttons.items():
            if rect.collidepoint(pos):
                return button_type
        return None
    
    def toggle_collapse(self):
        """Toggle collapsed state"""
        self.is_collapsed = not self.is_collapsed