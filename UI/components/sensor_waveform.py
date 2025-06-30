"""Individual sensor waveform component"""

import pygame
import numpy as np
from ..utils.colors import Colors
from ..utils.fonts import FontManager

class SensorWaveform:
    """Displays a single sensor waveform (accelerometer or gyroscope)"""
    
    def __init__(self, screen, sensor_type='accel'):
        self.screen = screen
        self.sensor_type = sensor_type
        self.font_manager = FontManager()
        self.line_thickness = 2
        
    def draw(self, x, y, width, height, sensor_history, y_range, title, current_values=None):
        """Draw the sensor waveform"""
        if len(sensor_history) < 2:
            return
            
        # Background
        pygame.draw.rect(self.screen, Colors.WAVEFORM_BG, (x, y, width, height))
        pygame.draw.rect(self.screen, (80, 80, 100), (x, y, width, height), 1)
        
        # Title and axis values on the same line
        title_y = y - 20
        title_surface = self.font_manager.render_text(title, 'small', Colors.TEXT_SECONDARY)
        self.screen.blit(title_surface, (x, title_y))
        
        # Draw axis values on the same line as title
        if current_values is not None:
            self._draw_axis_values(x + 300, title_y, current_values)
        
        # Draw grid
        self._draw_grid(x, y, width, height)
        
        # Draw zero line
        zero_y = y + height // 2
        pygame.draw.line(self.screen, (140, 140, 160), (x, zero_y), (x + width, zero_y), 2)
        
        # Draw waveforms
        self._draw_waveforms(x, y, width, height, sensor_history, y_range)
        
        # Draw range indicators
        self._draw_range_indicators(x, y, height, y_range)
        
        # Draw magnitude
        if len(sensor_history) > 0:
            magnitude = np.linalg.norm(sensor_history[-1])
            mag_text = self.font_manager.render_text(f"Magnitude: {magnitude:.3f}", 'tiny', Colors.TEXT_SECONDARY)
            self.screen.blit(mag_text, (x + 10, y + height + 5))
    
    def _draw_axis_values(self, x, y, values):
        """Draw X, Y, Z axis values"""
        axis_colors = [Colors.AXIS_X, Colors.AXIS_Y, Colors.AXIS_Z]
        axis_labels = ['X', 'Y', 'Z']
        
        for i, (label, color, value) in enumerate(zip(axis_labels, axis_colors, values)):
            # Color box
            box_x = x + i * 120
            pygame.draw.rect(self.screen, color, (box_x, y + 2, 12, 12))
            pygame.draw.rect(self.screen, Colors.TEXT, (box_x, y + 2, 12, 12), 1)
            
            # Value text
            text = self.font_manager.render_text(f"{label}: {value:+.3f}", 'small', Colors.TEXT)
            self.screen.blit(text, (box_x + 16, y))
    
    def _draw_waveforms(self, x, y, width, height, sensor_history, y_range):
        """Draw the actual waveform lines"""
        sensor_array = np.array(sensor_history)
        n_samples = len(sensor_array)
        
        axis_colors = [Colors.AXIS_X, Colors.AXIS_Y, Colors.AXIS_Z]
        
        for axis_idx in range(3):
            color = axis_colors[axis_idx]
            values = sensor_array[:, axis_idx]
            
            # Create points
            points = []
            for i, value in enumerate(values):
                plot_x = x + int((i / max(1, n_samples - 1)) * width)
                
                # Clamp and normalize
                clamped_value = max(y_range[0], min(y_range[1], value))
                normalized = (clamped_value - y_range[0]) / (y_range[1] - y_range[0])
                plot_y = y + height - int(normalized * height)
                
                points.append((plot_x, plot_y))
            
            # Draw waveform
            if len(points) > 1:
                pygame.draw.lines(self.screen, color, False, points, self.line_thickness)
    
    def _draw_grid(self, x, y, width, height):
        """Draw grid lines"""
        # Horizontal lines
        for i in range(1, 5):
            grid_y = y + (i * height // 5)
            pygame.draw.line(self.screen, Colors.GRID, (x, grid_y), (x + width, grid_y), 1)
        
        # Vertical lines
        for i in range(1, 9):
            grid_x = x + (i * width // 9)
            pygame.draw.line(self.screen, Colors.GRID, (grid_x, y), (grid_x, y + height), 1)
    
    def _draw_range_indicators(self, x, y, height, y_range):
        """Draw range indicators"""
        top_text = self.font_manager.render_text(f"+{y_range[1]:.1f}", 'tiny', Colors.TEXT_SECONDARY)
        bottom_text = self.font_manager.render_text(f"{y_range[0]:.1f}", 'tiny', Colors.TEXT_SECONDARY)
        
        self.screen.blit(top_text, (x + 5, y + 2))
        self.screen.blit(bottom_text, (x + 5, y + height - 15))