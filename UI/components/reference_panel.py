"""Reference coordinate system panel showing global frame with identity quaternion"""

import pygame
import numpy as np
from scipy.spatial.transform import Rotation as R
from ..utils.colors import Colors
from ..utils.fonts import FontManager
from ..utils.renderer_3d import Renderer3D

class ReferencePanel:
    """Shows the global coordinate system with standard identity orientation"""
    
    def __init__(self, screen, x, y, width, height):
        self.screen = screen
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        
        self.font_manager = FontManager()
        self.renderer = Renderer3D(screen)
        
        # Define center of the reference visualization
        self.ref_center = (self.x + self.width - 120, self.y + 70)
    
    def draw(self):
        """Draw the reference panel with standard reference frame"""
        # Title
        title = self.font_manager.render_text("Global Coordinate System", 'large', Colors.TEXT)
        self.screen.blit(title, (self.x + 20, self.y))
        
        # Draw coordinate explanations
        self._draw_coordinate_explanations()
        
        # Draw standard reference coordinate system (identity quaternion)
        self._draw_standard_reference()
    
    def _draw_coordinate_explanations(self):
        """Draw axis explanations on the left side with directional clarifications"""
        explanations = [
            ("X-axis (Red):", "LEFT", Colors.AXIS_X),
            ("Y-axis (Green):", "UP", Colors.AXIS_Y),
            ("Z-axis (Blue):", "FORWARD", Colors.AXIS_Z)
        ]
        
        start_y = self.y + 50
        
        for i, (axis_name, direction, color) in enumerate(explanations):
            y_pos = start_y + i * 25
            
            # Axis name
            axis_text = self.font_manager.render_text(axis_name, 'small', color)
            self.screen.blit(axis_text, (self.x + 20, y_pos))
            
            # Direction explanation
            dir_text = self.font_manager.render_text(direction, 'small', Colors.TEXT_TERTIARY)
            self.screen.blit(dir_text, (self.x + 120, y_pos))
        
        # Add clarification text for Z direction below the axis descriptions
        clarification_texts = [
            "Forward: into screen (filled circle)",
            "Backward: out of screen (hollow circle)"
        ]
        
        # Add some space after the axis descriptions
        clarification_y = start_y + len(explanations) * 25 + 5
        
        for i, text in enumerate(clarification_texts):
            # Create text surface
            text_surface = self.font_manager.render_text(text, 'small', 
                                                       Colors.TEXT_SECONDARY)
            # Position and draw
            text_y = clarification_y + i * 20
            self.screen.blit(text_surface, (self.x + 20, text_y))
    
    def _draw_standard_reference(self):
        """Draw the 3D reference coordinate system with identity quaternion"""
        # Background circle
        pygame.draw.circle(self.screen, (30, 30, 40), self.ref_center, 55)
        pygame.draw.circle(self.screen, (100, 100, 120), self.ref_center, 55, 2)
        
        # Create identity quaternion (no rotation)
        identity_quat = np.array([0, 0, 0, 1])
        
        # Use the standard draw_3d_axes method
        axis_colors = [Colors.AXIS_X, Colors.AXIS_Y, Colors.AXIS_Z]
        self.renderer.draw_3d_axes(
            self.ref_center, 
            identity_quat,
            axis_length=45, 
            axis_colors=axis_colors, 
            font_manager=self.font_manager,
            device_type='global'
        )