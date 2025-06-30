"""3D rendering utilities with consistent coordinate systems and calibration visualization"""

import numpy as np
import pygame
import math
from scipy.spatial.transform import Rotation as R
import logging

logger = logging.getLogger(__name__)

class Renderer3D:
    """Handles 3D rendering operations with calibration-aware coordinate handling"""
    
    # Device 3D scales (for different shapes)
    DEVICE_SCALES = {
        'phone': (0.4, 1.0, 0.08),      # iPhone: thin rectangle
        'headphone': (0.25, 0.25, 0.25),   # AirPods: smaller cube
        'watch': (0.8, 0.8, 0.3),       # Watch: square with depth
        'glasses': (1.5, 0.5, 0.1)      # AR Glasses: wide horizontal
    }
    
    def __init__(self, screen):
        self.screen = screen
    
    def orthographic_project(self, point_3d: np.ndarray, center: tuple, scale: float = 1.0) -> tuple:
        """
        Orthographic projection that consistently converts from global frame to screen space.
        Global frame: X:left (+), Y:up (+), Z:forward (+)
        Screen space: X:right (+), Y:down (+)
        """
        x, y, z = point_3d
        
        # Convert from global frame to screen space - reverse the signs to flip directions
        screen_x = int(center[0] - x * scale)  # Flip X (left → right)
        screen_y = int(center[1] - y * scale)  # Flip Y (up → down)
        
        return screen_x, screen_y
    
    def create_device_vertices(self, device_type: str, size: float):
        """
        Create device-specific 3D vertices consistently in global coordinate system.
        Global frame: X:left (+), Y:up (+), Z:forward (+)
        """
        scale_x, scale_y, scale_z = self.DEVICE_SCALES.get(device_type, (0.5, 0.5, 0.5))
        
        # Standard cuboid vertices in global frame
        # Define faces consistently so rotation appears correct
        vertices = np.array([
            # Bottom face (Y-negative)
            [scale_x, -scale_y, -scale_z],     # 0: left-bottom-back
            [-scale_x, -scale_y, -scale_z],    # 1: right-bottom-back
            [-scale_x, -scale_y, scale_z],     # 2: right-bottom-front
            [scale_x, -scale_y, scale_z],      # 3: left-bottom-front
            
            # Top face (Y-positive)
            [scale_x, scale_y, -scale_z],      # 4: left-top-back
            [-scale_x, scale_y, -scale_z],     # 5: right-top-back
            [-scale_x, scale_y, scale_z],      # 6: right-top-front
            [scale_x, scale_y, scale_z]        # 7: left-top-front
        ]) * size
        
        # Apply device-specific adjustments to better represent the shape
        if device_type == 'glasses':
            # Make glasses wider in X, thinner in Z
            adjustment = np.array([1.5, 1.0, 0.5])
            vertices = vertices * adjustment
        elif device_type == 'phone':
            # Make phone taller in Y, thinner in Z
            adjustment = np.array([1.0, 1.2, 0.4])
            vertices = vertices * adjustment
        elif device_type == 'watch':
            # Make watch more square, thicker in Z
            adjustment = np.array([1.0, 1.0, 0.6])
            vertices = vertices * adjustment
        elif device_type == 'headphone':
            # Make headphones smaller overall
            vertices = vertices * 0.8
        
        return vertices
    
    def calculate_face_normal(self, vertices: np.ndarray, face: list) -> np.ndarray:
        """Calculate face normal for lighting"""
        if len(face) < 3:
            return np.array([0, 0, 1])
        
        # Get three points of the face
        p1, p2, p3 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        
        # Calculate normal using cross product
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        
        # Normalize
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
        
        return normal
    
    def draw_arrow(self, center, direction, color, length=30):
        """
        Draw arrow with optional Z-axis indicator (circle only, no line).
        """
        # Normalize direction
        if np.linalg.norm(direction) == 0:
            return
        
        direction = direction / np.linalg.norm(direction)
        
        # Calculate end point
        end_x = center[0] + direction[0] * length
        end_y = center[1] + direction[1] * length
        end_point = (int(end_x), int(end_y))

        # Draw main line
        line_thickness = 3
        pygame.draw.line(self.screen, color, center, end_point, line_thickness)
        
        # Draw arrowhead
        perp = np.array([-direction[1], direction[0]])  # 2D perpendicular
        arrow_size = 10
        arrow_back = np.array([
            end_x - direction[0] * arrow_size,
            end_y - direction[1] * arrow_size
        ])
        arrow_left = arrow_back + perp * arrow_size * 0.5
        arrow_right = arrow_back - perp * arrow_size * 0.5

        pygame.draw.line(self.screen, color, end_point, (int(arrow_left[0]), int(arrow_left[1])), line_thickness)
        pygame.draw.line(self.screen, color, end_point, (int(arrow_right[0]), int(arrow_right[1])), line_thickness)

    
    def draw_3d_axes(self, center, quaternion, axis_length=40, 
        axis_colors=None, font_manager=None, device_type='phone', 
        is_calibrated=False, is_reference=False):
        """
        Draw 3D coordinate axes with simplified Z-axis representation.
        Global frame: X:left (+), Y:up (+), Z:forward (+)
        """
        if axis_colors is None:
            axis_colors = [(255, 60, 60), (60, 255, 60), (60, 60, 255)]  # R, G, B
        
        axes_directions = np.array([
            [1, 0, 0],    # X-axis
            [0, 1, 0],    # Y-axis
            [0, 0, 1]     # Z-axis
        ])
        
        if quaternion is not None and np.linalg.norm(quaternion) > 0:
            try:
                rotation = R.from_quat(quaternion)
                rotated_axes = rotation.apply(axes_directions)
            except Exception as e:
                logger.error(f"Error rotating axes: {e}")
                rotated_axes = axes_directions
        else:
            rotated_axes = axes_directions
        
        axis_labels = ['X', 'Y', 'Z']
        
        for i, (direction, color, label) in enumerate(zip(rotated_axes, axis_colors, axis_labels)):
            if i == 2:
                continue  # ✅ Skip Z-axis (no arrow, no label)

            # Simple fake 3D to 2D projection
            depth_scale = 0.5
            projected_x = -direction[0] * (1 + direction[2] * depth_scale)
            projected_y = -direction[1] * (1 + direction[2] * depth_scale)
            screen_direction = np.array([projected_x, projected_y])

            self.draw_arrow(center, screen_direction, color, length=axis_length)

            # Draw label
            if font_manager:
                label_pos = (
                    center[0] + screen_direction[0] * (axis_length + 15),
                    center[1] + screen_direction[1] * (axis_length + 15)
                )
                label_text = font_manager.render_text(label, 'small', color)
                label_bg = pygame.Surface((label_text.get_width() + 4, label_text.get_height() + 4))
                label_bg.fill((0, 0, 0))
                label_bg.set_alpha(128)
                self.screen.blit(label_bg, (label_pos[0] - 2, label_pos[1] - 2))
                self.screen.blit(label_text, label_pos)


   