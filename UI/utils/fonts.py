"""Font management for the IMU visualizer"""

import pygame

class FontManager:
    """Manages fonts for the application"""
    
    def __init__(self):
        self.fonts = {}
        self._initialize_fonts()
    
    def _initialize_fonts(self):
        """Initialize all fonts used in the application"""
        self.fonts = {
            'large': pygame.font.Font(None, 32),
            'medium': pygame.font.Font(None, 24),
            'small': pygame.font.Font(None, 18),
            'tiny': pygame.font.Font(None, 14)
        }
    
    def get(self, size='medium'):
        """Get font by size name"""
        return self.fonts.get(size, self.fonts['medium'])
    
    def render_text(self, text, size='medium', color=(255, 255, 255), antialias=True):
        """Render text with specified font size and color"""
        font = self.get(size)
        return font.render(text, antialias, color)
    
    def get_text_size(self, text, size='medium'):
        """Get the width and height of rendered text"""
        font = self.get(size)
        return font.size(text)