"""Calibration button component"""

import pygame
from ..utils.colors import Colors
from ..utils.fonts import FontManager

class CalibrationButton:
    """Calibration button with hover effects"""
    
    def __init__(self, screen, x, y, width=180, height=45):
        self.screen = screen
        self.rect = pygame.Rect(x, y, width, height)
        self.font_manager = FontManager()
        self.is_hovered = False
    
    def update(self, mouse_pos):
        """Update button state based on mouse position"""
        self.is_hovered = self.rect.collidepoint(mouse_pos)
    
    def draw(self):
        """Draw the calibration button"""
        # Button color based on hover state
        button_color = Colors.BUTTON_HOVER if self.is_hovered else Colors.BUTTON
        
        # Draw shadow
        shadow_offset = 2
        shadow_rect = self.rect.copy()
        shadow_rect.x += shadow_offset
        shadow_rect.y += shadow_offset
        pygame.draw.rect(self.screen, (0, 0, 0), shadow_rect)
        
        # Draw button
        pygame.draw.rect(self.screen, button_color, self.rect)
        pygame.draw.rect(self.screen, Colors.TEXT, self.rect, 2)
        
        # Button text
        text = self.font_manager.render_text("CALIBRATE", 'medium', Colors.TEXT)
        text_rect = text.get_rect(center=self.rect.center)
        self.screen.blit(text, text_rect)
        
        # Instruction text above button
        instruction = self.font_manager.render_text("Set current as reference", 'small', Colors.TEXT_SECONDARY)
        instruction_rect = instruction.get_rect(centerx=self.rect.centerx, bottom=self.rect.top - 5)
        self.screen.blit(instruction, instruction_rect)
    
    def is_clicked(self, event):
        """Check if button was clicked"""
        if event.type == pygame.MOUSEBUTTONDOWN:
            return self.rect.collidepoint(event.pos)
        return False