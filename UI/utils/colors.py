"""Color definitions for the IMU visualizer with AR Glasses support"""

class Colors:
    """Enhanced color scheme for IMU visualizer with AR Glasses"""
    
    # Background colors
    BG = (15, 15, 25)
    PANEL = (25, 25, 35)
    WAVEFORM_BG = (20, 20, 30)
    
    # UI elements
    BUTTON = (70, 130, 180)
    BUTTON_HOVER = (90, 150, 200)
    
    # Text colors
    TEXT = (255, 255, 255)
    TEXT_SECONDARY = (180, 180, 180)
    TEXT_TERTIARY = (120, 120, 120)
    
    # Device colors
    PHONE = (255, 100, 100)        # Red - Phone
    HEADPHONE = (100, 255, 100)    # Green - AirPods/Headphones
    WATCH = (100, 100, 255)        # Blue - Apple Watch
    GLASSES = (255, 150, 50)       # Orange - AR Glasses
    
    # Axis colors
    AXIS_X = (255, 60, 60)      # Red - X axis
    AXIS_Y = (60, 255, 60)      # Green - Y axis  
    AXIS_Z = (60, 60, 255)      # Blue - Z axis
    
    # Status colors
    CALIBRATED = (50, 255, 50)      # Green - Calibrated
    UNCALIBRATED = (255, 200, 50)   # Yellow - Uncalibrated
    REFERENCE = (255, 150, 255)     # Magenta - Reference device
    
    # Grid and highlights
    GRID = (40, 40, 50)
    CONE_HIGHLIGHT = (255, 255, 100)
    
    # Device status colors
    INACTIVE = (100, 100, 100)
    WAITING = (150, 150, 150)
    
    @classmethod
    def get_device_color(cls, device_name):
        """Get color for specific device type"""
        device_colors = {
            'phone': cls.PHONE,
            'headphone': cls.HEADPHONE,
            'watch': cls.WATCH,
            'glasses': cls.GLASSES
        }
        return device_colors.get(device_name, cls.TEXT_SECONDARY)
    
    @classmethod
    def get_axis_color(cls, axis_index):
        """Get color for axis by index (0=X, 1=Y, 2=Z)"""
        axis_colors = [cls.AXIS_X, cls.AXIS_Y, cls.AXIS_Z]
        return axis_colors[axis_index] if 0 <= axis_index < 3 else cls.TEXT_SECONDARY