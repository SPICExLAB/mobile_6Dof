"""Dynamic device grid layout manager with AR Glasses support"""

class DeviceGridLayout:
    """Manages dynamic grid layout for device panels including AR glasses"""
    
    def __init__(self, box_size=180, margin=20, max_per_row=2):
        self.box_size = box_size
        self.margin = margin
        self.max_per_row = max_per_row
        
        # Device order - AR glasses added
        self.device_order = ['phone', 'headphone', 'watch', 'glasses']
        
        # Starting position for device grid
        self.start_x = 30
        self.start_y = 200  # Below the reference section
        
        # Panel width constraint
        self.panel_width = 480  # Approximate left panel width
    
    def calculate_positions(self, active_devices):
        """Calculate device positions based on connected devices"""
        positions = {}
        
        # Separate active and inactive devices
        inactive_devices = [d for d in self.device_order if d not in active_devices]
        
        if len(active_devices) == 0:
            # All inactive - show as small boxes
            self._layout_all_inactive(positions)
        
        elif len(active_devices) == 1:
            # One active - large box, others small below
            self._layout_one_active(positions, active_devices[0], inactive_devices)
        
        elif len(active_devices) == 2:
            # Two active - side by side, others small below
            self._layout_two_active(positions, active_devices, inactive_devices)
        
        elif len(active_devices) == 3:
            # Three active - 2 on top, 1 below, 1 small inactive
            self._layout_three_active(positions, active_devices, inactive_devices)
        
        else:
            # All active or more - 2x2 grid
            self._layout_all_active(positions, active_devices)
        
        return positions
    
    def _layout_all_inactive(self, positions):
        """Layout when all devices are inactive"""
        small_size = 80
        small_margin = 15
        
        # AR glasses on top (horizontal rectangle)
        glasses_width = 120
        glasses_height = 40
        glasses_x = (self.panel_width - glasses_width) // 2
        glasses_y = self.start_y
        
        positions['glasses'] = {
            'center': (glasses_x + glasses_width // 2, glasses_y + glasses_height // 2),
            'bounds': (glasses_x, glasses_y, glasses_width, glasses_height),
            'size': small_size,
            'active': False
        }
        
        # Other devices in a row below
        other_devices = ['phone', 'headphone', 'watch']
        total_width = len(other_devices) * small_size + (len(other_devices) - 1) * small_margin
        start_x = (self.panel_width - total_width) // 2
        devices_y = glasses_y + glasses_height + 20
        
        for i, device in enumerate(other_devices):
            x = start_x + i * (small_size + small_margin)
            
            positions[device] = {
                'center': (x + small_size // 2, devices_y + small_size // 2),
                'bounds': (x, devices_y, small_size, small_size),
                'size': small_size,
                'active': False
            }
    
    def _layout_one_active(self, positions, active_device, inactive_devices):
        """Layout when one device is active"""
        # Active device - large centered box
        large_size = 200
        x = (self.panel_width - large_size) // 2
        y = self.start_y
        
        positions[active_device] = {
            'center': (x + large_size // 2, y + large_size // 2),
            'bounds': (x, y, large_size, large_size),
            'size': large_size,
            'active': True
        }
        
        # Inactive devices below
        self._layout_inactive_devices_below(positions, inactive_devices, y + large_size + 20)
    
    def _layout_two_active(self, positions, active_devices, inactive_devices):
        """Layout when two devices are active"""
        box_size = 180
        total_active_width = 2 * box_size + self.margin
        start_x = (self.panel_width - total_active_width) // 2
        
        for i, device in enumerate(active_devices):
            x = start_x + i * (box_size + self.margin)
            y = self.start_y
            
            positions[device] = {
                'center': (x + box_size // 2, y + box_size // 2),
                'bounds': (x, y, box_size, box_size),
                'size': box_size,
                'active': True
            }
        
        # Inactive devices below
        self._layout_inactive_devices_below(positions, inactive_devices, self.start_y + box_size + 20)
    
    def _layout_three_active(self, positions, active_devices, inactive_devices):
        """Layout when three devices are active"""
        box_size = 160
        
        # Check if glasses is active for special positioning
        if 'glasses' in active_devices:
            # Put glasses on top (horizontal), other two below
            glasses_width = box_size * 1.5
            glasses_height = box_size * 0.6
            glasses_x = (self.panel_width - glasses_width) // 2
            glasses_y = self.start_y
            
            positions['glasses'] = {
                'center': (glasses_x + glasses_width // 2, glasses_y + glasses_height // 2),
                'bounds': (glasses_x, glasses_y, glasses_width, glasses_height),
                'size': box_size,
                'active': True
            }
            
            # Other two active devices below
            other_active = [d for d in active_devices if d != 'glasses']
            if len(other_active) >= 2:
                total_width = 2 * box_size + self.margin
                start_x = (self.panel_width - total_width) // 2
                devices_y = glasses_y + glasses_height + 20
                
                for i, device in enumerate(other_active[:2]):
                    x = start_x + i * (box_size + self.margin)
                    
                    positions[device] = {
                        'center': (x + box_size // 2, devices_y + box_size // 2),
                        'bounds': (x, devices_y, box_size, box_size),
                        'size': box_size,
                        'active': True
                    }
                
                # Remaining inactive devices small below
                if inactive_devices:
                    self._layout_inactive_devices_below(positions, inactive_devices, 
                                                      devices_y + box_size + 20)
        else:
            # No glasses active - arrange 3 in triangle or line
            if len(active_devices) == 3:
                # Triangle layout: 2 on top, 1 below
                total_width = 2 * box_size + self.margin
                start_x = (self.panel_width - total_width) // 2
                
                # Top two
                for i in range(2):
                    device = active_devices[i]
                    x = start_x + i * (box_size + self.margin)
                    y = self.start_y
                    
                    positions[device] = {
                        'center': (x + box_size // 2, y + box_size // 2),
                        'bounds': (x, y, box_size, box_size),
                        'size': box_size,
                        'active': True
                    }
                
                # Bottom one centered
                device = active_devices[2]
                x = (self.panel_width - box_size) // 2
                y = self.start_y + box_size + 20
                
                positions[device] = {
                    'center': (x + box_size // 2, y + box_size // 2),
                    'bounds': (x, y, box_size, box_size),
                    'size': box_size,
                    'active': True
                }
                
                # Inactive below
                if inactive_devices:
                    self._layout_inactive_devices_below(positions, inactive_devices, 
                                                      y + box_size + 20)
    
    def _layout_all_active(self, positions, active_devices):
        """Layout when all devices are active"""
        box_size = 150
        
        # Special case for glasses - place horizontally at top
        if 'glasses' in active_devices:
            glasses_width = box_size * 1.8
            glasses_height = box_size * 0.5
            glasses_x = (self.panel_width - glasses_width) // 2
            glasses_y = self.start_y
            
            positions['glasses'] = {
                'center': (glasses_x + glasses_width // 2, glasses_y + glasses_height // 2),
                'bounds': (glasses_x, glasses_y, glasses_width, glasses_height),
                'size': box_size,
                'active': True
            }
            
            # Other devices in 2x2 or line below
            other_devices = [d for d in active_devices if d != 'glasses']
            self._layout_remaining_devices(positions, other_devices, box_size, 
                                         glasses_y + glasses_height + 20)
        else:
            # Standard 2x2 grid
            self._layout_grid(positions, active_devices, box_size)
    
    def _layout_remaining_devices(self, positions, devices, box_size, start_y):
        """Layout remaining devices below glasses"""
        if len(devices) <= 2:
            # Side by side
            total_width = len(devices) * box_size + (len(devices) - 1) * self.margin
            start_x = (self.panel_width - total_width) // 2
            
            for i, device in enumerate(devices):
                x = start_x + i * (box_size + self.margin)
                
                positions[device] = {
                    'center': (x + box_size // 2, start_y + box_size // 2),
                    'bounds': (x, start_y, box_size, box_size),
                    'size': box_size,
                    'active': True
                }
        else:
            # Grid layout
            self._layout_grid(positions, devices, box_size, start_y)
    
    def _layout_grid(self, positions, devices, box_size, start_y=None):
        """Layout devices in a grid"""
        if start_y is None:
            start_y = self.start_y
            
        col_width = box_size + self.margin
        row_height = box_size + self.margin
        
        # Center the grid
        grid_width = 2 * box_size + self.margin
        start_x = (self.panel_width - grid_width) // 2
        
        for i, device in enumerate(devices[:4]):  # Max 4 devices
            row = i // 2
            col = i % 2
            
            x = start_x + col * col_width
            y = start_y + row * row_height
            
            positions[device] = {
                'center': (x + box_size // 2, y + box_size // 2),
                'bounds': (x, y, box_size, box_size),
                'size': box_size,
                'active': True
            }
    
    def _layout_inactive_devices_below(self, positions, inactive_devices, start_y):
        """Layout inactive devices in a row below active ones"""
        if not inactive_devices:
            return
            
        small_size = 60
        small_margin = 10
        
        # Special handling for glasses in inactive list
        if 'glasses' in inactive_devices:
            glasses_width = small_size * 1.5
            glasses_height = small_size * 0.6
            glasses_x = (self.panel_width - glasses_width) // 2
            
            positions['glasses'] = {
                'center': (glasses_x + glasses_width // 2, start_y + glasses_height // 2),
                'bounds': (glasses_x, start_y, glasses_width, glasses_height),
                'size': small_size,
                'active': False
            }
            
            # Remove glasses from the list and adjust start_y for others
            inactive_devices = [d for d in inactive_devices if d != 'glasses']
            start_y += glasses_height + 15
        
        # Layout remaining inactive devices
        if inactive_devices:
            total_width = len(inactive_devices) * small_size + (len(inactive_devices) - 1) * small_margin
            start_x = (self.panel_width - total_width) // 2
            
            for i, device in enumerate(inactive_devices):
                x = start_x + i * (small_size + small_margin)
                
                positions[device] = {
                    'center': (x + small_size // 2, start_y + small_size // 2),
                    'bounds': (x, start_y, small_size, small_size),
                    'size': small_size,
                    'active': False
                }
    
    def get_panel_height(self, num_devices):
        """Calculate total height needed for all device panels"""
        # Account for glasses potentially taking extra vertical space
        base_rows = (num_devices + self.max_per_row - 1) // self.max_per_row
        extra_height = 60 if num_devices >= 4 else 0  # Extra space for glasses
        return self.start_y + base_rows * (self.box_size + self.margin) + self.margin + extra_height