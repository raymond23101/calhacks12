#!/usr/bin/env python3
"""
Motor Controller for Haptic Feedback
Maps 2x6 depth grid to motors with intensity based on proximity
"""

import numpy as np
import time
from typing import List, Tuple

class MotorController:
    """
    Controls haptic motors based on depth grid
    - Each grid cell (2x6) corresponds to a motor
    - Motor intensity proportional to object proximity
    - Only activates above area threshold
    """
    
    def __init__(self, 
                 grid_shape: Tuple[int, int] = (2, 6),
                 min_distance: float = 1.6,  # feet (closer than this = max intensity)
                 max_distance: float = 10.0,  # feet (farther than this = no activation)
                 area_threshold: float = 0.3,  # minimum area coverage to activate (0-1)
                 intensity_range: Tuple[int, int] = (0, 255)):  # PWM range
        """
        Initialize motor controller
        
        Args:
            grid_shape: Shape of depth grid (rows, cols)
            min_distance: Distance for maximum intensity (feet)
            max_distance: Distance for minimum intensity (feet)
            area_threshold: Minimum coverage threshold to activate motor
            intensity_range: PWM intensity range (min, max)
        """
        self.grid_shape = grid_shape
        self.num_motors = grid_shape[0] * grid_shape[1]
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.area_threshold = area_threshold
        self.intensity_min, self.intensity_max = intensity_range
        
        # Current motor states
        self.motor_intensities = np.zeros(self.num_motors, dtype=np.uint8)
        
        # Hardware interface (placeholder - replace with actual GPIO/serial)
        self.hardware_connected = False
        
        print(f"✓ Motor Controller initialized: {self.num_motors} motors ({grid_shape[0]}x{grid_shape[1]})")
        print(f"  Distance range: {min_distance}ft - {max_distance}ft")
        print(f"  Area threshold: {area_threshold * 100:.0f}%")
        
    def calculate_intensity(self, depth_value: float, area_coverage: float = 1.0) -> int:
        """
        Calculate motor intensity based on depth and area coverage
        
        Args:
            depth_value: Average depth in feet (0 = no data)
            area_coverage: Fraction of grid cell with valid data (0-1)
            
        Returns:
            Motor intensity (0-255)
        """
        # No activation if below area threshold or no depth data
        if area_coverage < self.area_threshold or depth_value <= 0:
            return 0
        
        # Clamp depth to range
        depth_value = np.clip(depth_value, self.min_distance, self.max_distance)
        
        # Calculate intensity (closer = stronger)
        # Linear mapping: min_distance -> max_intensity, max_distance -> min_intensity
        normalized = (self.max_distance - depth_value) / (self.max_distance - self.min_distance)
        intensity = int(normalized * (self.intensity_max - self.intensity_min) + self.intensity_min)
        
        return np.clip(intensity, self.intensity_min, self.intensity_max)
    
    def update_from_depth_grid(self, depth_grid: np.ndarray, verbose: bool = False):
        """
        Update motor intensities from depth grid
        
        Args:
            depth_grid: 2D array of depth values (rows x cols) in feet
            verbose: Print motor states
        """
        if depth_grid.shape != self.grid_shape:
            print(f"⚠️  Depth grid shape {depth_grid.shape} doesn't match expected {self.grid_shape}")
            return
        
        # Calculate area coverage for each cell (assuming depth > 0 means valid data)
        area_coverage = (depth_grid > 0).astype(float)
        
        # Flatten grid to motor array
        depth_flat = depth_grid.flatten()
        area_flat = area_coverage.flatten()
        
        # Calculate intensities for all motors
        new_intensities = np.array([
            self.calculate_intensity(depth_flat[i], area_flat[i])
            for i in range(self.num_motors)
        ], dtype=np.uint8)
        
        # Update motor states
        self.motor_intensities = new_intensities
        
        # Apply to hardware
        self._apply_to_hardware()
        
        if verbose:
            self._print_motor_states(depth_grid)
    
    def _apply_to_hardware(self):
        """
        Apply motor intensities to actual hardware
        Override this method with actual GPIO/serial control
        """
        if self.hardware_connected:
            # TODO: Implement actual hardware control
            # Example: Send via serial, GPIO, I2C, etc.
            pass
        # For now, just simulate
    
    def _print_motor_states(self, depth_grid: np.ndarray):
        """Print current motor states in grid format"""
        print("\n🎮 Motor Grid (Intensity / Depth):")
        for i in range(self.grid_shape[0]):
            row_str = "  "
            for j in range(self.grid_shape[1]):
                idx = i * self.grid_shape[1] + j
                intensity = self.motor_intensities[idx]
                depth = depth_grid[i, j]
                
                # Visual indicator
                if intensity == 0:
                    indicator = "░░░"
                elif intensity < 85:
                    indicator = "▒▒▒"
                elif intensity < 170:
                    indicator = "▓▓▓"
                else:
                    indicator = "███"
                
                row_str += f"{indicator}({intensity:3d}/{depth:4.1f}ft) "
            print(row_str)
        print()
    
    def set_motor(self, motor_index: int, intensity: int):
        """
        Manually set a specific motor intensity
        
        Args:
            motor_index: Motor index (0 to num_motors-1)
            intensity: Intensity value (0-255)
        """
        if 0 <= motor_index < self.num_motors:
            self.motor_intensities[motor_index] = np.clip(intensity, 0, 255)
            self._apply_to_hardware()
        else:
            print(f"⚠️  Invalid motor index: {motor_index}")
    
    def set_all_motors(self, intensity: int):
        """Set all motors to the same intensity"""
        self.motor_intensities[:] = np.clip(intensity, 0, 255)
        self._apply_to_hardware()
    
    def disable_all_motors(self):
        """Turn off all motors"""
        self.set_all_motors(0)
    
    def test_motors(self, duration: float = 0.5):
        """
        Test all motors sequentially
        
        Args:
            duration: Time to activate each motor (seconds)
        """
        print("\n🔧 Testing motors sequentially...")
        for i in range(self.num_motors):
            print(f"  Motor {i}: ON")
            self.set_motor(i, 255)
            time.sleep(duration)
            self.set_motor(i, 0)
        print("✓ Motor test complete")
    
    def get_motor_grid(self) -> np.ndarray:
        """Get current motor intensities as 2D grid"""
        return self.motor_intensities.reshape(self.grid_shape)
    
    def connect_hardware(self, port: str = None):
        """
        Connect to actual motor hardware
        
        Args:
            port: Serial port or GPIO interface
        """
        # TODO: Implement actual hardware connection
        # Example: serial.Serial(port, baudrate=115200)
        print(f"⚠️  Hardware connection not implemented yet")
        print(f"    Motors will run in simulation mode")
        self.hardware_connected = False


def main():
    """Demo: Motor controller with simulated depth data"""
    print("Motor Controller Demo\n")
    
    # Initialize controller
    motors = MotorController(
        grid_shape=(2, 6),
        min_distance=1.6,
        max_distance=10.0,
        area_threshold=0.3
    )
    
    # Test sequence
    print("\n1. Testing all motors...")
    motors.test_motors(duration=0.3)
    
    # Simulate depth grid data
    print("\n2. Simulating depth-based activation...")
    
    # Scene 1: Object close on left side
    print("\n  Scene 1: Object close on left (3.3ft)")
    depth_grid = np.array([
        [3.3, 8.2, 9.8, 11.5, 13.1, 14.8],  # Top row (in feet)
        [3.9, 8.9, 10.5, 12.1, 13.8, 15.4]   # Bottom row
    ])
    motors.update_from_depth_grid(depth_grid, verbose=True)
    time.sleep(1)
    
    # Scene 2: Object close in center
    print("\n  Scene 2: Object close in center (2.6ft)")
    depth_grid = np.array([
        [9.8, 6.6, 2.6, 3.0, 6.6, 9.8],  # Top row (in feet)
        [10.5, 7.2, 3.3, 3.6, 7.2, 10.5]   # Bottom row
    ])
    motors.update_from_depth_grid(depth_grid, verbose=True)
    time.sleep(1)
    
    # Scene 3: Clear scene (far away)
    print("\n  Scene 3: Clear scene (all far)")
    depth_grid = np.full((2, 6), 13.1)  # 13.1 feet (4 meters)
    motors.update_from_depth_grid(depth_grid, verbose=True)
    time.sleep(1)
    
    # Turn off all motors
    motors.disable_all_motors()
    print("\n✓ Demo complete")


if __name__ == "__main__":
    main()

