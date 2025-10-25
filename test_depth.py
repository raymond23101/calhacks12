#!/usr/bin/env python3
"""
Real-time depth grid analyzer for OAK-D camera
Divides depth image into 2x6 grid and sums depth values in each region
"""

import cv2
import depthai as dai
import numpy as np
import time

def create_depth_pipeline():
    """Create OAK-D pipeline for stereo depth and RGB"""
    pipeline = dai.Pipeline()
    
    # Create RGB camera
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(640, 400)  # Match depth resolution aspect ratio
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    
    # Create stereo depth nodes
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    
    # Configure mono cameras
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    
    # Configure stereo depth
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(False)
    
    # Link nodes
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)
    
    # Create outputs
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)
    
    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)
    
    return pipeline

def analyze_depth_grid(depth_frame, grid_rows=2, grid_cols=6):
    """
    Divide depth frame into grid and sum depth values in each region
    
    Args:
        depth_frame: Depth frame from OAK-D camera
        grid_rows: Number of rows in grid (default: 2)
        grid_cols: Number of columns in grid (default: 6)
    
    Returns:
        2D array of depth sums for each grid region
    """
    height, width = depth_frame.shape
    
    # Calculate region dimensions
    region_height = height // grid_rows
    region_width = width // grid_cols
    
    # Create result array
    depth_sums = np.zeros((grid_rows, grid_cols), dtype=np.float64)
    
    # Sum depth values in each region
    for row in range(grid_rows):
        for col in range(grid_cols):
            # Calculate region boundaries
            y_start = row * region_height
            y_end = (row + 1) * region_height if row < grid_rows - 1 else height
            x_start = col * region_width
            x_end = (col + 1) * region_width if col < grid_cols - 1 else width
            
            # Extract region and sum depth values
            region = depth_frame[y_start:y_end, x_start:x_end]
            depth_sums[row, col] = np.sum(region, dtype=np.float64)
    
    return depth_sums

def main():
    """Main loop for real-time depth grid analysis"""
    print("Initializing OAK-D camera for depth sensing...")
    print("Grid size: 2 rows x 6 columns")
    print("Press 'q' to quit\n")
    
    # Create pipeline and connect to device
    pipeline = create_depth_pipeline()
    device = dai.Device(pipeline)
    
    # Get output queues
    rgb_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    depth_queue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    
    print("✓ Camera initialized successfully!")
    print("Starting real-time depth analysis...\n")
    
    frame_count = 0
    start_time = time.time()
    rgb_frame = None
    
    try:
        while True:
            # Get RGB frame
            if rgb_queue.has():
                rgb_data = rgb_queue.get()
                rgb_frame = rgb_data.getCvFrame()
            
            # Get depth frame
            if depth_queue.has():
                depth_data = depth_queue.get()
                depth_frame = depth_data.getFrame()
                
                # Analyze depth grid
                depth_sums = analyze_depth_grid(depth_frame, grid_rows=2, grid_cols=6)
                
                # Print results
                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                print(f"\r[Frame {frame_count:5d} | FPS: {fps:6.2f}]", end=" ")
                print(depth_sums.astype(np.int64))
                
                # Visualize: Overlay depth on RGB
                if rgb_frame is not None:
                    # Normalize depth for visualization
                    depth_normalized = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                    
                    # Resize depth to match RGB frame size if needed
                    if depth_colored.shape != rgb_frame.shape:
                        depth_colored = cv2.resize(depth_colored, (rgb_frame.shape[1], rgb_frame.shape[0]))
                    
                    # Blend RGB with depth heatmap (50% overlay)
                    overlay = cv2.addWeighted(rgb_frame, 0.5, depth_colored, 0.5, 0)
                    
                    # Draw grid lines on overlay
                    height, width = overlay.shape[:2]
                    region_height = height // 2
                    region_width = width // 6
                    
                    # Draw horizontal line
                    cv2.line(overlay, (0, region_height), (width, region_height), (255, 255, 255), 2)
                    
                    # Draw vertical lines
                    for i in range(1, 6):
                        x = i * region_width
                        cv2.line(overlay, (x, 0), (x, height), (255, 255, 255), 2)
                    
                    # Add text labels for grid cells
                    for row in range(2):
                        for col in range(6):
                            x = col * region_width + region_width // 2
                            y = row * region_height + region_height // 2
                            depth_value = depth_sums[row, col] / 1000000  # Convert to millions for readability
                            text = f"{depth_value:.1f}M"
                            
                            # Add background rectangle for text
                            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                            cv2.rectangle(overlay, 
                                        (x - text_size[0]//2 - 5, y - text_size[1]//2 - 5),
                                        (x + text_size[0]//2 + 5, y + text_size[1]//2 + 5),
                                        (0, 0, 0), -1)
                            
                            # Draw text
                            cv2.putText(overlay, text, 
                                      (x - text_size[0]//2, y + text_size[1]//2),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    cv2.imshow('RGB + Depth Overlay (2x6 Grid)', overlay)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        print("\n\nCleaning up...")
        cv2.destroyAllWindows()
        device.close()
        print("✓ Done!")

if __name__ == "__main__":
    main()

