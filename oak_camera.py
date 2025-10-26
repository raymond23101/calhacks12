#!/usr/bin/env python3
"""
OAK-D Lite Camera Interface
Based on OAKLite library pattern - RGB + Stereo Depth only
"""

import cv2
import numpy as np
import depthai as dai
import threading
import time


class OAKCamera:
    """
    OAK-D Lite Camera - RGB and Stereo Depth
    Following OAKLite library pattern
    """
    
    FOCAL_LEN = 441
    BASELINE = 3.1  # cm
    MAX_DIST = 12 * 25  # 300 cm
    
    def __init__(self,
                 res="480",
                 median='7x7',
                 lrcheck=True,
                 extended=False,
                 subpixel=True,
                 min_depth=300,
                 max_depth=5000,
                 DEBUG_MODE=True):
        """
        Initialize OAK-D Lite camera
        
        Args:
            res: Resolution ("800" | "720" | "480")
            median: Median filter ("OFF" | "3x3" | "5x5" | "7x7")
            lrcheck: Left-right check for better occlusion handling
            extended: Extended disparity for closer minimum depth
            subpixel: Subpixel mode for better accuracy at distance
            min_depth: Minimum depth in millimeters (default 300mm = 30cm)
            max_depth: Maximum depth in millimeters (default 5000mm = 5m)
            DEBUG_MODE: Enable debug output
        """
        self.readingMutex = False
        self.DEBUG_MODE = DEBUG_MODE
        
        # Color map for depth visualization
        self.cvColorMap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_TURBO)
        self.cvColorMap[0] = [0, 0, 0]
        
        # Resolution configuration
        RES_MAP = {
            '800': {'w': 1280, 'h': 800, 'res': dai.MonoCameraProperties.SensorResolution.THE_800_P},
            '720': {'w': 1280, 'h': 720, 'res': dai.MonoCameraProperties.SensorResolution.THE_720_P},
            '480': {'w': 640, 'h': 480, 'res': dai.MonoCameraProperties.SensorResolution.THE_480_P}
        }
        
        # Median filter configuration
        medianMap = {
            "OFF": dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF,
            "3x3": dai.StereoDepthProperties.MedianFilter.KERNEL_3x3,
            "5x5": dai.StereoDepthProperties.MedianFilter.KERNEL_5x5,
            "7x7": dai.StereoDepthProperties.MedianFilter.KERNEL_7x7,
        }
        
        if res not in RES_MAP:
            raise ValueError(f"Unsupported resolution: {res}")
        if median not in medianMap:
            raise ValueError(f"Unsupported median: {median}")
        
        self.resolution = RES_MAP[res]
        self.median = medianMap[median]
        self.lrcheck = lrcheck
        self.extended = extended
        self.subpixel = subpixel
        self.min_depth = min_depth
        self.max_depth = max_depth
        
        print("="*70)
        print("OAK-D LITE CAMERA")
        print("="*70)
        print(f"Resolution:         {self.resolution['w']}x{self.resolution['h']}")
        print(f"Depth range:        {self.min_depth}mm - {self.max_depth}mm ({self.min_depth/10:.1f}cm - {self.max_depth/10:.1f}cm)")
        print(f"Left-Right check:   {self.lrcheck}")
        print(f"Extended disparity: {self.extended}")
        print(f"Subpixel:           {self.subpixel}")
        print(f"Median filtering:   {median}")
        print()
        
        # Depth grid temporal smoothing (noise reduction)
        self.previous_depth_grid = None
        self.depth_grid_alpha = 0.3  # EMA smoothing factor (0.3 = 70% old, 30% new)
        
        # Create device and pipeline
        self._create_pipeline()
        
    def _create_pipeline(self):
        """Create DepthAI pipeline"""
        print("Creating pipeline...")
        self.device = dai.Device()
        self.pipeline = dai.Pipeline()
        
        # RGB Camera
        print("  - RGB Camera (CAM_A)")
        self.camRgb = self.pipeline.create(dai.node.ColorCamera)
        self.camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        self.camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        # Use ISP scaling instead of video size to avoid cropping
        self.camRgb.setIspScale(1, 3)  # Scale down to 1920/3 = 640 width
        self.camRgb.setInterleaved(False)
        self.camRgb.setFps(30)
        
        # Mono cameras
        print("  - Mono Cameras (left, right)")
        self.camLeft = self.pipeline.create(dai.node.MonoCamera)
        self.camRight = self.pipeline.create(dai.node.MonoCamera)
        
        self.camLeft.setCamera("left")
        self.camRight.setCamera("right")
        self.camLeft.setResolution(self.resolution['res'])
        self.camRight.setResolution(self.resolution['res'])
        self.camLeft.setFps(30)
        self.camRight.setFps(30)
        
        # Stereo depth
        print("  - StereoDepth node")
        self.stereo = self.pipeline.create(dai.node.StereoDepth)
        self.stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        self.stereo.initialConfig.setMedianFilter(self.median)
        self.stereo.setRectifyEdgeFillColor(0)
        self.stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        self.stereo.setLeftRightCheck(self.lrcheck)
        self.stereo.setExtendedDisparity(self.extended)
        self.stereo.setSubpixel(self.subpixel)
        
        # Optimize depth quality - remove graininess and noise
        config = self.stereo.initialConfig.get()
        
        # Set depth range to filter noise (removes very close and very far readings)
        config.postProcessing.thresholdFilter.minRange = self.min_depth  # Min depth in mm
        config.postProcessing.thresholdFilter.maxRange = self.max_depth  # Max depth in mm
        
        # Temporal filtering - smooth over time to reduce flickering/grain
        config.postProcessing.temporalFilter.enable = True
        config.postProcessing.temporalFilter.alpha = 0.4  # Increased for smoother depth
        config.postProcessing.temporalFilter.delta = 20  # Filter larger changes for stability
        config.postProcessing.temporalFilter.persistencyMode = dai.StereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_8_OUT_OF_8
        
        # Spatial edge-preserving filtering - removes graininess while keeping edges
        config.postProcessing.spatialFilter.enable = True
        config.postProcessing.spatialFilter.holeFillingRadius = 2
        config.postProcessing.spatialFilter.numIterations = 1
        config.postProcessing.spatialFilter.alpha = 0.5  # Edge preservation
        config.postProcessing.spatialFilter.delta = 32  # Spatial filter delta
        
        # Decimation - reduce resolution for faster processing (optional, improves smoothness)
        config.postProcessing.decimationFilter.decimationFactor = 1  # No decimation
        config.postProcessing.decimationFilter.decimationMode = dai.StereoDepthConfig.PostProcessing.DecimationFilter.DecimationMode.PIXEL_SKIPPING
        
        # Speckle filtering - removes small isolated noise
        config.postProcessing.speckleFilter.enable = True
        config.postProcessing.speckleFilter.speckleRange = 50
        
        # Higher confidence threshold for cleaner depth (removes uncertain pixels)
        config.costMatching.confidenceThreshold = 230
        
        self.stereo.initialConfig.set(config)
        
        # Output streams
        print("  - Creating output streams")
        self.rgbOut = self.pipeline.create(dai.node.XLinkOut)
        self.xoutLeft = self.pipeline.create(dai.node.XLinkOut)
        self.xoutRight = self.pipeline.create(dai.node.XLinkOut)
        self.xoutDisparity = self.pipeline.create(dai.node.XLinkOut)
        self.xoutDepth = self.pipeline.create(dai.node.XLinkOut)
        
        self.rgbOut.setStreamName("rgb")
        self.xoutLeft.setStreamName("left")
        self.xoutRight.setStreamName("right")
        self.xoutDisparity.setStreamName("disparity")
        self.xoutDepth.setStreamName("depth")
        
        # Link nodes
        print("  - Linking pipeline nodes")
        self.camLeft.out.link(self.stereo.left)
        self.camRight.out.link(self.stereo.right)
        self.stereo.syncedLeft.link(self.xoutLeft.input)
        self.stereo.syncedRight.link(self.xoutRight.input)
        self.stereo.disparity.link(self.xoutDisparity.input)
        self.stereo.depth.link(self.xoutDepth.input)
        self.camRgb.video.link(self.rgbOut.input)
        
        self.streams = ["rgb", "left", "right", "disparity", "depth"]
        
        print("✓ Pipeline created\n")
    
    def startDevice(self):
        """Start the device and begin frame reading thread"""
        print("Starting device...")
        self.frameBufferDict = {}
        self.readingMutex = False
        self.running = True
        
        # Start background thread
        self.readThread = threading.Thread(target=self.readFrames, daemon=True)
        self.readThread.start()
        
        print("✓ Device started\n")
    
    def readFrames(self):
        """Background thread to continuously read frames"""
        print("Starting DepthAI device")
        with self.device:
            self.device.startPipeline(self.pipeline)
            
            # Get USB speed info
            try:
                usb_speed = self.device.getUsbSpeed()
                device_name = self.device.getDeviceName()
                print(f"✓ Device: {device_name}")
                print(f"✓ USB Speed: {usb_speed.name}")
                if usb_speed == dai.UsbSpeed.HIGH:
                    print("⚠️  WARNING: USB 2.0 detected!")
                    print("   For best performance, use USB 3.0 port\n")
            except:
                pass
            
            # Create output queues with minimal buffer for lowest latency
            qRgb = self.device.getOutputQueue("rgb", maxSize=1, blocking=False)
            qLeft = self.device.getOutputQueue("left", maxSize=1, blocking=False)
            qRight = self.device.getOutputQueue("right", maxSize=1, blocking=False)
            qDisp = self.device.getOutputQueue("disparity", maxSize=1, blocking=False)
            qDepth = self.device.getOutputQueue("depth", maxSize=1, blocking=False)
            
            # Pre-compute disparity normalization factor
            maxDisp = self.stereo.initialConfig.getMaxDisparity()
            dispScale = 255.0 / maxDisp
            
            while self.running:
                # Process all available frames without blocking on mutex for each frame
                # This significantly improves throughput
                
                # RGB
                inRgb = qRgb.tryGet()
                if inRgb is not None and not self.readingMutex:
                    self.frameBufferDict["rgb"] = inRgb.getCvFrame()
                
                # Left
                inLeft = qLeft.tryGet()
                if inLeft is not None and not self.readingMutex:
                    self.frameBufferDict["left"] = inLeft.getCvFrame()
                
                # Right
                inRight = qRight.tryGet()
                if inRight is not None and not self.readingMutex:
                    self.frameBufferDict["right"] = inRight.getCvFrame()
                
                # Disparity - optimized processing
                inDisp = qDisp.tryGet()
                if inDisp is not None and not self.readingMutex:
                    dispFrame = inDisp.getFrame()
                    self.frameBufferDict["disparity_raw"] = dispFrame
                    # Fast disparity colorization
                    disp = (dispFrame * dispScale).astype(np.uint8)
                    self.frameBufferDict["disparity"] = cv2.applyColorMap(disp, self.cvColorMap)
                
                # Depth
                inDepth = qDepth.tryGet()
                if inDepth is not None and not self.readingMutex:
                    self.frameBufferDict["depth"] = inDepth.getFrame().astype(np.uint16)
                
                # Small sleep to prevent CPU spinning
                time.sleep(0.001)
    
    def getFrames(self, names):
        """
        Get frames from buffer (thread-safe)
        
        Args:
            names: List of frame names ["rgb", "depth", "disparity", "left", "right"]
        
        Returns:
            List of frames in same order as names
        """
        temp = []
        self.readingMutex = True
        
        for name in names:
            if name in self.frameBufferDict:
                temp.append(self.frameBufferDict[name].copy())
            else:
                temp.append(None)
        
        self.readingMutex = False
        return temp
    
    def halt(self):
        """Wait for mutex release"""
        while self.readingMutex:
            time.sleep(0.001)
    
    def getDepth(self, x, y):
        """
        Get depth value at pixel coordinates
        
        Args:
            x, y: Pixel coordinates
        
        Returns:
            Depth in centimeters
        """
        frames = self.getFrames(["disparity_raw"])
        disp = frames[0]
        
        if disp is None:
            return self.MAX_DIST
        
        h, w = disp.shape
        if y >= h or x >= w or y < 0 or x < 0:
            return self.MAX_DIST
        
        dispVal = disp[int(y), int(x)]
        if dispVal == 0:
            return self.MAX_DIST
        
        return self.FOCAL_LEN * self.BASELINE / dispVal
    
    def getDepthGrid(self, rows=2, cols=6, min_cluster_pixels=15, depth_tolerance_feet=1.0):
        """
        Get depth grid prioritizing closest objects with cluster validation
        
        Algorithm:
        1. Find closest depth in region
        2. Check if there's a cluster of similar depths (avoid noise)
        3. If substantial cluster exists, use closest depth
        4. Otherwise, fall back to average depth
        
        Args:
            rows: Number of rows (default 2)
            cols: Number of columns (default 6)
            min_cluster_pixels: Minimum pixels for valid close object (default 15)
            depth_tolerance_feet: Max depth difference for clustering (default 1.0 feet)
        
        Returns:
            numpy array of shape (rows, cols) with depth values in FEET
        """
        frames = self.getFrames(["depth"])
        depth_frame = frames[0]
        
        if depth_frame is None:
            return np.zeros((rows, cols), dtype=np.float32)
        
        h, w = depth_frame.shape[:2] if len(depth_frame.shape) == 3 else depth_frame.shape
        cell_h = h // rows
        cell_w = w // cols
        
        grid = np.zeros((rows, cols), dtype=np.float32)
        
        # Convert tolerance to millimeters for comparison
        tolerance_mm = depth_tolerance_feet * 30.48 * 10.0  # feet -> cm -> mm
        
        for r in range(rows):
            for c in range(cols):
                y1 = r * cell_h
                y2 = (r + 1) * cell_h if r < rows - 1 else h
                x1 = c * cell_w
                x2 = (c + 1) * cell_w if c < cols - 1 else w
                
                # Get depth values in this cell (depth frame is in millimeters)
                if len(depth_frame.shape) == 3:
                    cell = depth_frame[y1:y2, x1:x2, 0]  # Take first channel if 3D
                else:
                    cell = depth_frame[y1:y2, x1:x2]
                
                # Filter out invalid depth values (0 or very large)
                valid_depths = cell[(cell > 0) & (cell < 65535)]
                
                if len(valid_depths) > 0:
                    # Find minimum (closest) depth
                    min_depth_mm = np.min(valid_depths)
                    
                    # Count pixels within tolerance of minimum depth (cluster detection)
                    close_cluster = valid_depths[valid_depths <= (min_depth_mm + tolerance_mm)]
                    cluster_size = len(close_cluster)
                    
                    # If substantial cluster exists at close depth, use it (avoids noise)
                    if cluster_size >= min_cluster_pixels:
                        # Use closest depth (prioritize small close objects)
                        depth_mm = min_depth_mm
                    else:
                        # Fall back to average depth (no significant close object)
                        depth_mm = np.mean(valid_depths)
                    
                    # Convert to feet
                    depth_cm = depth_mm / 10.0  # mm to cm
                    depth_feet = depth_cm / 30.48  # cm to feet
                    grid[r, c] = depth_feet
                else:
                    grid[r, c] = 0.0  # No valid depth data
        
        # Apply temporal smoothing to reduce flickering
        if self.previous_depth_grid is not None:
            # Exponential Moving Average (EMA)
            # new_value = alpha * current + (1 - alpha) * previous
            # This smooths out rapid changes while still responding to real movement
            for r in range(rows):
                for c in range(cols):
                    if grid[r, c] > 0:  # Only smooth valid readings
                        if self.previous_depth_grid[r, c] > 0:  # Previous was valid
                            # Smooth transition
                            grid[r, c] = (self.depth_grid_alpha * grid[r, c] + 
                                        (1 - self.depth_grid_alpha) * self.previous_depth_grid[r, c])
                        # else: first valid reading, use as-is
        
        # Store for next frame
        self.previous_depth_grid = grid.copy()
        
        return grid
    
    def stop(self):
        """Stop the device and cleanup"""
        print("\nStopping device...")
        self.running = False
        if hasattr(self, 'readThread'):
            self.readThread.join(timeout=2.0)
        print("✓ Device stopped")


def main():
    """Demo: Display RGB, depth, and 2x6 depth grid"""
    print("OAK-D Lite Camera Demo\n")
    
    # Create camera with optimized settings
    # min_depth=300mm (30cm), max_depth=5000mm (5m) to filter noise
    cam = OAKCamera(res="480", median="5x5", min_depth=300, max_depth=5000, DEBUG_MODE=True)
    
    # Start device
    cam.startDevice()
    
    # Wait for frames
    time.sleep(2.0)
    
    print("Running camera demo...")
    print("Controls:")
    print("  'q' - Quit")
    print("  'd' - Toggle depth overlay")
    print("  'g' - Toggle grid overlay")
    print()
    
    show_depth = True
    show_grid = True
    frame_count = 0
    start_time = time.time()
    fps = 0.0
    
    # Pre-allocate visualization arrays for better performance
    depth_grid = np.zeros((2, 6), dtype=np.int64)
    
    try:
        while True:
            # Get frames
            frames = cam.getFrames(["rgb", "disparity"])
            rgb = frames[0]
            disparity = frames[1]
            
            if rgb is None or disparity is None:
                time.sleep(0.001)
                continue
            
            frame_count += 1
            
            # Calculate FPS every 10 frames to reduce overhead
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                if elapsed > 0.5:
                    fps = frame_count / elapsed
                    frame_count = 0
                    start_time = time.time()
            
            # Prepare RGB and depth for display
            h, w = rgb.shape[:2]
            
            # Resize disparity to match RGB if needed
            if disparity.shape[:2] != (h, w):
                depth_display = cv2.resize(disparity, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                depth_display = disparity
            
            # RGB is already the right size
            rgb_display = rgb
            
            # Add FPS to RGB
            info = f"FPS: {fps:.1f}"
            cv2.putText(rgb_display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(rgb_display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(rgb_display, "RGB", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add label to depth
            cv2.putText(depth_display, "Depth Heatmap", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw grid on depth if enabled
            if show_grid:
                depth_grid = cam.getDepthGrid(rows=2, cols=6)
                
                rows, cols = 2, 6
                cell_h = h // rows
                cell_w = w // cols
                
                # Draw grid lines on depth
                for r in range(1, rows):
                    y = r * cell_h
                    cv2.line(depth_display, (0, y), (w, y), (0, 255, 0), 2)
                
                for c in range(1, cols):
                    x = c * cell_w
                    cv2.line(depth_display, (x, 0), (x, h), (0, 255, 0), 2)
                
                # Draw depth sums on depth heatmap
                for r in range(rows):
                    for c in range(cols):
                        y = r * cell_h + cell_h // 2
                        x = c * cell_w + cell_w // 2
                        text = f"{int(depth_grid[r, c]/1000)}K"
                        
                        # Simple text with black outline
                        cv2.putText(depth_display, text, (x - 20, y + 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                        cv2.putText(depth_display, text, (x - 20, y + 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Print grid occasionally
            if show_grid and frame_count == 1:
                print(f"[FPS: {fps:5.1f}] Depth Grid:")
                print(depth_grid)
                print()
            
            # Show side-by-side if depth enabled, otherwise just RGB
            if show_depth:
                # Combine side-by-side
                display = np.hstack([rgb_display, depth_display])
            else:
                display = rgb_display
            
            cv2.imshow("OAK-D Lite Camera", display)
            
            # Handle keys
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('d'):
                show_depth = not show_depth
                print(f"Depth overlay: {'ON' if show_depth else 'OFF'}")
            elif key == ord('g'):
                show_grid = not show_grid
                print(f"Grid overlay: {'ON' if show_grid else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print("✓ Done")


if __name__ == "__main__":
    main()

