import cv2
import numpy as np
import pyrealsense2 as rs

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

# Alignment setup
align_to = rs.stream.color
align = rs.align(align_to)

def create_visualization(color_img, depth_img, processing_steps):
    """Create a comprehensive visualization grid"""
    # Convert all processing steps to BGR for display
    vis_steps = {}
    for name, img in processing_steps.items():
        if name == "detection":
            continue  # Skip the detection tuple
        vis_steps[name] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
    
    # Top row: Original, Grayscale, Blurred
    row1 = np.hstack((vis_steps["1_Original"],
                     vis_steps["2_Grayscale"],
                     vis_steps["3_Blurred"]))
    
    # Middle row: Threshold, Eroded, Dilated
    row2 = np.hstack((vis_steps["4_Threshold"],
                     vis_steps["5_Eroded"],
                     vis_steps["6_Dilated"]))
    
    # Bottom row: Detection result and Depth
    detection_vis = vis_steps["1_Original"].copy()
    if "detection" in processing_steps:
        (x,y), radius, distance = processing_steps["detection"]
        center = (int(x), int(y))
        cv2.circle(detection_vis, center, int(radius), (0,255,0), 2)
        cv2.putText(detection_vis, f"{distance:.2f}m", 
                   (center[0]-30, center[1]-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_img, alpha=0.03),
        cv2.COLORMAP_JET
    )
    row3 = np.hstack((detection_vis, depth_colormap))
    
    # Combine all rows
    return np.vstack((row1, row2, row3))

try:
    while True:
        # Get frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert images
        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(depth_frame.get_data())

        # Processing pipeline
        processing_steps = {
            "1_Original": color_img.copy(),
            "2_Grayscale": cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY),
        }
        
        # Blur
        processing_steps["3_Blurred"] = cv2.GaussianBlur(
            processing_steps["2_Grayscale"], (11, 11), 0)
        
        # Threshold
        processing_steps["4_Threshold"] = cv2.adaptiveThreshold(
            processing_steps["3_Blurred"], 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morphological operations
        kernel = np.ones((5,5), np.uint8)
        processing_steps["5_Eroded"] = cv2.erode(
            processing_steps["4_Threshold"], kernel, iterations=1)
        processing_steps["6_Dilated"] = cv2.dilate(
            processing_steps["5_Eroded"], kernel, iterations=2)
        
        # Detection
        contours, _ = cv2.findContours(
            processing_steps["6_Dilated"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 300:
                (x,y), radius = cv2.minEnclosingCircle(largest_contour)
                center = (int(x), int(y))
                distance = depth_frame.get_distance(center[0], center[1])
                processing_steps["detection"] = ((x,y), radius, distance)

        # Create and show visualization
        visualization = create_visualization(color_img, depth_img, processing_steps)
        cv2.imshow("Ball Detection Pipeline", visualization)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()