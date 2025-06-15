import cv2
import numpy as np
import pyrealsense2 as rs
import json

# Load extrinsic parameters
with open('realsense_extrinsics.json') as f:
    extrinsics = json.load(f)

# Convert to numpy arrays
rvec = np.array(extrinsics['rotation_vector'])
tvec = np.array(extrinsics['translation_vector'])
rmat = np.array(extrinsics['rotation_matrix'])  # Rotation matrix

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable streams
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

# Alignment setup
align_to = rs.stream.color
align = rs.align(align_to)

# Add this after alignment setup
depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

# Color ranges (HSV)
BLUE_LOWER = (75, 100, 30)    # Hue (75-150), Saturation (100-255), Value (30-150)
BLUE_UPPER = (150, 255, 150)
YELLOW_LOWER = (20, 100, 100)  # Hue (20-40), Saturation (100-255), Value (100-255)
YELLOW_UPPER = (40, 255, 255)

# Initialize trajectory storage (stores pixel coordinates)
trajectory = []
max_trajectory_points = 50  # Adjust based on desired trail length

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
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_img, alpha=0.03), 
            cv2.COLORMAP_JET
        )

        # Convert to HSV
        hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)  # For subpixel refinement

        # Create masks for both colors
        blue_mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
        yellow_mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
        combined_mask = cv2.bitwise_or(blue_mask, yellow_mask)

        # Noise reduction
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        clean_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(clean_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            if radius > 10:  # Only proceed if radius is reasonable
                center = (int(x), int(y))
                radius = int(radius)

                # Get color label
                pixel_center = hsv[int(y), int(x)]  # Get HSV at center
                if (YELLOW_LOWER[0] <= pixel_center[0] <= YELLOW_UPPER[0] and
                    YELLOW_LOWER[1] <= pixel_center[1] <= YELLOW_UPPER[1] and
                    YELLOW_LOWER[2] <= pixel_center[2] <= YELLOW_UPPER[2]):
                    color_name = "YELLOW"
                    circle_color = (0, 255, 255)  # Yellow in BGR
                else:
                    color_name = "BLUE"
                    circle_color = (255, 0, 0)    # Blue in BGR

                # Get distance in meters
                distance = depth_frame.get_distance(center[0], center[1])
                
                # Convert 2D pixel to 3D point (camera frame)
                ball_camera = rs.rs2_deproject_pixel_to_point(
                    depth_intrinsics,
                    [center[0], center[1]],
                    distance
                )
                ball_camera = np.array(ball_camera)  # Convert to numpy array
                
                # Transform to world frame
                ball_world = rmat @ ball_camera + tvec.flatten()

                # Store pixel position for trajectory
                trajectory.append(center)
                if len(trajectory) > max_trajectory_points:
                    trajectory.pop(0)  # Remove oldest point

                print(f"\n{color_name} ball detected!")
                print(f"Pixel coordinates: {center}")
                print(f"Camera frame (X,Y,Z): {ball_camera}")
                print(f"World frame (X,Y,Z): {ball_world}")

                # Draw trajectory (on color_img only)
                if len(trajectory) > 1:
                    for i in range(1, len(trajectory)):
                        cv2.line(color_img, trajectory[i-1], trajectory[i], (0, 255, 0), 2)  # Green trail

                # Draw on both images
                for img in [color_img, depth_colormap]:
                    cv2.circle(img, center, radius, circle_color, 2)
                    cv2.circle(img, center, 3, (0, 0, 255), -1)  # Red center dot
                    
                    # Display info
                    cv2.putText(img, f"{color_name} {distance:.2f}m", 
                               (center[0] - 50, center[1] - radius - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display side-by-side
        combined = np.hstack((color_img, depth_colormap))
        cv2.namedWindow("Dual-Color Ball Tracking", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Dual-Color Ball Tracking", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()