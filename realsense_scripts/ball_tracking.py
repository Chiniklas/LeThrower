import cv2
import numpy as np
import pyrealsense2 as rs
import json
from collections import deque

# Load extrinsic parameters
with open('realsense_extrinsics.json') as f:
    extrinsics = json.load(f)

# Convert to numpy arrays
rvec = np.array(extrinsics['rotation_vector'])
tvec = np.array(extrinsics['translation_vector'])
rmat = np.array(extrinsics['rotation_matrix'])

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

# Alignment setup
align_to = rs.stream.color
align = rs.align(align_to)
depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

# Color ranges (HSV)
BLUE_LOWER = (75, 100, 30)
BLUE_UPPER = (150, 255, 150)
YELLOW_LOWER = (20, 100, 100)
YELLOW_UPPER = (40, 255, 255)

# Initialize trajectory storage
trajectory = deque(maxlen=50)
world_positions = deque(maxlen=50)
timestamps = deque(maxlen=50)

# Physics constants
GRAVITY = 9.81  # m/s²

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            continue

        current_time = color_frame.get_timestamp() / 1000
        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)

        # Processing
        hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
        yellow_mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
        combined_mask = cv2.bitwise_or(blue_mask, yellow_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        clean_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)

        contours, _ = cv2.findContours(clean_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))  # Convert to integers
            radius = int(radius)  # Convert radius to integer
            
            if radius > 10:
                # Ensure coordinates are within image bounds
                if (center[0] >= 0 and center[0] < color_img.shape[1] and 
                    center[1] >= 0 and center[1] < color_img.shape[0]):
                    
                    # Color detection - now using safe integer indices
                    pixel_center = hsv[center[1], center[0]]  # Note: hsv uses (y,x) indexing
                    
                    if (YELLOW_LOWER[0] <= pixel_center[0] <= YELLOW_UPPER[0] and
                        YELLOW_LOWER[1] <= pixel_center[1] <= YELLOW_UPPER[1] and
                        YELLOW_LOWER[2] <= pixel_center[2] <= YELLOW_UPPER[2]):
                        color_name = "YELLOW"
                        circle_color = (0, 255, 255)
                    else:
                        color_name = "BLUE"
                        circle_color = (255, 0, 0)

                    # 3D position
                    distance = depth_frame.get_distance(center[0], center[1])
                    ball_camera = rs.rs2_deproject_pixel_to_point(
                        depth_intrinsics, 
                        [center[0], center[1]], 
                        distance
                    )
                    ball_world = rmat @ np.array(ball_camera) + tvec.flatten()

                    # Store data
                    if len(world_positions) == 0 or np.linalg.norm(ball_world - world_positions[-1]) > 0.01:
                        world_positions.append(ball_world)
                        timestamps.append(current_time)
                    trajectory.append(center)

                    # Velocity calculation
                    velocity = np.zeros(3)
                    if len(world_positions) >= 2 and (timestamps[-1] - timestamps[-2]) > 0:
                        velocity = (world_positions[-1] - world_positions[-2]) / (timestamps[-1] - timestamps[-2])

                    # Trajectory prediction
                    prediction_steps = 20
                    predicted_pixels = []
                    current_pos = ball_world.copy()
                    current_vel = velocity.copy() if np.linalg.norm(velocity) > 0.1 else np.array([0.1, 0.1, 0.1])

                    for i in range(prediction_steps):
                        dt = 0.15 / prediction_steps
                        current_vel[2] -= GRAVITY * dt
                        current_pos += current_vel * dt
                        
                        # Convert to pixel coordinates
                        camera_point = np.linalg.inv(rmat) @ (current_pos - tvec.flatten())
                        pixel = rs.rs2_project_point_to_pixel(depth_intrinsics, camera_point)
                        pixel = (max(0, min(int(pixel[0]), color_img.shape[1]-1)), 
                                max(0, min(int(pixel[1]), color_img.shape[0]-1)))
                        predicted_pixels.append(pixel)
                        
                        # Draw prediction
                        alpha = i / prediction_steps
                        color = (0, int(255 * (1 - alpha)), int(255 * alpha))
                        cv2.circle(color_img, pixel, 2, color, -1)
                        if i > 0:
                            cv2.line(color_img, predicted_pixels[i-1], pixel, (0, 0, 255), 3)
                    # print(f"Predicted trajectory: {predicted_pixels}")
                    # input()
                    # Draw current position
                    for img in [color_img, depth_colormap]:
                        cv2.circle(img, center, radius, circle_color, 2)
                        cv2.circle(img, center, 3, (0, 0, 255), -1)
                        cv2.putText(img, f"{color_name} {distance:.2f}m", 
                                  (center[0] - 50, center[1] - radius - 15),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(img, f"Vel: {np.linalg.norm(velocity):.2f}m/s",
                                  (center[0] - 50, center[1] - radius - 35),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # Draw trajectory
                    if len(trajectory) > 1:
                        for i in range(1, len(trajectory)):
                            cv2.line(color_img, trajectory[i-1], trajectory[i], (0, 255, 0), 2)

        # Display
        combined = np.hstack((color_img, depth_colormap))
        cv2.imshow("Ball Tracking", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()