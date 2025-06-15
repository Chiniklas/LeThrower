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
YELLOW_BALL_LOWER = (20, 100, 100)  # Yellow balls
YELLOW_BALL_UPPER = (40, 255, 255)

BLUE_BIN_LOWER = (75, 100, 30)     # Blue bins
BLUE_BIN_UPPER = (150, 255, 150)
RED_BIN_LOWER1 = (0, 80, 50)       # Red bins (two ranges)
RED_BIN_UPPER1 = (10, 255, 255)
RED_BIN_LOWER2 = (160, 80, 50)
RED_BIN_UPPER2 = (180, 255, 255)

# Detection parameters
BIN_MIN_AREA = 3000                # Minimum contour area for bins
BIN_SOLIDITY_THRESH = 0.85         # Shape completeness for bins
BALL_MIN_RADIUS = 10               # Minimum ball radius

# Trajectory prediction
trajectory = deque(maxlen=20)       # Stores pixel positions
world_positions = deque(maxlen=20)  # Stores 3D world positions
timestamps = deque(maxlen=20)       # Stores timestamps
GRAVITY = 9.81                     # m/s²
PREDICTION_STEPS = 50               # Number of prediction steps
PREDICTION_TIME = 0.1              # Seconds to predict ahead

class TrajectoryPredictor:
    def __init__(self):
        self.positions = deque(maxlen=10)
        self.timestamps = deque(maxlen=10)
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
    
    def update(self, position, timestamp):
        if len(self.positions) > 0:
            dt = max(0.001, timestamp - self.timestamps[-1])
            new_velocity = (position - self.positions[-1]) / dt
            self.acceleration = (new_velocity - self.velocity) / dt
            self.velocity = new_velocity
        self.positions.append(position)
        self.timestamps.append(timestamp)
    
    def predict(self, steps, dt):
        predictions = []
        current_pos = self.positions[-1].copy()
        current_vel = self.velocity.copy()
        
        for _ in range(steps):
            current_vel[2] -= GRAVITY * dt  # Apply gravity in Z-axis
            current_vel += self.acceleration * dt
            current_pos += current_vel * dt
            predictions.append(current_pos.copy())
        
        return predictions

predictor = TrajectoryPredictor()

def detect_bins(color_img, depth_frame, hsv):
    """Detect and annotate bins in the image"""
    rgb_with_bins = color_img.copy()
    
    # Create masks for bins
    blue_bin_mask = cv2.inRange(hsv, BLUE_BIN_LOWER, BLUE_BIN_UPPER)
    red_mask1 = cv2.inRange(hsv, RED_BIN_LOWER1, RED_BIN_UPPER1)
    red_mask2 = cv2.inRange(hsv, RED_BIN_LOWER2, RED_BIN_UPPER2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    # Clean up masks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    blue_bin_cleaned = cv2.morphologyEx(blue_bin_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    blue_bin_cleaned = cv2.morphologyEx(blue_bin_cleaned, cv2.MORPH_CLOSE, kernel, iterations=3)
    red_cleaned = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    red_cleaned = cv2.morphologyEx(red_cleaned, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Process blue bin contours
    blue_bin_contours, _ = cv2.findContours(blue_bin_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in blue_bin_contours:
        area = cv2.contourArea(cnt)
        if area >= BIN_MIN_AREA:
            hull = cv2.convexHull(cnt)
            solidity = float(area)/cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
            
            if solidity > BIN_SOLIDITY_THRESH:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"]/M["m00"])
                    cY = int(M["m01"]/M["m00"])
                    
                    # Get distance and coordinates
                    distance = depth_frame.get_distance(cX, cY)
                    bin_camera = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [cX, cY], distance)
                    
                    # Draw cross at center
                    cross_size = 15
                    cv2.line(rgb_with_bins, (cX-cross_size, cY), (cX+cross_size, cY), (255, 255, 255), 2)
                    cv2.line(rgb_with_bins, (cX, cY-cross_size), (cX, cY+cross_size), (255, 255, 255), 2)
                    
                    # Draw contour
                    cv2.drawContours(rgb_with_bins, [cnt], -1, (255, 0, 0), 2)
                    
                    # Display information
                    info_y_offset = 20
                    cv2.putText(rgb_with_bins, f"Blue Bin", (cX-40, cY-info_y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(rgb_with_bins, f"Dist: {distance:.2f}m", (cX-40, cY+info_y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(rgb_with_bins, f"Cam: [{bin_camera[0]:.2f}, {bin_camera[1]:.2f}, {bin_camera[2]:.2f}]", 
                              (cX-80, cY+info_y_offset*2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Process red bin contours
    red_contours, _ = cv2.findContours(red_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in red_contours:
        area = cv2.contourArea(cnt)
        if area >= BIN_MIN_AREA:
            hull = cv2.convexHull(cnt)
            solidity = float(area)/cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
            
            if solidity > BIN_SOLIDITY_THRESH:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"]/M["m00"])
                    cY = int(M["m01"]/M["m00"])
                    
                    # Get distance and coordinates
                    distance = depth_frame.get_distance(cX, cY)
                    bin_camera = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [cX, cY], distance)
                    
                    # Draw cross at center
                    cross_size = 15
                    cv2.line(rgb_with_bins, (cX-cross_size, cY), (cX+cross_size, cY), (255, 255, 255), 2)
                    cv2.line(rgb_with_bins, (cX, cY-cross_size), (cX, cY+cross_size), (255, 255, 255), 2)
                    
                    # Draw contour
                    cv2.drawContours(rgb_with_bins, [cnt], -1, (0, 0, 255), 2)
                    
                    # Display information
                    info_y_offset = 20
                    cv2.putText(rgb_with_bins, f"Red Bin", (cX-30, cY-info_y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(rgb_with_bins, f"Dist: {distance:.2f}m", (cX-40, cY+info_y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(rgb_with_bins, f"Cam: [{bin_camera[0]:.2f}, {bin_camera[1]:.2f}, {bin_camera[2]:.2f}]", 
                              (cX-80, cY+info_y_offset*2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return rgb_with_bins

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            continue

        current_time = color_frame.get_timestamp() / 1000  # Convert to seconds
        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(depth_frame.get_data())
        hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
        
        # Detect bins with cross markers
        detection_img = detect_bins(color_img, depth_frame, hsv)
        
        # Detect yellow balls only
        yellow_mask = cv2.inRange(hsv, YELLOW_BALL_LOWER, YELLOW_BALL_UPPER)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        clean_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        contours, _ = cv2.findContours(clean_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            if radius > BALL_MIN_RADIUS:
                center = (int(x), int(y))
                distance = depth_frame.get_distance(center[0], center[1])
                ball_camera = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [center[0], center[1]], distance)
                ball_world = rmat @ np.array(ball_camera) + tvec.flatten()
                
                # Update trajectory predictor
                predictor.update(ball_world, current_time)
                
                # Store trajectory
                trajectory.append(center)
                
                # Predict future positions
                predicted_world_positions = predictor.predict(PREDICTION_STEPS, PREDICTION_TIME/PREDICTION_STEPS)
                predicted_pixels = []
                
                for pos in predicted_world_positions:
                    # Convert world position back to pixels
                    camera_point = np.linalg.inv(rmat) @ (pos - tvec.flatten())
                    pixel = rs.rs2_project_point_to_pixel(depth_intrinsics, camera_point)
                    pixel = (int(pixel[0]), int(pixel[1]))
                    predicted_pixels.append(pixel)
                
                # Draw prediction
                for i in range(1, len(predicted_pixels)):
                    alpha = i / len(predicted_pixels)
                    color = (0, int(255 * (1 - alpha)), int(255 * alpha))
                    cv2.line(detection_img, predicted_pixels[i-1], predicted_pixels[i], color, 2)
                
                # Draw current position
                cv2.circle(detection_img, center, int(radius), (0, 255, 255), 2)
                cv2.circle(detection_img, center, 3, (0, 0, 255), -1)
                
                # Display info
                velocity_mag = np.linalg.norm(predictor.velocity)
                cv2.putText(detection_img, f"Yellow Ball {distance:.2f}m", 
                          (center[0] - 70, center[1] - int(radius) - 15),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(detection_img, f"Vel: {velocity_mag:.2f}m/s",
                          (center[0] - 70, center[1] - int(radius) - 35),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Draw trajectory
                if len(trajectory) > 1:
                    for i in range(1, len(trajectory)):
                        cv2.line(detection_img, trajectory[i-1], trajectory[i], (0, 255, 0), 2)

        # Create depth visualization
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
        
        # Display
        combined = np.hstack((detection_img, depth_colormap))
        cv2.imshow("Ball Tracking + Bin Detection", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()