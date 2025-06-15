import cv2
import numpy as np
import pyrealsense2 as rs

# Initialize RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

# Alignment
align_to = rs.stream.color
align = rs.align(align_to)
depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

# HSV Ranges
BLUE_LOWER = (75, 100, 30)
BLUE_UPPER = (150, 255, 150)
YELLOW_LOWER = (20, 100, 100)
YELLOW_UPPER = (40, 255, 255)

try:
    while True:
        # Get frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            continue

        # Prepare images
        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
        
        # Convert to HSV and grayscale (ADDED THIS LINE)
        hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)  # For subpixel refinement

        # Create masks
        blue_mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
        yellow_mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
        combined_mask = cv2.bitwise_or(blue_mask, yellow_mask)

        # Noise reduction
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        clean_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            if 10 < radius < 100:
                # Initial center estimate
                center = (int(x), int(y))
                
                # === PRECISE CENTER REFINEMENT ===
                # Prepare contour points for refinement (must be np.float32)
                contour_pts = contour.astype(np.float32)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
                precise_center = cv2.cornerSubPix(
                    gray,
                    contour_pts,
                    winSize=(5,5),
                    zeroZone=(-1,-1),
                    criteria=criteria
                )
                # Get mean of refined points
                x_precise, y_precise = precise_center.mean(axis=0)[0]
                center_precise = (int(x_precise), int(y_precise))

                # === ROBUST COLOR DETECTION ===
                # Sample 3x3 area around center
                h, w = hsv.shape[:2]
                x_min = max(0, center_precise[0]-1)
                x_max = min(w, center_precise[0]+2)
                y_min = max(0, center_precise[1]-1)
                y_max = min(h, center_precise[1]+2)
                avg_hsv = np.mean(hsv[y_min:y_max, x_min:x_max], axis=(0,1))

                # Classify color
                if (YELLOW_LOWER[0] <= avg_hsv[0] <= YELLOW_UPPER[0] and
                    YELLOW_LOWER[1] <= avg_hsv[1] <= YELLOW_UPPER[1] and
                    YELLOW_LOWER[2] <= avg_hsv[2] <= YELLOW_UPPER[2]):
                    color_name = "YELLOW"
                    circle_color = (0, 255, 255)  # Yellow
                else:
                    color_name = "BLUE"
                    circle_color = (255, 0, 0)    # Blue

                # === DEPTH MEASUREMENT ===
                distance = depth_frame.get_distance(center_precise[0], center_precise[1])
                ball_3d = rs.rs2_deproject_pixel_to_point(
                    depth_intrinsics,
                    [center_precise[0], center_precise[1]],
                    distance
                )

                # === VISUALIZATION ===
                for img in [color_img, depth_colormap]:
                    # Draw refined center
                    cv2.circle(img, center_precise, int(radius), circle_color, 2)
                    cv2.drawMarker(
                        img,
                        center_precise,
                        color=(0, 0, 255),
                        markerType=cv2.MARKER_CROSS,
                        markerSize=20,
                        thickness=2
                    )
                    cv2.putText(
                        img,
                        f"{color_name} {distance:.2f}m",
                        (center_precise[0]-50, center_precise[1]-int(radius)-15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255,255,255),
                        1
                    )

                print(f"{color_name} ball at ({ball_3d[0]:.2f}, {ball_3d[1]:.2f}, {ball_3d[2]:.2f}) m")

        # Display
        combined = np.hstack((color_img, depth_colormap))
        cv2.imshow("Dual-Color Ball Tracking", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()