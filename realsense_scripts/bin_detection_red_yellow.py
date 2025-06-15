import cv2
import numpy as np
import pyrealsense2 as rs

# Initialize RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Color ranges in HSV
# YELLOW_LOWER = (15, 80, 50)
# YELLOW_UPPER = (40, 255, 255)

# NOW is blue
YELLOW_LOWER = (75, 100, 30)
YELLOW_UPPER = (150, 255, 150)

# Red has two ranges because it wraps around 0 in HSV
RED_LOWER1 = (0, 80, 50)
RED_UPPER1 = (10, 255, 255)
RED_LOWER2 = (160, 80, 50)
RED_UPPER2 = (180, 255, 255)

# Detection parameters
MIN_AREA = 3000               # Minimum contour area
SOLIDITY_THRESH = 0.85        # Shape completeness (1 = perfect)

try:
    while True:
        # Get frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        
        # Process image
        rgb = np.asanyarray(color_frame.get_data())
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        
        # Create color masks
        yellow_mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
        
        # Red requires two masks due to HSV wrap-around
        red_mask1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
        red_mask2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Clean up masks
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        
        # Process yellow mask
        yellow_cleaned = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        yellow_cleaned = cv2.morphologyEx(yellow_cleaned, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Process red mask
        red_cleaned = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        red_cleaned = cv2.morphologyEx(red_cleaned, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Create binary threshold (for grayscale)
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Create visualization images
        rgb_with_contours = rgb.copy()
        
        # Process yellow contours
        yellow_contours, _ = cv2.findContours(yellow_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in yellow_contours:
            area = cv2.contourArea(cnt)
            if area >= MIN_AREA:
                hull = cv2.convexHull(cnt)
                solidity = float(area)/cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
                
                if solidity > SOLIDITY_THRESH:
                    # Draw yellow contour (BGR color (0,255,255) = yellow)
                    cv2.drawContours(rgb_with_contours, [cnt], -1, (0, 255, 255), 2)
                    cv2.drawContours(rgb_with_contours, [hull], -1, (30, 200, 200), 1)
                    
                    # Label
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"]/M["m00"])
                        cY = int(M["m01"]/M["m00"])
                        cv2.putText(rgb_with_contours, f"Yellow Bin", (cX-60, cY-20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(rgb_with_contours, f"Area: {area:.0f}", (cX-40, cY+20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Process red contours
        red_contours, _ = cv2.findContours(red_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in red_contours:
            area = cv2.contourArea(cnt)
            if area >= MIN_AREA:
                hull = cv2.convexHull(cnt)
                solidity = float(area)/cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
                
                if solidity > SOLIDITY_THRESH:
                    # Draw red contour (BGR color (0,0,255) = red)
                    cv2.drawContours(rgb_with_contours, [cnt], -1, (0, 0, 255), 2)
                    cv2.drawContours(rgb_with_contours, [hull], -1, (0, 100, 255), 1)
                    
                    # Label
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"]/M["m00"])
                        cY = int(M["m01"]/M["m00"])
                        cv2.putText(rgb_with_contours, f"Red Bin", (cX-50, cY-20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(rgb_with_contours, f"Area: {area:.0f}", (cX-40, cY+20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Convert all visualization images to 3-channel BGR
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        yellow_cleaned_bgr = cv2.cvtColor(yellow_cleaned, cv2.COLOR_GRAY2BGR)
        red_cleaned_bgr = cv2.cvtColor(red_cleaned, cv2.COLOR_GRAY2BGR)
        
        # Combine masks for display
        combined_masks = cv2.bitwise_or(yellow_cleaned_bgr, red_cleaned_bgr)
        
        # Add labels
        def add_label(img, text):
            cv2.putText(img, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        add_label(rgb_with_contours, "RGB with Color Contours")
        add_label(gray_bgr, "Grayscale")
        add_label(binary_bgr, "Binary Threshold")
        add_label(combined_masks, "Combined Color Masks")
        
        # Create 2x2 grid
        top_row = np.hstack((rgb_with_contours, gray_bgr))
        bottom_row = np.hstack((binary_bgr, combined_masks))
        grid = np.vstack((top_row, bottom_row))
        
        # Display
        cv2.imshow("Color Bin Detection (2x2 View)", grid)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()