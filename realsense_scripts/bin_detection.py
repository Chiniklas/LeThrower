import cv2
import numpy as np
import pyrealsense2 as rs

# Initialize RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Yellow range in HSV
YELLOW_LOWER = (15, 80, 50)
YELLOW_UPPER = (40, 255, 255)

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
        
        # Create yellow mask
        yellow_mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        cleaned = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Create binary threshold (for grayscale)
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Create visualization images
        rgb_with_mask = rgb.copy()
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= MIN_AREA:
                hull = cv2.convexHull(cnt)
                solidity = float(area)/cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
                
                if solidity > SOLIDITY_THRESH:
                    # Draw yellow contour (BGR color (0,255,255) = yellow)
                    cv2.drawContours(rgb_with_mask, [cnt], -1, (0, 255, 255), 2)
                    cv2.drawContours(rgb_with_mask, [hull], -1, (30, 200, 200), 1)
                    
                    # Label
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"]/M["m00"])
                        cY = int(M["m01"]/M["m00"])
                        cv2.putText(rgb_with_mask, f"Yellow Bin", (cX-60, cY-20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(rgb_with_mask, f"Area: {area:.0f}", (cX-40, cY+20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Convert all visualization images to 3-channel BGR
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        cleaned_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        
        # Add labels
        def add_label(img, text):
            cv2.putText(img, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        add_label(rgb_with_mask, "RGB with Yellow Mask")
        add_label(gray_bgr, "Grayscale")
        add_label(binary_bgr, "Binary Threshold")
        add_label(cleaned_bgr, "Cleaned Yellow Mask")
        
        # Create 2x2 grid
        top_row = np.hstack((rgb_with_mask, gray_bgr))
        bottom_row = np.hstack((binary_bgr, cleaned_bgr))
        grid = np.vstack((top_row, bottom_row))
        
        # Display
        cv2.imshow("Yellow Bin Opening Detection (2x2 View)", grid)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()