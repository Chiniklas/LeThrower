import cv2
import numpy as np

# Load image from file
image_path = "debug_frame.png"  # Change to your image path
# image_path = "checkerboard.png" 
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Could not read image from {image_path}")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Checkerboard settings (inner corners)
pattern_size = (4, 6)  # (columns, rows)

# Detect corners
found, corners = cv2.findChessboardCorners(
    gray, pattern_size,
    flags=cv2.CALIB_CB_ADAPTIVE_THRESH + 
          cv2.CALIB_CB_NORMALIZE_IMAGE
)

if found:
    # Refine corner locations (sub-pixel accuracy)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    
    # Draw and display the corners
    cv2.drawChessboardCorners(img, pattern_size, corners_refined, found)
    
    # Print first corner coordinates
    print(f"First corner position: {corners_refined[0][0]}")
    
    # Show result
    cv2.imshow("Detected Corners", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Checkerboard not detected! Possible issues:")
    print("- Wrong pattern_size (currently set to (5,8) for inner corners)")
    print("- Low contrast between squares")
    print("- Extreme perspective distortion")
    print("- Blurry image or glare on checkerboard")