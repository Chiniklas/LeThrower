import cv2
import numpy as np

# Load and display the image
img = cv2.imread("debug_frame.png")
if img is None:
    print("Error: File not found or corrupted!")
else:
    print(f"Image loaded. Dimensions: {img.shape} (HxWxC)")
    cv2.imshow("Your Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding
thresh = cv2.adaptiveThreshold(gray, 100, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                              cv2.THRESH_BINARY_INV, 11, 2)

cv2.imshow("Thresholded", thresh)
cv2.waitKey(0)

pattern_size = (8, 5)  # Columns x Rows (inner corners)

# Try multiple detection methods
flags = [
    cv2.CALIB_CB_ADAPTIVE_THRESH,
    cv2.CALIB_CB_NORMALIZE_IMAGE,
    cv2.CALIB_CB_FILTER_QUADS,
    cv2.CALIB_CB_FAST_CHECK
]

for flag in flags:
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, flag)
    print(f"Flag {flag}: {'Found' if ret else 'Not found'}")