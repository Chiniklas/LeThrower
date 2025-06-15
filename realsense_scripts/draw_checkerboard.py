import numpy as np
import cv2

# Configuration
COLS = 5   # Number of inner corners (columns)
ROWS = 8   # Number of inner corners (rows)
SQUARE_SIZE_MM = 30  # Physical size of each square (mm)
DPI = 300            # Print resolution (dots per inch)
MARGIN_MM = 20       # White border around pattern

# Calculate dimensions
MM_TO_INCH = 25.4
square_size_px = int(SQUARE_SIZE_MM * DPI / MM_TO_INCH)
margin_px = int(MARGIN_MM * DPI / MM_TO_INCH)

# Total image size (add 1 square for outer edges)
width_px = (COLS + 1) * square_size_px + 2 * margin_px
height_px = (ROWS + 1) * square_size_px + 2 * margin_px

# Create white background
img = np.ones((height_px, width_px), dtype=np.uint8) * 255

# Draw checkerboard
for i in range(ROWS + 1):
    for j in range(COLS + 1):
        if (i + j) % 2 == 0:
            y1 = margin_px + i * square_size_px
            y2 = y1 + square_size_px
            x1 = margin_px + j * square_size_px
            x2 = x1 + square_size_px
            img[y1:y2, x1:x2] = 0  # Black square

# Add size markers
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, f"{SQUARE_SIZE_MM}mm squares | Print at {DPI}DPI", 
            (margin_px, margin_px//2), font, 0.5, 0, 1, cv2.LINE_AA)

# Save and display
cv2.imwrite("calibration_pattern_5x8_30mm.png", img)
cv2.imshow("Checkerboard Pattern", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Generated: calibration_pattern_5x8_30mm.png")
print(f"Physical size: {(COLS+1)*SQUARE_SIZE_MM/10:.1f}cm x {(ROWS+1)*SQUARE_SIZE_MM/10:.1f}cm")