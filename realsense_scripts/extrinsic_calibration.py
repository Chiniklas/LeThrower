import pyrealsense2 as rs
import cv2
import numpy as np
import json
from datetime import datetime

# Load intrinsic parameters
with open('realsense_intrinsics.json') as f:
    intrinsics = json.load(f)

# Camera setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, intrinsics['width'], intrinsics['height'], rs.format.bgr8, 30)

def save_extrinsics(rvec, tvec, rmat, distance, filename):
    """Save extrinsic parameters to JSON file"""
    extrinsics = {
        'date_calibrated': datetime.now().isoformat(),
        'rotation_vector': rvec.tolist(),  # Convert numpy array to list
        'translation_vector': tvec.tolist(),
        'rotation_matrix': rmat.tolist(),
        'distance_to_target': float(distance),
        'notes': 'Extrinsic parameters from checkerboard calibration'
    }
    
    with open(filename, 'w') as f:
        json.dump(extrinsics, f, indent=4)
    print(f"\nExtrinsic parameters saved to {filename}")

try:
    # Step 1: Start camera and verify connection
    print("Step 1: Starting camera...")
    pipeline.start(config)
    print("Camera started successfully!")

    # Step 2: Capture single frame
    print("\nStep 2: Capturing frame...")
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        print("Error: No frame captured!")
        exit()
    
    img = np.asanyarray(color_frame.get_data())
    cv2.imwrite("1_raw_capture.png", img)
    print("Saved raw image as '1_raw_capture.png' - Please verify image looks correct")
    cv2.imshow("Raw Capture", img)
    cv2.waitKey(2000)  # Show for 2 seconds

    # Step 3: Convert to grayscale
    print("\nStep 3: Converting to grayscale...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("2_grayscale.png", gray)
    print("Saved grayscale as '2_grayscale.png'")
    cv2.imshow("Grayscale", gray)
    cv2.waitKey(2000)

    # Step 4: Apply preprocessing
    print("\nStep 4: Applying preprocessing...")
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    gray = cv2.equalizeHist(gray)  # Improves contrast
    cv2.imwrite("3_preprocessed.png", gray)
    print("Saved preprocessed as '3_preprocessed.png'")
    cv2.imshow("Preprocessed", gray)
    cv2.waitKey(2000)

    # Step 5: Detect checkerboard
    print("\nStep 5: Attempting checkerboard detection...")
    CHECKERBOARD = (4,7)  # Columns x Rows (INNER corners)
    
    # Try different detection methods
    retry_methods = [
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FILTER_QUADS
    ]
    
    corners = None
    for i, method in enumerate(retry_methods):
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, method)
        if ret:
            print(f"Success with method {i+1}")
            break
    
    if ret:
        print("Checkerboard detected! Refining corners...")
        # Refine corner locations
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_subpix = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        
        # Draw and save results
        img_corners = cv2.drawChessboardCorners(img.copy(), CHECKERBOARD, corners_subpix, ret)
        cv2.imwrite("4_corners_detected.png", img_corners)
        print("Saved detected corners as '4_corners_detected.png'")
        cv2.imshow("Detected Corners", img_corners)
        cv2.waitKey(2000)

        # Step 6: Calculate extrinsic parameters
        print("\nStep 6: Calculating extrinsic parameters...")
        
        # Create camera matrix from intrinsics
        camera_matrix = np.array([
            [intrinsics['fx'], 0, intrinsics['ppx']],
            [0, intrinsics['fy'], intrinsics['ppy']],
            [0, 0, 1]
        ])
        dist_coeffs = np.array(intrinsics['coeffs'])
        
        # Prepare 3D points (assuming 30mm squares - CHANGE THIS TO YOUR ACTUAL SIZE)
        square_size = 0.035  # 30mm in meters
        objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1,2) * square_size
        
        # Solve for pose
        success, rvec, tvec = cv2.solvePnP(objp, corners_subpix, camera_matrix, dist_coeffs)
        
        if success:
            print("Calibration successful!")
            print(f"Rotation Vector (radians):\n{rvec}")
            print(f"Translation Vector (meters):\n{tvec}")
            
            # Convert rotation vector to matrix
            rmat, _ = cv2.Rodrigues(rvec)
            print(f"Rotation Matrix:\n{rmat}")
            
            # Calculate distance to checkerboard
            distance = np.linalg.norm(tvec)
            print(f"Distance to checkerboard: {distance:.2f} meters")
            
            # Visualize axes
            img_axes = cv2.drawFrameAxes(img.copy(), camera_matrix, dist_coeffs, rvec, tvec, 0.1)
            cv2.imwrite("5_pose_visualization.png", img_axes)
            print("Saved pose visualization as '5_pose_visualization.png'")
            cv2.imshow("Pose Visualization", img_axes)
            cv2.waitKey(2000)
            
            # Step 7: Save extrinsic parameters
            save_extrinsics(rvec, tvec, rmat, distance, 'realsense_extrinsics.json')
            
        else:
            print("Calibration failed - could not solve PnP")
    else:
        print("ERROR: Checkerboard not detected. Please verify:")
        print("- Physical pattern matches (4,7) inner corners")
        print("- Entire pattern is visible in the image")
        print("- Good lighting conditions")
        print("- Check saved images for troubleshooting")

finally:
    print("\nCleaning up resources...")
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Done!")