import pyrealsense2 as rs
import numpy as np
import cv2

# Parameters
depth_threshold_meters = 1.5  # Background cut-off (in meters)

# Initialize pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable color and depth streams at 1280x720 resolution
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

# Align depth to color frame
align_to = rs.stream.color
align = rs.align(align_to)

# Start streaming
pipeline.start(config)

try:
    while True:
        # Get aligned frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert to numpy arrays
        depth_img = np.asanyarray(depth_frame.get_data())
        color_img = np.asanyarray(color_frame.get_data())

        # Background removal
        depth_mask = (depth_img > 0) & (depth_img < depth_threshold_meters * 1000)
        depth_mask_3c = np.dstack((depth_mask, depth_mask, depth_mask))
        color_removed_bg = np.where(depth_mask_3c, color_img, 0).astype(np.uint8)

        # Depth colormap for visualization
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_img, alpha=0.03),
            cv2.COLORMAP_JET
        )

        # Resize if needed (optional if both are already 1280x720)
        if depth_colormap.shape != color_removed_bg.shape:
            depth_colormap = cv2.resize(depth_colormap, (color_removed_bg.shape[1], color_removed_bg.shape[0]))

        # Annotate (optional)
        cv2.putText(color_removed_bg, "Color (BG Removed)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(depth_colormap, "Depth Map", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Concatenate views horizontally
        combined = np.hstack((color_removed_bg, depth_colormap))

        # Show
        cv2.imshow("RealSense Color + Depth", combined)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
