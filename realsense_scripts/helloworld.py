import pyrealsense2 as rs
import numpy as np
import cv2

# Configure streams: IR and Depth
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)  # IR left camera

pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        ir_frame = frames.get_infrared_frame(1)  # stream index 1 = left IR

        if not depth_frame or not ir_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        ir_image = np.asanyarray(ir_frame.get_data())

        # Normalize depth for visualization
        depth_colormap = cv2.convertScaleAbs(depth_image, alpha=0.03)
        depth_colormap = cv2.applyColorMap(depth_colormap, cv2.COLORMAP_JET)

        # IR is already 8-bit grayscale, just convert to BGR for side-by-side display
        ir_colormap = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)

        # Stack IR and depth horizontally
        images = np.hstack((ir_colormap, depth_colormap))

        cv2.imshow('IR (left) and Depth', images)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
