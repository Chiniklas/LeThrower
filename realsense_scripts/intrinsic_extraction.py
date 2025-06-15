import pyrealsense2 as rs
import numpy as np
import json

def save_intrinsics(intrinsics, filename="realsense_intrinsics.json"):
    """Save RealSense intrinsics to a JSON file."""
    data = {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "fx": float(intrinsics.fx),  # Convert to native Python float
        "fy": float(intrinsics.fy),
        "ppx": float(intrinsics.ppx),
        "ppy": float(intrinsics.ppy),
        "model": str(intrinsics.model),
        "coeffs": [float(c) for c in intrinsics.coeffs]  # Convert numpy floats
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

# Initialize pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

try:
    profile = pipeline.start(config)
    
    # Get intrinsics
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    
    print("Camera intrinsics:")
    print(f"  Resolution: {intrinsics.width}x{intrinsics.height}")
    print(f"  Focal length (fx, fy): {intrinsics.fx:.2f}, {intrinsics.fy:.2f}")
    print(f"  Principal point (cx, cy): {intrinsics.ppx:.2f}, {intrinsics.ppy:.2f}")
    print(f"  Distortion model: {intrinsics.model}")
    print(f"  Distortion coeffs: {np.array(intrinsics.coeffs).round(5)}")
    
    # Save to file
    save_intrinsics(intrinsics)
    print(f"\nIntrinsics saved to 'realsense_intrinsics.json'")

finally:
    pipeline.stop()