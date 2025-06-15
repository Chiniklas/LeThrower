import time
import numpy as np
from lerobot.common.robots import make_robot_from_config
from lerobot.common.robots.so101_follower.so101_follower import SO101FollowerConfig

robot_cfg = SO101FollowerConfig(
    port="/dev/ttyACM0",        
    id="follower_arm_1",
    use_degrees=True            
)


robot = make_robot_from_config(robot_cfg)
robot.connect()

try:

    prev_q = None
    prev_time = None

    while True:
        obs = robot.get_observation()

       
        q = np.array([
            obs["shoulder_pan.pos"],
            obs["shoulder_lift.pos"],
            obs["elbow_flex.pos"],
            obs["wrist_flex.pos"],
            obs["wrist_roll.pos"],
            obs["gripper.pos"],
        ], dtype=np.float32)

        now = time.time()

        if prev_q is not None:
            dt = now - prev_time
            qdot = (q - prev_q) / dt
            print(f"\nq (°): {q}")
            print(f"\q (in rad): {np.deg2rad(q)}")
            print(f"vel (°/s): {qdot}")
        else:
            print(f"\first pos (°): {q}")
            

        prev_q = q
        prev_time = now

        time.sleep(0.05)  # 20Hz

except KeyboardInterrupt:
    print("Disconnect...")
finally:
    robot.disconnect()
