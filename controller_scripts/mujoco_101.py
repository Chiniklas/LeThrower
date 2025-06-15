import json
import time
import numpy as np
import mujoco
import mujoco.viewer


model = mujoco.MjModel.from_xml_path("/home/zewen/Downloads/SO-ARM100-main/Simulation/SO101/so101_new_calib.xml")
data = mujoco.MjData(model)


with open("/home/zewen/Downloads/robot_control_sequence_state_ep1.json", "r") as f:
    joint_sequence = json.load(f)["joint_sequence"]

joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

with mujoco.viewer.launch_passive(model, data) as viewer:
    print(f" {len(joint_sequence)} ..")

    for i, joint_cmd in enumerate(joint_sequence):
        for j, name in enumerate(joint_names):
            data.qpos[j] = joint_cmd[f"{name}.pos"] * np.pi / 180.0  
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.05)

    print("✅ ")
