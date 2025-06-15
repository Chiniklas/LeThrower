import time
from lerobot.common.robots import make_robot_from_config
from lerobot.common.robots.so101_follower.so101_follower import SO101FollowerConfig
import json
from pathlib import Path

robot_cfg = SO101FollowerConfig(
    port="/dev/ttyACM0",        
    id="follower_arm_1",
    use_degrees=True          
)


robot = make_robot_from_config(robot_cfg)
robot.connect()

try:
 
    target_position = {
        "shoulder_pan.pos": -15.95,
        "shoulder_lift.pos": -2.02,
        "elbow_flex.pos":-71.78,
        "wrist_flex.pos": -0.0,
        "wrist_roll.pos": 0.75,
        "gripper.pos": 0.28, 
    }
    json_path = Path("/home/zewen/Downloads/robot_control_sequence_state_ep1.json")  
    step_delay = 1                         

    with open(json_path, "r") as f:
        data = json.load(f)
    joint_sequence = data["joint_sequence"]
    # offset all the joint 1 plus 180 degrees 
    joint_sequence = [
        {k: v + 8 if k == "shoulder_pan.pos" else v for k, v in joint_cmd.items()}
        for joint_cmd in joint_sequence
    ]

    # offset all the joint 2 minus 180 degrees 
    joint_sequence = [
        {k: v + 90 if k == "shoulder_lift.pos" else v for k, v in joint_cmd.items()}
        for joint_cmd in joint_sequence
    ]
    #offset all the joint 3 plus 90 degrees
    joint_sequence = [
        {k: v-90  if k == "elbow_flex.pos" else v for k, v in joint_cmd.items()}
        for joint_cmd in joint_sequence
    ]    

    # 发送动作
    robot.send_action(joint_sequence[0]) 

    time.sleep(5.0)

finally:
    robot.disconnect()
