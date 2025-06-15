from lerobot.common.model.kinematics import RobotKinematics
import numpy as np

kin = RobotKinematics(robot_type="so_new_calibration")

q_init = np.array([0.0, -30.0, 45.0, 10.0, 0.0,0.0], dtype=np.float32)

desired_pose = np.eye(4)
desired_pose[:3, 3] = [0.25, 0.0, 0.40]  

# 调用 IK 解算
q_solution = kin.ik(
    current_joint_pos=q_init,
    desired_ee_pose=desired_pose,
    position_only=True,          
    frame="gripper_tip",         
    max_iterations=100,         
    learning_rate=0.5            
)

# 输出结果
print("IK q）:", q_solution)
# take the solution with 0.001
q_solution = np.round(q_solution, 3)  # 
print("IK q）:", q_solution)
print("IK q）: %", np.deg2rad(q_solution))
# 检查 FK(逆解结果) ≈ 目标位姿
pose_fk = kin.forward_kinematics(q_solution, frame="gripper_tip")
print("FK ）:\n", pose_fk)
print("Error:", np.linalg.norm(pose_fk[:3, 3] - desired_pose[:3, 3]))
