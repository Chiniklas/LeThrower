import gymnasium as gym
import numpy as np
import sapien
import pygame
import time

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import RecordEpisode


import tyro
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union

@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PushCube-v1"
    """The environment ID of the task you want to simulate"""

    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "none"
    """Observation mode"""

    robot_uids: Annotated[Optional[str], tyro.conf.arg(aliases=["-r"])] = None
    """Robot UID(s) to use. Can be a comma separated list of UIDs or empty string to have no agents. If not given then defaults to the environments default robot"""

    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
    """Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'"""

    reward_mode: Optional[str] = None
    """Reward mode"""

    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    """Number of environments to run."""

    control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = None
    """Control mode"""

    render_mode: str = "rgb_array"
    """Render mode"""

    shader: str = "default"
    """Change shader used for all cameras in the environment for rendering. Default is 'minimal' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""

    record_dir: Optional[str] = None
    """Directory to save recordings"""

    pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
    """If using human render mode, auto pauses the simulation upon loading"""

    quiet: bool = False
    """Disable verbose output."""

    seed: Annotated[Optional[Union[int, List[int]]], tyro.conf.arg(aliases=["-s"])] = None
    """Seed(s) for random actions and simulator. Can be a single integer or a list of integers. Default is None (no seeds)"""

def get_mapped_joints(robot):
    """
    Get the current joint positions from the robot and map them correctly to the target joints.
    
    The mapping is:
    - full_joints[0,2] → current_joints[0,1] (base x position and base rotation)
    - full_joints[3,6,9,11,13] → current_joints[2,3,4,5,6] (first arm joints)
    - full_joints[4,7,10,12,14] → current_joints[7,8,9,10,11] (second arm joints)
    
    Returns:
        np.ndarray: Mapped joint positions with shape matching the target_joints
    """
    if robot is None:
        return np.zeros(16)  # Default size for action
        
    # Get full joint positions
    full_joints = robot.get_qpos()
    
    # Convert tensor to numpy array if needed
    if hasattr(full_joints, 'numpy'):
        full_joints = full_joints.numpy()
    
    # Handle case where it's a 2D tensor/array
    if full_joints.ndim > 1:
        full_joints = full_joints.squeeze()
    
    # Create the mapped joints array with correct size
    mapped_joints = np.zeros(16)
    
    # Map the joints according to the specified mapping
    if len(full_joints) >= 15:
        # Base joints: [0,2] → [0,1]
        mapped_joints[0] = full_joints[0]  # Base X position
        mapped_joints[1] = full_joints[2]  # Base rotation
        
        # First arm: [3,6,9,11,13] → [2,3,4,5,6]
        mapped_joints[2] = full_joints[3]
        mapped_joints[3] = full_joints[6]
        mapped_joints[4] = full_joints[9]
        mapped_joints[5] = full_joints[11]
        mapped_joints[6] = full_joints[13]
        
        # Second arm: [4,7,10,12,14] → [7,8,9,10,11]
        mapped_joints[7] = full_joints[4]
        mapped_joints[8] = full_joints[7]
        mapped_joints[9] = full_joints[10]
        mapped_joints[10] = full_joints[12]
        mapped_joints[11] = full_joints[14]
        mapped_joints[12] = full_joints[15]
        mapped_joints[13] = full_joints[16]
    
    return mapped_joints

def main(args: Args):
    pygame.init()
    
    screen_width, screen_height = 600, 750
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Control Window - Use keys to move")
    font = pygame.font.SysFont(None, 24)
    
    np.set_printoptions(suppress=True, precision=3)
    verbose = not args.quiet
    if isinstance(args.seed, int):
        args.seed = [args.seed]
    if args.seed is not None:
        np.random.seed(args.seed[0])
    parallel_in_single_scene = args.render_mode == "human"
    if args.render_mode == "human" and args.obs_mode in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"]:
        print("Disabling parallel single scene/GUI render as observation mode is a visual one. Change observation mode to state or state_dict to see a parallel env render")
        parallel_in_single_scene = False
    if args.render_mode == "human" and args.num_envs == 1:
        parallel_in_single_scene = False
    env_kwargs = dict(
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        num_envs=args.num_envs,
        sim_backend=args.sim_backend,
        enable_shadow=True,
        parallel_in_single_scene=parallel_in_single_scene,
    )
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = tuple(args.robot_uids.split(","))
        if len(env_kwargs["robot_uids"]) == 1:
            env_kwargs["robot_uids"] = env_kwargs["robot_uids"][0]
    env: BaseEnv = gym.make(
        args.env_id,
        **env_kwargs
    )
    record_dir = args.record_dir
    if record_dir:
        record_dir = record_dir.format(env_id=args.env_id)
        env = RecordEpisode(env, record_dir, info_on_video=False, save_trajectory=False, max_steps_per_video=gym_utils.find_max_episode_steps_value(env))

    if verbose:
        print("Observation space", env.observation_space)
        print("Action space", env.action_space)
        if env.unwrapped.agent is not None:
            print("Control mode", env.unwrapped.control_mode)
        print("Reward mode", env.unwrapped.reward_mode)

    obs, _ = env.reset(seed=args.seed, options=dict(reconfigure=True))
    if args.seed is not None and env.action_space is not None:
            env.action_space.seed(args.seed[0])
    if args.render_mode is not None:
        viewer = env.render()
        if isinstance(viewer, sapien.utils.Viewer):
            viewer.paused = args.pause
        env.render()
    
    action = env.action_space.sample() if env.action_space is not None else None
    action = np.zeros_like(action)
    
    # Initialize target joint positions with zeros
    target_joints = np.zeros_like(action)
    # Define the step size for changing target joints
    joint_step = 0.01
    # Define the gain for the proportional controller as a list for each joint
    p_gain = np.ones_like(action)  # Default all gains to 1.0
    # Specific gains can be adjusted here
    p_gain[0] = 1.0     # Base forward/backward
    p_gain[1] = 0.5     # Base rotation - lower gain for smoother turning
    p_gain[2:7] = 1.0   # First arm joints
    p_gain[7:12] = 1.0  # Second arm joints
    p_gain[12:14] = 0.04  # Gripper joints
    
    # Get initial joint positions if available
    current_joints = np.zeros_like(action)
    robot = None
    
    # Try to get the robot instance for direct access
    if hasattr(env.unwrapped, "agent"):
        robot = env.unwrapped.agent.robot
    elif hasattr(env.unwrapped, "agents") and len(env.unwrapped.agents) > 0:
        robot = env.unwrapped.agents[0]  # Get the first robot if multiple exist
    
    print("robot", robot)
    
    # Get the correctly mapped joints
    current_joints = get_mapped_joints(robot)
    
    # Ensure target_joints is a numpy array with the same shape as current_joints
    target_joints = np.zeros_like(current_joints)
    
    # Add step counter for warmup phase
    step_counter = 0
    warmup_steps = 50
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                env.close()
                return
        
        keys = pygame.key.get_pressed()
        
        # Update target joint positions based on key presses - only after warmup
        if step_counter >= warmup_steps:
            # Base forward/backward - direct control
            if keys[pygame.K_w]:
                action[0] = 0.1  # Forward
            elif keys[pygame.K_s]:
                action[0] = -0.1  # Backward
            else:
                action[0] = 0.0  # Stop forward/backward movement
                
            # Base turning - using target_joints and P control
            if keys[pygame.K_a]:
                target_joints[1] += joint_step*2  # Turn left
            elif keys[pygame.K_d]:
                target_joints[1] -= joint_step*2  # Turn right
            
            # Arm control - using target_joints and P control
            if keys[pygame.K_7]:
                target_joints[2] += joint_step
            if keys[pygame.K_y]:
                target_joints[2] -= joint_step
            if keys[pygame.K_8]:
                target_joints[3] += joint_step
            if keys[pygame.K_u]:
                target_joints[3] -= joint_step
            if keys[pygame.K_9]:
                target_joints[4] += joint_step
            if keys[pygame.K_i]:
                target_joints[4] -= joint_step
            if keys[pygame.K_0]:
                target_joints[5] += joint_step
            if keys[pygame.K_o]:
                target_joints[5] -= joint_step
            if keys[pygame.K_MINUS]:
                target_joints[6] += joint_step
            if keys[pygame.K_p]:
                target_joints[6] -= joint_step
            
            if keys[pygame.K_h]:
                target_joints[7] += joint_step
            if keys[pygame.K_n]:
                target_joints[7] -= joint_step
            if keys[pygame.K_j]:
                target_joints[8] += joint_step
            if keys[pygame.K_m]:
                target_joints[8] -= joint_step
            if keys[pygame.K_k]:
                target_joints[9] += joint_step
            if keys[pygame.K_COMMA]:
                target_joints[9] -= joint_step
            if keys[pygame.K_l]:
                target_joints[10] += joint_step
            if keys[pygame.K_PERIOD]:
                target_joints[10] -= joint_step
            if keys[pygame.K_SEMICOLON]:
                target_joints[11] += joint_step
            if keys[pygame.K_SLASH]:
                target_joints[11] -= joint_step
            
            # Gripper control - toggle between open and closed
            if keys[pygame.K_f]:
                # Toggle first gripper (index 12)
                if target_joints[12] < 0.4:  # If closed or partially closed
                    target_joints[12] = 2.5  # Open
                else:
                    target_joints[12] = 0.1  # Close
                # Add a small delay to prevent multiple toggles
                pygame.time.delay(200)
                
            if keys[pygame.K_g]:
                # Toggle second gripper (index 13)
                if target_joints[13] < 0.4:  # If closed or partially closed
                    target_joints[13] = 2.5  # Open
                else:
                    target_joints[13] = 0.1  # Close
                # Add a small delay to prevent multiple toggles
                pygame.time.delay(200)
        
        # Get current joint positions using our mapping function
        current_joints = get_mapped_joints(robot)
        
        # Simple P controller for arm joints only (not base)
        if step_counter < warmup_steps:
            action = np.zeros_like(action)
        else:
            # Apply P control to turning (index 1) and arm joints (indices 2-11) and grippers (indices 12-13)
            # Base forward/backward (index 0) is already set directly above
            for i in range(1, len(action)):
                action[i] = p_gain[i] * (target_joints[i] - current_joints[i])
        
        # Clip actions to be within reasonable bound
        
        screen.fill((0, 0, 0))
        
        text = font.render("Controls:", True, (255, 255, 255))
        screen.blit(text, (10, 10))
        
        # Add warmup status to display
        if step_counter < warmup_steps:
            warmup_text = font.render(f"WARMUP: {step_counter}/{warmup_steps} steps", True, (255, 0, 0))
            screen.blit(warmup_text, (300, 10))
        
        control_texts = [
            "W/S: joint[0] (+/-)",
            "A/D: joint[1] (+/-)",
            "Y/U: joint[2] (+/-)",
            "8/I: joint[3] (+/-)",
            "9/O: joint[4] (+/-)",
            "0/P: joint[5] (+/-)",
            "-/[: joint[6] (+/-)",
            "H/N: joint[7] (+/-)",
            "J/M: joint[8] (+/-)",
            "K/,: joint[9] (+/-)",
            "L/.: joint[10] (+/-)",
            ";/?: joint[11] (+/-)",
            "R: Reset targets to current"
        ]
        
        col_height = len(control_texts) // 2 + len(control_texts) % 2
        for i, txt in enumerate(control_texts):
            col = 0 if i < col_height else 1
            row = i if i < col_height else i - col_height
            ctrl_text = font.render(txt, True, (255, 255, 255))
            screen.blit(ctrl_text, (10 + col * 200, 40 + row * 25))
        
        # Display full joints (before mapping)
        y_pos = 40 + col_height * 30 + 10
        
        # Get full joint positions
        full_joints = robot.get_qpos() if robot is not None else np.zeros(17)
        
        # Convert tensor to numpy array if needed
        if hasattr(full_joints, 'numpy'):
            full_joints = full_joints.numpy()
        
        # Handle case where it's a 2D tensor/array
        if full_joints.ndim > 1:
            full_joints = full_joints.squeeze()
            
        # Display full joints in two rows
        full_joints_text1 = font.render(
            f"Full Joints (1-8): {np.round(full_joints[:8], 2)}", 
            True, (255, 150, 0)
        )
        screen.blit(full_joints_text1, (10, y_pos))
        y_pos += 25
        
        full_joints_text2 = font.render(
            f"Full Joints (9-17): {np.round(full_joints[8:], 2)}", 
            True, (255, 150, 0)
        )
        screen.blit(full_joints_text2, (10, y_pos))
        y_pos += 30
        
        # Display current joint positions in three logical groups
        # Group 1: Base control [0,1]
        base_joints = current_joints[0:2]
        base_text = font.render(
            f"Base [0,1]: {np.round(base_joints, 2)}", 
            True, (255, 255, 0)
        )
        screen.blit(base_text, (10, y_pos))
        
        # Group 2: First arm [2,3,4,5,6]
        y_pos += 25
        arm1_joints = current_joints[2:7]
        arm1_text = font.render(
            f"Arm 1 [2,3,4,5,6]: {np.round(arm1_joints, 2)}", 
            True, (255, 255, 0)
        )
        screen.blit(arm1_text, (10, y_pos))
        
        # Group 3: Second arm [7,8,9,10,11]
        y_pos += 25
        arm2_joints = current_joints[7:12]
        arm2_text = font.render(
            f"Arm 2 [7,8,9,10,11]: {np.round(arm2_joints, 2)}", 
            True, (255, 255, 0)
        )
        screen.blit(arm2_text, (10, y_pos))
        
        # Display target joint positions in three logical groups
        y_pos += 35
        
        # Group 1: Base control [0,1]
        base_targets = target_joints[0:2]
        base_target_text = font.render(
            f"Base Target [0,1]: {np.round(base_targets, 2)}", 
            True, (0, 255, 0)
        )
        screen.blit(base_target_text, (10, y_pos))
        
        # Group 2: First arm [2,3,4,5,6]
        y_pos += 25
        arm1_targets = target_joints[2:7]
        arm1_target_text = font.render(
            f"Arm 1 Target [2,3,4,5,6]: {np.round(arm1_targets, 2)}", 
            True, (0, 255, 0)
        )
        screen.blit(arm1_target_text, (10, y_pos))
        
        # Group 3: Second arm [7,8,9,10,11]
        y_pos += 25
        arm2_targets = target_joints[7:12]
        arm2_target_text = font.render(
            f"Arm 2 Target [7,8,9,10,11]: {np.round(arm2_targets, 2)}", 
            True, (0, 255, 0)
        )
        screen.blit(arm2_target_text, (10, y_pos))
        
        # Group 4: Grippers [12,13]
        y_pos += 25
        gripper_targets = target_joints[12:14]
        gripper_target_text = font.render(
            f"Grippers Target [12,13]: {np.round(gripper_targets, 2)}", 
            True, (0, 255, 0)
        )
        screen.blit(gripper_target_text, (10, y_pos))
        
        # Display current action values (velocities) in three logical groups
        y_pos += 35
        
        # Group 1: Base control [0,1]
        base_actions = action[0:2]
        base_action_text = font.render(
            f"Base Velocity [0,1]: {np.round(base_actions, 2)}", 
            True, (255, 255, 255)
        )
        screen.blit(base_action_text, (10, y_pos))
        
        # Group 2: First arm [2,3,4,5,6]
        y_pos += 25
        arm1_actions = action[2:7]
        arm1_action_text = font.render(
            f"Arm 1 Velocity [2,3,4,5,6]: {np.round(arm1_actions, 2)}", 
            True, (255, 255, 255)
        )
        screen.blit(arm1_action_text, (10, y_pos))
        
        # Group 3: Second arm [7,8,9,10,11]
        y_pos += 25
        arm2_actions = action[7:12]
        arm2_action_text = font.render(
            f"Arm 2 Velocity [7,8,9,10,11]: {np.round(arm2_actions, 2)}", 
            True, (255, 255, 255)
        )
        screen.blit(arm2_action_text, (10, y_pos))
        
        # Group 4: Grippers [12,13]
        y_pos += 25
        gripper_actions = action[12:14]
        gripper_action_text = font.render(
            f"Grippers Velocity [12,13]: {np.round(gripper_actions, 2)}", 
            True, (255, 255, 255)
        )
        screen.blit(gripper_action_text, (10, y_pos))
        
        pygame.display.flip()
        
        obs, reward, terminated, truncated, info = env.step(action)
        step_counter += 1
        
        if args.render_mode is not None:
            env.render()
        
        time.sleep(0.01)
        
        if args.render_mode is None or args.render_mode != "human":
            if (terminated | truncated).any():
                break
    
    pygame.quit()
    env.close()

    if record_dir:
        print(f"Saving video to {record_dir}")


if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)
