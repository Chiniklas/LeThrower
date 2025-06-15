import torch
import math
import genesis as gs
import numpy as np
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

class SoArmEnv:
    def __init__(self,
                 num_envs,
                 env_cfg,
                 obs_cfg,
                 reward_cfg,
                 command_cfg,
                 show_viewer=False):
        # Basic configuration
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device
        
        # Control frequency and time step
        self.control_freq = env_cfg.get("control_freq", 50)  # Hz
        self.dt = 1.0 / self.control_freq
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)
        
        # Freeze period configuration - keep robot still at start, wait for ball to stabilize
        self.freeze_duration_s = env_cfg.get("freeze_duration_s", 1.0)  # Default freeze 1 second
        self.freeze_duration_steps = math.ceil(self.freeze_duration_s / self.dt)
        
        # Store configurations
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        
        # Observation and reward scaling
        self.obs_scales = obs_cfg.get("obs_scales", {})
        self.reward_scales = reward_cfg.get("reward_scales", {})
        
        # Initialize Genesis scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=4,
            ),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, -2.0, 2.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )
        
        # Add ground plane
        self.scene.add_entity(gs.morphs.Plane(fixed=True))
        
        # If visualization enabled, can add angle direction indicators (optional)
        self.show_viewer = show_viewer
        # Angle direction training doesn't need center markers
        
        # Robot arm initial position
        self.base_pos = torch.tensor([0.0, 0.0, 0.82], device=self.device)
        
        # Load robot arm MJCF
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file=env_cfg["robot_file"],
                pos=self.base_pos.cpu().numpy(),
            )
        )
        
        self.scene.build(n_envs=num_envs)
        
        # Get joint indices
        self.joint_names = env_cfg["joint_names"]
        self.motors_dof_idx = [
            self.robot.get_joint(name).dof_idx_local
            for name in self.joint_names
        ]
        
        # Get ball's free joint (for resetting ball position)
        self.ball_joint = self.robot.get_joint("ball_free")
        self.ball_joint_dof_indices = self.ball_joint.dof_idx_local
        
        # Get ball and end-effector links
        self.ball_link = self.robot.get_link("ball")
        self.ee_link = self.robot.get_link("Wrist_Pitch_Roll_2")
        
        # Set PD gains
        kp = env_cfg["kp"] * torch.ones(self.num_actions, dtype=torch.float32)
        kd = env_cfg["kd"] * torch.ones(self.num_actions, dtype=torch.float32)
        self.robot.set_dofs_kp(kp, self.motors_dof_idx)
        self.robot.set_dofs_kv(kd, self.motors_dof_idx)
        
        # Fixed P gain (no longer randomized)
        # Removed P gain domain randomization, use fixed values
        
        # Set torque limits
        self.robot.set_dofs_force_range(
            lower=-env_cfg["torque_limit"] * torch.ones(self.num_actions, dtype=torch.float32),
            upper= env_cfg["torque_limit"] * torch.ones(self.num_actions, dtype=torch.float32),
            dofs_idx_local=self.motors_dof_idx,
        )
        
        # Prepare reward functions
        self.reward_functions = {}
        self.episode_sums = {}
        for name, scale in self.reward_scales.items():
            self.reward_scales[name] = scale
            self.reward_functions[name] = getattr(self, f"_reward_{name}")
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device)
        
        # Initialize buffers
        self._init_buffers()
        
        # Long-distance forward throwing training - fixed target
        
        # Throw detection related
        self.throw_detected = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.max_ball_height = torch.zeros((self.num_envs,), device=self.device)
        self.ball_landing_pos = torch.zeros((self.num_envs, 2), device=self.device)
        self.ball_landed = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        
        # Freeze period counter - used to keep robot still at start
        self.freeze_counter = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)
        
        # Remove curriculum learning system, use long-distance forward throwing training
        
    def _init_buffers(self):
        N, A, C = self.num_envs, self.num_actions, self.num_commands
        
        self.obs_buf = torch.zeros((N, self.num_obs), device=self.device)
        self.rew_buf = torch.zeros((N,), device=self.device)
        self.reset_buf = torch.ones((N,), device=self.device, dtype=torch.int32)
        self.episode_length_buf = torch.zeros((N,), device=self.device, dtype=torch.int32)
        
        # Long-distance right-forward throwing: fixed target position [100, 50] (encourage Rotation_R usage)
        self.target_position = torch.tensor(
            self.command_cfg["target_position"], device=self.device
        ).unsqueeze(0).expand(N, -1)  # [N, 2]
        
        # No longer need commands buffer (target is fixed)
        if C > 0:
            self.commands = torch.zeros((N, C), device=self.device)
        else:
            self.commands = None
        
        # Action and state buffers
        self.actions = torch.zeros((N, A), device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        
        # Ball state
        self.ball_pos = torch.zeros((N, 3), device=self.device)
        self.ball_vel = torch.zeros((N, 3), device=self.device)
        self.ball_init_pos = torch.tensor([-0.3, 0.0, 2.0], device=self.device)
        
        # Throw detection related
        self.throw_velocity_recorded = torch.zeros((N, 3), device=self.device)  # Record ball velocity at throw
        self.throw_velocity_magnitude = torch.zeros((N,), device=self.device)
        
        # Default joint angles
        self.default_dof_pos = torch.tensor(
            self.env_cfg["default_joint_angles"],
            device=self.device
        )
        
        self.extras = {"episode": {}}
    
    def _resample_commands(self, env_ids):
        """Long-distance right-forward throwing: target is fixed, no resampling needed"""
        # Target position fixed at [100, 50], no resampling needed
        # Target position already initialized in _init_buffers
        pass
    
    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
            
        # 1. Immediately reset robot arm joint positions and velocities to default values (reference dodo_env.py)
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[env_ids],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=env_ids
        )
        
        # 2. Reset robot base position and orientation (ensure entire robot system at correct position)
        base_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).reshape(1, 4)
        
        self.robot.set_pos(
            pos=self.base_pos.unsqueeze(0).expand(len(env_ids), 3),
            zero_velocity=True,
            envs_idx=env_ids
        )
        self.robot.set_quat(
            quat=base_quat.expand(len(env_ids), 4),
            zero_velocity=True,
            envs_idx=env_ids
        )
        
        # 3. Reset ball position (free joint needs explicit reset)
        ball_dof_count = len(self.ball_joint_dof_indices)
        if ball_dof_count == 6:
            # 6DOF: [x, y, z, rx, ry, rz] (axis-angle representation)
            ball_reset_state = torch.zeros((len(env_ids), 6), device=self.device)
            ball_reset_state[:, :3] = self.ball_init_pos  # Position
            # Rotation is zero (axis-angle representation)
        elif ball_dof_count == 7:
            # 7DOF: [x, y, z, qw, qx, qy, qz] (quaternion representation)
            ball_reset_state = torch.zeros((len(env_ids), 7), device=self.device)
            ball_reset_state[:, :3] = self.ball_init_pos  # Position
            ball_reset_state[:, 3] = 1.0  # qw = 1 (unit quaternion)
            # qx, qy, qz = 0 (already zero)
        else:
            # Other cases, only reset position DOF
            print(f"⚠️  Ball joint DOF count: {ball_dof_count}, using simplified reset")
            ball_reset_state = self.ball_init_pos.unsqueeze(0).repeat(len(env_ids), 1)
        
        try:
            self.robot.set_dofs_position(
                position=ball_reset_state,
                dofs_idx_local=self.ball_joint_dof_indices,
                zero_velocity=True,
                envs_idx=env_ids
            )
        except Exception as e:
            print(f"⚠️  Ball reset failed: {e}")
        
        # 4. Zero all DOF velocities (let both robot and ball return to rest state)
        self.robot.zero_all_dofs_velocity(envs_idx=env_ids)
        
        # 5. Update cached variables
        self.ball_pos[env_ids] = self.ball_init_pos
        self.ball_vel[env_ids] = 0.0
        
        # 6. Reset buffers
        self.last_actions[env_ids] = 0
        self.last_dof_vel[env_ids] = 0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = True
        
        # 7. Reset throw detection state
        self.throw_detected[env_ids] = False
        self.max_ball_height[env_ids] = self.ball_init_pos[2]
        self.ball_landing_pos[env_ids] = 0.0
        self.ball_landed[env_ids] = False
        self.throw_velocity_recorded[env_ids] = 0.0
        self.throw_velocity_magnitude[env_ids] = 0.0
        
        # 8. Reset freeze counter (keep robot still at start)
        self.freeze_counter[env_ids] = self.freeze_duration_steps
        
        # 9. P gain is fixed, no randomization needed
        
        # 10. Record episode rewards
        self.extras["episode"] = {}
        for name in self.episode_sums:
            if len(env_ids) > 0:
                avg = torch.mean(self.episode_sums[name][env_ids]).item()
                self.extras["episode"][f"rew_{name}"] = avg
                self.episode_sums[name][env_ids] = 0.0
        
        # 11. Sample new target positions (ensure randomization each reset)
        # Note: This generates new center positions for each reset environment
        self._resample_commands(env_ids)
        
        # 12. Long-distance forward throwing training doesn't need curriculum learning updates
        
        # Display reset information in visualization mode (environment 0 only)
        if self.show_viewer and 0 in env_ids:
            print(f"🔄 Environment 0 reset complete")
            print(f"  Default joint angles: {self.default_dof_pos.cpu().numpy()}")
            print(f"  Freeze time: {self.freeze_duration_s} seconds ({self.freeze_duration_steps} steps)")
            print(f"  Robot base position: {self.base_pos.cpu().numpy()}")
            print(f"  Ball initial position: {self.ball_init_pos.cpu().numpy()}")
            print(f"  Throw target: [100.0, 50.0] (long-distance right-forward throwing, encourage Rotation_R usage)")
            print(f"  P gain: {self.env_cfg['kp']} (fixed value)")
        
    def reset(self):
        self.reset_buf[:] = 1
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        
        # Update observation buffer: only contains joint positions and velocities
        self.obs_buf = torch.cat([
            self.dof_pos * self.obs_scales.get("dof_pos", 1.0),        # Joint positions
            self.dof_vel * self.obs_scales.get("dof_vel", 0.05),       # Joint velocities
        ], dim=-1)
        
        return self.obs_buf, None
        
    def step(self, actions):
        # Action clipping
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        
        # Freeze period logic: ignore input actions during freeze, strictly maintain default posture
        is_frozen = self.freeze_counter > 0
        
        # For frozen environments, force default posture; for non-frozen environments, use policy actions
        target_pos = torch.where(
            is_frozen.unsqueeze(1),  # Broadcast to [num_envs, num_actions]
            self.default_dof_pos.unsqueeze(0).expand(self.num_envs, -1),  # Strictly use default posture when frozen
            self.actions * self.env_cfg["action_scale"] + self.default_dof_pos  # Use policy actions when normal
        )
        
        self.robot.control_dofs_position(target_pos, self.motors_dof_idx)
        
        # Simulation step
        self.scene.step()
        
        # Update freeze counter
        self.freeze_counter = torch.max(self.freeze_counter - 1, torch.zeros_like(self.freeze_counter))
        
        # Display freeze state in visualization mode (environment 0 only)
        if self.show_viewer and self.episode_length_buf[0] % 25 == 0:  # Display every 0.5 seconds
            if self.freeze_counter[0] > 0:
                remaining_time = self.freeze_counter[0].item() * self.dt
                if remaining_time > 0.1:  # Avoid displaying very small numbers
                    print(f"🕐 Environment 0 freezing... Remaining: {remaining_time:.1f} seconds")
        
        # Update state
        self.episode_length_buf += 1
        
        # Get joint states
        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)
        
        # Get ball state
        self.ball_pos[:] = self.ball_link.get_pos()
        self.ball_vel[:] = self.ball_link.get_vel()
        
        # Get end-effector position
        ee_pos = self.ee_link.get_pos()
        
        # Detect throw (ball leaves end-effector) - but not during freeze period
        ball_ee_dist = torch.norm(self.ball_pos - ee_pos, dim=1)
        is_not_frozen = self.freeze_counter <= 0
        just_thrown = (~self.throw_detected) & (ball_ee_dist > 0.15) & is_not_frozen
        
        # Record ball velocity at throw (for velocity reward)
        if torch.any(just_thrown):
            self.throw_velocity_recorded[just_thrown] = self.ball_vel[just_thrown]
            self.throw_velocity_magnitude[just_thrown] = torch.norm(self.ball_vel[just_thrown], dim=1)
        
        self.throw_detected |= just_thrown
        
        # Update maximum height
        self.max_ball_height = torch.max(self.max_ball_height, self.ball_pos[:, 2])
        
        # Detect landing (ball reaches ground plane height z <= 0.0)
        ball_on_ground = (self.ball_pos[:, 2] <= 1.2) & self.throw_detected
        just_landed = ball_on_ground & (~self.ball_landed)
        
        # Quickly capture landing position and print (for reward calculation)
        if torch.any(just_landed):
            landed_envs = just_landed.nonzero(as_tuple=False).flatten()
            for env_id in landed_envs:
                landing_x = self.ball_pos[env_id, 0].item()
                landing_y = self.ball_pos[env_id, 1].item()
                # Calculate throw distance
                landing_dist = torch.norm(self.ball_pos[env_id, :2] - self.base_pos[:2]).item()
                # Calculate forward direction deviation
                direction_vector = self.ball_pos[env_id, :2] - self.base_pos[:2]
                forward_direction = torch.tensor([0.0, 1.0], device=self.device)
                cos_similarity = torch.dot(direction_vector / torch.norm(direction_vector), forward_direction).item()
                direction_angle_deg = np.degrees(np.arccos(np.clip(cos_similarity, -1.0, 1.0)))
                # print(f"🎯 Environment {env_id} - Ball landed: ({landing_x:.3f}, {landing_y:.3f}), Distance: {landing_dist:.2f}m, Forward deviation: {direction_angle_deg:.1f}°")
                
        self.ball_landing_pos[just_landed] = self.ball_pos[just_landed, :2]
        self.ball_landed |= ball_on_ground
        
        # Long-distance forward throwing training: target position is fixed
        
        # Termination conditions
        done = self.episode_length_buf > self.max_episode_length
        # Only end episode due to ball landing when not in freeze period
        is_not_frozen = self.freeze_counter <= 0
        done |= self.ball_landed & is_not_frozen  # End episode when ball lands and not in freeze period
        self.reset_buf = done
        
        # Calculate rewards (only give basic rewards during freeze period)
        self.rew_buf[:] = 0
        is_frozen = self.freeze_counter > 0
        
        for name, fn in self.reward_functions.items():
            scale = self.reward_scales.get(name, 0.0)
            r = fn() * scale
            
            # During freeze period, only keep basic smoothness and stability rewards
            if name in ['throw_distance_reward', 'forward_direction_reward', 'velocity_magnitude_reward', 'throw_success'] and torch.any(is_frozen):
                r = torch.where(is_frozen, torch.zeros_like(r), r)
            
            self.rew_buf += r
            self.episode_sums[name] += r
        
        # Assemble observations: only contains joint positions and velocities
        self.obs_buf = torch.cat([
            self.dof_pos * self.obs_scales.get("dof_pos", 1.0),        # Joint positions
            self.dof_vel * self.obs_scales.get("dof_vel", 0.05),       # Joint velocities
        ], dim=-1)
        
        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.dof_vel
        
        # Ensure extras contains observations key expected by OnPolicyRunner
        self.extras["observations"] = {"critic": self.obs_buf}
        
        # Reset environments that need resetting
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())
        
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def get_observations(self):
        # Ensure extras contains observations key expected by OnPolicyRunner
        self.extras["observations"] = {"critic": self.obs_buf}
        return self.obs_buf, self.extras
    
    def get_privileged_observations(self):
        return None
    
    # Reward functions - long-distance forward throwing
    def _reward_throw_distance_reward(self):
        """Reward: Throw distance - farther is better, focus on maximizing throw distance"""
        reward = torch.zeros(self.num_envs, device=self.device)
        
        # Only calculate distance reward for landed balls
        landed_mask = self.ball_landed
        
        if torch.any(landed_mask):
            # Calculate throw distance (from robot base to landing point)
            landing_pos = self.ball_landing_pos[landed_mask]
            base_pos = self.base_pos[:2]  # Robot base XY position
            landing_dist = torch.norm(landing_pos - base_pos, dim=1)
            
            # Distance reward: linear reward + exponential reward (encourage ultra-long distance)
            # Basic linear reward: 1 point per meter
            linear_reward = landing_dist
            
            # Long-distance exponential reward: encourage throws over 10 meters
            exponential_reward = torch.where(
                landing_dist > 10.0,
                torch.exp((landing_dist - 10.0) / 20.0),  # Exponential growth after 10 meters
                torch.zeros_like(landing_dist)
            )
            
            # Combined reward
            distance_reward = linear_reward + exponential_reward
            
            reward[landed_mask] = distance_reward
        
        return reward
    
    def _reward_forward_direction_reward(self):

        reward = torch.zeros(self.num_envs, device=self.device)
        
        landed_mask = self.ball_landed
        
        if torch.any(landed_mask):
   
            landing_pos = self.ball_landing_pos[landed_mask]
            base_pos = self.base_pos[:2]
            direction_vector = landing_pos - base_pos
            
            forward_direction = torch.tensor([0.0, 1.0], device=self.device)
            
            direction_norm = torch.norm(direction_vector, dim=1)
            
            # 
            valid_throws = direction_norm > 0.1
            direction_reward = torch.zeros_like(direction_norm)
            
            if torch.any(valid_throws):
    
                normalized_direction = direction_vector[valid_throws] / direction_norm[valid_throws].unsqueeze(1)
                
                cos_similarity = torch.sum(normalized_direction * forward_direction, dim=1)
                
                direction_reward[valid_throws] = torch.clamp(cos_similarity, 0.0, 1.0)
            
            reward[landed_mask] = direction_reward
        
        return reward
    
    def _reward_velocity_magnitude_reward(self):

        reward = torch.zeros(self.num_envs, device=self.device)
        
        thrown_mask = self.throw_detected
        
        if torch.any(thrown_mask):
   
            velocity_magnitude = self.throw_velocity_magnitude[thrown_mask]
            
            linear_velocity_reward = velocity_magnitude
            
            exponential_velocity_reward = torch.where(
                velocity_magnitude > 5.0,
                torch.exp((velocity_magnitude - 5.0) / 5.0),
                torch.zeros_like(velocity_magnitude)
            )
            
            velocity_reward = linear_velocity_reward + exponential_velocity_reward
            reward[thrown_mask] = velocity_reward
        
        return reward
    
    def _reward_wrist_roll_lock(self):

        return torch.zeros(self.num_envs, device=self.device)
    
    def _reward_throw_success(self):

        return self.throw_detected.float()
    
    def _reward_joint_vel_penalty(self):

        return -torch.sum(self.dof_vel**2, dim=1)
    
    def _reward_action_smoothness(self):
       
        return -torch.sum((self.actions - self.last_actions)**2, dim=1)
    
    def _reward_energy_penalty(self):

        torques = self.actions * self.env_cfg.get("torque_limit", 35.0)
        energy = torch.sum(torques**2, dim=1)
        return -energy
    
    def _reward_rotation_r_usage_reward(self):

        reward = torch.zeros(self.num_envs, device=self.device)
        
        rotation_r_pos = self.dof_pos[:, 0]  
        rotation_r_vel = torch.abs(self.dof_vel[:, 0])  
        rotation_r_action = torch.abs(self.actions[:, 0]) 
    
        target_rotation_angle = torch.atan2(torch.tensor(50.0, device=self.device), 
                                          torch.tensor(100.0, device=self.device))
        

        angle_error = torch.abs(rotation_r_pos - target_rotation_angle)
        position_reward = torch.exp(-angle_error * 2.0) 
        
        velocity_reward = torch.clamp(rotation_r_vel * 5.0, 0.0, 1.0) 
        
        action_reward = torch.clamp(rotation_r_action * 3.0, 0.0, 1.0)
        
        reward = position_reward + velocity_reward * 0.3 + action_reward * 0.2
        
        return reward
    
    def _reward_target_alignment_reward(self):

        reward = torch.zeros(self.num_envs, device=self.device)
        
        landed_mask = self.ball_landed
        
        if torch.any(landed_mask):

            landing_pos = self.ball_landing_pos[landed_mask]
            base_pos = self.base_pos[:2]
            actual_direction = landing_pos - base_pos
            
            target_direction = torch.tensor([100.0, 50.0], device=self.device)
            
            actual_norm = torch.norm(actual_direction, dim=1)
            target_norm = torch.norm(target_direction)
            
            valid_throws = actual_norm > 0.1
            alignment_reward = torch.zeros_like(actual_norm)
            
            if torch.any(valid_throws):
                normalized_actual = actual_direction[valid_throws] / actual_norm[valid_throws].unsqueeze(1)
                normalized_target = target_direction / target_norm
                
                cos_similarity = torch.sum(normalized_actual * normalized_target, dim=1)
                alignment_reward[valid_throws] = torch.clamp(cos_similarity, 0.0, 1.0)
                

                target_distance = torch.norm(target_direction) 
                actual_distance = actual_norm[valid_throws]
                distance_similarity = torch.exp(-torch.abs(actual_distance - target_distance) / 20.0)
                

                alignment_reward[valid_throws] = alignment_reward[valid_throws] * 0.7 + distance_similarity * 0.3
            
            reward[landed_mask] = alignment_reward
        
        return reward