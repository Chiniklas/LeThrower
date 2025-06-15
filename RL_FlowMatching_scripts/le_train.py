import argparse
import os
import pickle
import shutil
import math
import numpy as np
from importlib import metadata
import genesis as gs
from rsl_rl.runners import OnPolicyRunner
import time

from le_env import SoArmEnv

import wandb
from dataclasses import dataclass

@dataclass
class EnvWrapper:
    env_cfg: dict
    obs_cfg: dict
    reward_cfg: dict
    command_cfg: dict

# Global list to track various metrics over iterations
training_stats = {
    'iterations': [],
    'episode_rewards': [],
    'episode_lengths': [],
    'value_loss': [],
    'surrogate_loss': [],
    'action_noise_std': [],
    'distance_rewards': [],
    'throw_height_rewards': [],
    'success_rates': []
}

def print_training_progress(iteration, stats, stage_info=""):
    """Print training progress information"""
    print(f"\n{'='*60}")
    print(f"Training Progress - Iteration {iteration} {stage_info}")
    print(f"{'='*60}")
    print(f"Episode Reward: {stats.get('episode_reward_mean', 0):.3f}")
    print(f"Episode Length: {stats.get('episode_length_mean', 0):.1f}")
    print(f"Value Loss: {stats.get('value_loss', 0):.6f}")
    print(f"Surrogate Loss: {stats.get('surrogate_loss', 0):.6f}")
    print(f"Action Noise: {stats.get('action_noise_std', 0):.3f}")
    
    # Curriculum learning information (if available)
    if 'curriculum_stage' in stats:
        stage_names = ["Forward Push", "Balance Transition", "Precise Aiming"]
        stage = int(stats.get('curriculum_stage', 0))
        progress = stats.get('curriculum_progress', 0.0)
        total_episodes = stats.get('total_episodes', 0)
        stage_episodes = stats.get('stage_episodes', 0)
        
        print(f"\n🎓 Curriculum Learning:")
        print(f"  Current Stage: {stage} ({stage_names[min(stage, 2)]})")
        print(f"  Overall Progress: {progress:.1%}")
        print(f"  Total Episodes: {total_episodes}")
        print(f"  Stage Episodes: {stage_episodes}")
    
    # Reward breakdown
    print(f"\nReward Breakdown:")
    reward_keys = ['distance_to_target', 'balanced_forward_targeting', 'ball_velocity_towards_target', 'progressive_distance',
                   'throw_success', 'early_throw_detection',
                   'avoid_original_position', 'xy_displacement_magnitude', 'force_movement_penalty', 'directional_consistency',
                   'ball_horizontal_distance', 'action_magnitude', 'exploration_bonus', 'throw_height', 'throw_distance',
                   'reasonable_distance_penalty', 'wrist_roll_penalty', 'joint_vel_penalty', 'action_smoothness', 'no_throw_penalty']
    for key in reward_keys:
        rew_key = f'rew_{key}'
        if rew_key in stats:
            print(f"  {key}: {stats[rew_key]:.4f}")
    
    print(f"{'='*60}")

def log_training_stats(iteration, stats, wandb_enabled=False):
    """Log training statistics"""
    # Add to global statistics
    training_stats['iterations'].append(iteration)
    training_stats['episode_rewards'].append(stats.get('episode_reward_mean', 0))
    training_stats['episode_lengths'].append(stats.get('episode_length_mean', 0))
    training_stats['value_loss'].append(stats.get('value_loss', 0))
    training_stats['surrogate_loss'].append(stats.get('surrogate_loss', 0))
    training_stats['action_noise_std'].append(stats.get('action_noise_std', 0))
    training_stats['distance_rewards'].append(stats.get('rew_distance_to_target', 0))
    training_stats['throw_height_rewards'].append(stats.get('rew_throw_height', 0))
    
    # Calculate success rate (distance reward > 0.5 considered success)
    success_rate = 1.0 if stats.get('rew_distance_to_target', 0) > 0.5 else 0.0
    training_stats['success_rates'].append(success_rate)
    
    # Log to wandb
    if wandb_enabled:
        wandb.log({
            "iteration": iteration,
            "episode_reward_mean": stats.get('episode_reward_mean', 0),
            "episode_length_mean": stats.get('episode_length_mean', 0),
            "value_loss": stats.get('value_loss', 0),
            "surrogate_loss": stats.get('surrogate_loss', 0),
            "action_noise_std": stats.get('action_noise_std', 0),
            "rew_distance_to_target": stats.get('rew_distance_to_target', 0),
            "rew_throw_height": stats.get('rew_throw_height', 0),
            "rew_throw_success": stats.get('rew_throw_success', 0),
            "success_rate": success_rate,
        }, step=iteration)

def get_train_cfg(exp_name, max_iterations):
    """PPO training configuration - optimized for exploration and throwing learning"""
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.03,  # Increase entropy coefficient, strongly encourage exploration
            "gamma": 0.97,  # Lower discount factor, focus more on immediate rewards
            "lam": 0.95,
            "learning_rate": 0.0008,  # Increase learning rate, accelerate learning
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [256, 128, 64],
            "critic_hidden_dims": [256, 128, 64],
            "init_noise_std": 1.5,  # Significantly increase initial noise, encourage early exploration
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,  # Log every iteration
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 32,  # Increase steps per environment
        "save_interval": 50,  # More frequent saving
        "empirical_normalization": None,  # Set to None to match other examples
        "seed": 1,
        "logger": "tensorboard",  # Ensure using tensorboard logging
        "tensorboard_subdir": "tb",
    }
    return train_cfg_dict

def get_cfgs():
    """Get all configurations - long-distance forward throwing training"""
    # Environment configuration
    env_cfg = {
        "num_actions": 5,  # 5 joints
        "robot_file": "/home/nvidiapc/dodo/Genesis/genesis/assets/xml/le_simulation/simulation/single_le_box.xml",
        "joint_names": [
            "Rotation_R",
            "Pitch_R", 
            "Elbow_R",
            "Wrist_Pitch_R",
            "Wrist_Roll_R"
        ],
        "default_joint_angles": [0.0, 3.14, 0.0, 0.0, 3.14],  # Maintain initial joint positions
        "kp": 50.0,  # Fixed position gain
        "kd": 5.0,   # Velocity gain
        "torque_limit": 35.0,  # Torque limit
        "episode_length_s": 5.0,  # Episode length
        "control_freq": 50,  # Control frequency 50Hz
        "action_scale": 0.7,  # Action scaling
        "clip_actions": 1.2,  # Action clipping range
        "freeze_duration_s": 0.4,  # Freeze duration
    }
    
    # Observation configuration - long-distance forward throwing (no target angle needed)
    obs_cfg = {
        "num_obs": 5 + 5,  # Joint positions(5) + joint velocities(5), removed target angle
        "obs_scales": {
            "dof_pos": 1.0,      # Joint positions
            "dof_vel": 0.05,     # Joint velocities
        },
    }
    
    # Reward configuration - long-distance forward throwing training (aggressive mode)
    reward_cfg = {
        "target_throw_position": [100.0, 50.0],  # Target throw position [x, y] - 100m forward, 50m right
        "distance_tolerance": 5.0,  # Distance tolerance (meters)
        "direction_tolerance": 0.3,  # Direction tolerance (radians, ~17 degrees)
        "reward_scales": {
            # Main rewards - long-distance throwing (significantly enhanced)
            "throw_distance_reward": 5000.0,  # Throw distance reward (farther is better) - 2.5x enhanced
            "forward_direction_reward": 3000.0,  # Forward direction reward - 2x enhanced
            "throw_success": 800.0,  # Basic throw success reward - enhanced
            "velocity_magnitude_reward": 1200.0,  # Ball initial velocity reward - 50% enhanced
            
            # Special rewards - encourage Rotation_R joint usage to aim at [100, 50]
            "rotation_r_usage_reward": 2000.0,  # Encourage using Rotation_R joint for lateral aiming
            "target_alignment_reward": 2500.0,  # Reward aiming at correct target direction [100, 50]
            
            # Remove constraints, encourage all joint free movement
            # "wrist_roll_lock": 0.0,  # Completely remove wrist roll lock, allow free twisting
            "joint_vel_penalty": 0.005,  # Greatly reduce joint velocity penalty, encourage fast movement
            "action_smoothness": 0.05,   # Reduce action smoothness penalty, allow more aggressive actions
            "energy_penalty": 0.002,   # Reduce energy penalty, encourage powerful throwing
        },
    }
    
    # Command configuration - fixed right-forward throwing target
    command_cfg = {
        "num_commands": 0,  # No commands needed, target is fixed
        "target_position": [100.0, 50.0],  # Fixed target: 100m forward, 50m right (encourage Rotation_R usage)
        "training_mode": "max_distance_forward",  # Training mode identifier
    }
    
    return env_cfg, obs_cfg, reward_cfg, command_cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="so_arm_throwing")
    parser.add_argument("-B", "--num_envs", type=int, default=16384)  # Large number of environments for exploration diversity
    #parser.add_argument("-N", "--num_envs", type=int, default=2)  # Visualize
    parser.add_argument("--max_iterations", type=int, default=700)  # Reduce iterations, rely on many environments for fast learning
    parser.add_argument("--wandb", action="store_true", help="Use wandb logging")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--vis", action="store_true", help="Enable visualization interface", default=False)  # Default disable visualization to improve training speed
    args = parser.parse_args()
    
    # Initialize Genesis
    gs.init(logging_level="info" if args.verbose else "warning")
    
    # Create log directory
    log_dir = f"logs/{args.exp_name}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    # Get configurations
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)
    
    # Save configurations
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )
    
    # Initialize wandb (if enabled)
    if args.wandb:
        wandb.init(
            project="so_arm_throwing",
            name=args.exp_name,
            config={
                "env_cfg": env_cfg,
                "obs_cfg": obs_cfg,
                "reward_cfg": reward_cfg,
                "command_cfg": command_cfg,
                "train_cfg": train_cfg,
                "visualization": args.vis,
            }
        )
    
    print(f"\n{'='*80}")
    print(f"Starting Training - {args.exp_name}")
    print(f"{'='*80}")
    print(f"Number of Environments: {args.num_envs}")
    print(f"Max Iterations: {args.max_iterations}")
    print(f"Control Frequency: {env_cfg['control_freq']} Hz")
    print(f"Episode Length: {env_cfg['episode_length_s']} seconds")
    print(f"Freeze Time: {env_cfg['freeze_duration_s']} seconds (let ball stabilize)")
    print(f"P Gain: {env_cfg['kp']} (fixed value, no longer randomized)")
    print(f"Observation Dimension: {obs_cfg['num_obs']} (target angle + joint states)")
    print(f"Log Directory: {log_dir}")
    print(f"Visualization: {'✅ Enabled' if args.vis else '❌ Disabled'}")
    
    print(f"\n🎯 Optimized Initial Configuration:")
    print(f"  Robot Joint Angles: {[f'{np.degrees(angle):.0f}°' for angle in env_cfg['default_joint_angles']]}")
    print(f"  Ball Initial Position: [-0.3, 0.0, 2.0] (from le_model_test optimization results)")
    print(f"  💡 This configuration verified by le_model_test.py for stability and reasonableness")
    
    # Target configuration information
    print(f"\n🎯 Long-distance Right-forward Throwing Training Configuration:")
    target_pos = command_cfg["target_position"]
    print(f"  Throw Target Position: [{target_pos[0]:.1f}, {target_pos[1]:.1f}] (meters) - Encourage Rotation_R usage")
    print(f"  Training Mode: {command_cfg['training_mode']}")
    print(f"  Direction Tolerance: {reward_cfg['direction_tolerance']:.2f} radians ({np.degrees(reward_cfg['direction_tolerance']):.1f}°)")
    print(f"  Distance Tolerance: {reward_cfg['distance_tolerance']:.1f}m")
    print(f"  Wrist Roll: Completely Free (no constraints)")
    print(f"  Rotation_R Reward Weight: {reward_cfg['reward_scales']['rotation_r_usage_reward']:.0f}")
    print(f"  Target Alignment Reward Weight: {reward_cfg['reward_scales']['target_alignment_reward']:.0f}")
    print(f"{'='*80}")
    
    # Create environment
    env = SoArmEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.vis,  # Pass visualization parameter
    )
    env.cfg = EnvWrapper(env_cfg, obs_cfg, reward_cfg, command_cfg)
    
    # Create trainer
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    
    # Debug information: check runner configuration
    print(f"\nDebug Information:")
    print(f"  Logger Type: {train_cfg.get('logger', 'None')}")
    print(f"  Log Interval: {train_cfg['runner']['log_interval']}")
    print(f"  Save Interval: {train_cfg.get('save_interval', 'None')}")
    print(f"  Runner Class: {runner.__class__.__name__}")
    print(f"  Device: {gs.device}")
    
    # Check if log writer exists
    if hasattr(runner, 'writer'):
        print(f"  TensorBoard Writer: {runner.writer is not None}")
        if runner.writer:
            print(f"  Log Directory: {runner.writer.log_dir}")
    
    print(f"  Environment Observation Dimension: {env.num_obs}")
    print(f"  Environment Action Dimension: {env.num_actions}")
    print(f"  Number of Environments: {env.num_envs}")
    print(f"{'='*80}")
    
    # Long-distance right-forward throwing training mode
    print(f"\nStarting long-distance right-forward throwing training...")
    print(f"Training Mode: Maximize right-forward throwing distance and encourage Rotation_R joint usage")
    print(f"Throw Target: [{target_pos[0]:.0f}, {target_pos[1]:.0f}] meters (100m forward, 50m right)")
    print(f"Key Features:")
    print(f"  ✅ Fixed initial joint positions")
    print(f"  ✅ All joints completely free movement (including wrist twisting)")
    print(f"  ✅ Distance maximization reward (2.5x enhanced)")
    print(f"  ✅ Forward direction reward (2x enhanced)")
    print(f"  ✅ Throw velocity reward (50% enhanced)")
    print(f"  ✅ Rotation_R usage reward (2000x weight)")
    print(f"  ✅ Target alignment reward (2500x weight)")
    print(f"  ✅ Reduced action smoothness constraints, allow aggressive actions")
    print(f"P Gain: {env_cfg['kp']} (fixed value)")
    print(f"Observation Space: joint positions + joint velocities ({obs_cfg['num_obs']} dimensions)")
    print(f"Max Iterations: {args.max_iterations}")
    
    total_iterations = 0
    start_time = time.time()
    
    # Batch training with periodic progress output
    batch_size = 50  # Train 50 iterations per batch
    batches = args.max_iterations // batch_size
    remainder = args.max_iterations % batch_size
    
    print(f"Will be divided into {batches} batches, {batch_size} iterations per batch")
    if remainder > 0:
        print(f"Last batch {remainder} iterations")
    
    for batch in range(batches):
        batch_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Batch {batch+1}/{batches}")
        print(f"Iteration {total_iterations+1}-{total_iterations+batch_size}")
        print(f"{'='*60}")
        
        # Batch training
        runner.learn(
            num_learning_iterations=batch_size, 
            init_at_random_ep_len=(batch == 0)
        )
        
        batch_time = time.time() - batch_start_time
        total_iterations += batch_size
        
        # Get training statistics
        stats = {}
        
        # Try to get latest training statistics from runner
        if hasattr(runner, 'logger') and runner.logger is not None:
            try:
                # Get latest training statistics
                if hasattr(runner.logger, 'statistics'):
                    stats = runner.logger.statistics
                elif hasattr(runner, 'writer') and hasattr(runner.writer, 'get_stats'):
                    stats = runner.writer.get_stats()
            except AttributeError:
                pass
        
        # Get basic statistics from environment
        if not stats:
            stats = {
                'episode_reward_mean': float(env.rew_buf.mean().item()),
                'episode_length_mean': float(env.episode_length_buf.float().mean().item()),
            }
            
            # Try to get reward breakdown from episode_sums
            if hasattr(env, 'episode_sums'):
                for name, values in env.episode_sums.items():
                    if values.numel() > 0:
                        stats[f'rew_{name}'] = float(values.mean().item())
        
        # Log statistics
        log_training_stats(total_iterations, stats, args.wandb)
        
        # Print progress
        elapsed_time = time.time() - start_time
        remaining_batches = batches - batch - 1 + (1 if remainder > 0 else 0)
        eta = (elapsed_time / (batch + 1)) * remaining_batches
        
        progress_info = f"(Progress: {total_iterations}/{args.max_iterations}, Time: {batch_time:.1f}s, ETA: {eta:.1f}s)"
        print_training_progress(total_iterations, stats, progress_info)
        
        # Periodic checkpoint saving
        if (batch + 1) % 10 == 0:  # Save every 10 batches
            checkpoint_path = os.path.join(log_dir, f"model_iter_{total_iterations}.pt")
            runner.save(checkpoint_path)
            print(f"Checkpoint saved to: {checkpoint_path}")
    
    # Train remaining iterations
    if remainder > 0:
        print(f"\n{'='*60}")
        print(f"Last batch: Iteration {total_iterations+1}-{total_iterations+remainder}")
        print(f"{'='*60}")
        
        runner.learn(
            num_learning_iterations=remainder, 
            init_at_random_ep_len=False
        )
        total_iterations += remainder
        
        # Final progress report
        stats = {
            'episode_reward_mean': float(env.rew_buf.mean().item()),
            'episode_length_mean': float(env.episode_length_buf.float().mean().item()),
        }
        if hasattr(env, 'episode_sums'):
            for name, values in env.episode_sums.items():
                if values.numel() > 0:
                    stats[f'rew_{name}'] = float(values.mean().item())
        
        log_training_stats(total_iterations, stats, args.wandb)
        print_training_progress(total_iterations, stats, "(Final)")
    
    # Save final model
    final_model_path = os.path.join(log_dir, "model_final.pt")
    runner.save(final_model_path)
    
    # Training completion summary
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"{'='*80}")
    print(f"Total Training Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Total Iterations: {total_iterations}")
    print(f"Log File Location: {log_dir}")
    print(f"Final Model Saved to: {final_model_path}")
    
    # Print final statistics
    if training_stats['episode_rewards']:
        final_reward = training_stats['episode_rewards'][-1]
        avg_reward = np.mean(training_stats['episode_rewards'][-100:])  # Average of last 100
        success_rate = np.mean(training_stats['success_rates'][-100:])  # Success rate of last 100
        
        print(f"\nFinal Statistics:")
        print(f"  Final Episode Reward: {final_reward:.3f}")
        print(f"  Recent 100 Average Reward: {avg_reward:.3f}")
        print(f"  Recent 100 Success Rate: {success_rate:.2%}")
    
    print(f"{'='*80}")
    
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()