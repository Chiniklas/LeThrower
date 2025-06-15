#!/usr/bin/env python3
"""
le_fm.py - Flow Matching for Robotic Arm Throwing Trajectories

This script:
1. Loads a trained RL checkpoint from le_train.py
2. Collects joint angle trajectories from successful throwing episodes
3. Trains a flow matching model to generate trajectories from noise
4. Generates new throwing trajectories for robot control

Usage:
python le_fm.py --ckpt 700 --collect_episodes 100 --train_epochs 150
"""

import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdiffeq
from torch.utils.data import Dataset, DataLoader
from torchcfm.models.unet import UNetModel
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
import genesis as gs
from rsl_rl.runners import OnPolicyRunner
from le_env import SoArmEnv

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories):
        """
        Dataset for joint angle trajectories
        trajectories: numpy array of shape (num_trajectories, num_joints, num_timesteps)
        """
        self.trajectories = trajectories
        print(f"Dataset shape: {trajectories.shape}")
        print(f"Trajectory range: [{trajectories.min():.3f}, {trajectories.max():.3f}]")

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        return torch.tensor(traj, dtype=torch.float32)

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class FCFlattenUpsampler(nn.Module):
    def __init__(self, input_shape=(5, 100), output_shape=(1, 28, 28)):
        """
        Upsample joint trajectories to image-like format for UNet processing
        input_shape: (num_joints, num_timesteps) 
        output_shape: (channels, height, width) for UNet
        """
        super().__init__()
        in_dim = input_shape[0] * input_shape[1]
        out_dim = output_shape[0] * output_shape[1] * output_shape[2]
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 1024),
            Mish(),
            nn.Linear(1024, 1024),
            Mish(),
            nn.Linear(1024, out_dim)
        )
        self.norm = nn.LayerNorm(out_dim)
        self.output_shape = output_shape

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.norm(x)
        x = x.view(-1, *self.output_shape)
        return x

class FCDownsampler(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), output_shape=(5, 100)):
        """
        Downsample image-like format back to joint trajectories
        input_shape: (channels, height, width) from UNet
        output_shape: (num_joints, num_timesteps)
        """
        super().__init__()
        in_dim = input_shape[0] * input_shape[1] * input_shape[2]
        out_dim = output_shape[0] * output_shape[1]
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 1024),
            Mish(),
            nn.Linear(1024, 1024),
            Mish(),
            nn.Linear(1024, out_dim)
        )
        self.output_shape = output_shape

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.view(x.size(0), *self.output_shape)

def load_trained_model(exp_name, ckpt):
    """Load trained RL model and environment"""
    log_dir = f"logs/{exp_name}"
    
    # Load configurations
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
    
    # Create environment for trajectory collection
    env = SoArmEnv(
        num_envs=1,  # Single environment for trajectory collection
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,  # No visualization during data collection
    )
    
    # Create runner and load model
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    
    # Load model checkpoint
    ckpt_name = f"model_{ckpt}.pt" if ckpt >= 0 else "model_final.pt"
    model_path = os.path.join(log_dir, ckpt_name)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    print(f"Loading model: {model_path}")
    runner.load(model_path)
    
    # Get inference policy
    policy = runner.get_inference_policy(device=gs.device)
    
    return env, policy, env_cfg

def collect_trajectories(env, policy, num_episodes, min_success_distance=5.0):
    """Collect joint angle trajectories from successful throwing episodes"""
    print(f"\nCollecting trajectories from {num_episodes} episodes...")
    
    trajectories = []
    episode_rewards = []
    throw_distances = []
    
    successful_episodes = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        
        episode_reward = 0
        episode_joints = []
        
        # Run episode
        with torch.no_grad():
            while True:
                # Get action from policy
                actions = policy(obs)
                
                # Record joint angles (actual positions after action)
                obs, rewards, dones, infos = env.step(actions)
                
                # Record current joint positions (in radians)
                current_joints = env.dof_pos[0].cpu().numpy()
                episode_joints.append(current_joints.copy())
                
                episode_reward += rewards[0].item()
                
                if dones[0]:
                    # Check if episode was successful
                    landing_pos = env.ball_landing_pos[0].cpu().numpy()
                    base_pos = env.base_pos[:2].cpu().numpy()
                    throw_distance = np.linalg.norm(landing_pos - base_pos)
                    
                    # Only keep trajectories from successful throws
                    if throw_distance >= min_success_distance and env.throw_detected[0]:
                        # Convert to numpy array: (num_timesteps, num_joints) -> (num_joints, num_timesteps)
                        trajectory = np.array(episode_joints).T
                        trajectories.append(trajectory)
                        episode_rewards.append(episode_reward)
                        throw_distances.append(throw_distance)
                        successful_episodes += 1
                        
                        if successful_episodes % 10 == 0:
                            print(f"  Collected {successful_episodes} successful episodes, "
                                  f"avg distance: {np.mean(throw_distances[-10:]):.2f}m")
                    
                    break
    
    print(f"\nCollection complete:")
    print(f"  Total episodes: {num_episodes}")
    print(f"  Successful episodes: {successful_episodes}")
    print(f"  Success rate: {successful_episodes/num_episodes*100:.1f}%")
    
    if successful_episodes == 0:
        raise ValueError("No successful episodes collected! Lower min_success_distance or train model longer.")
    
    # Convert to numpy array and normalize trajectories
    trajectories = np.array(trajectories)  # (num_successful, num_joints, num_timesteps)
    
    # Normalize joint angles to [-1, 1] range for better training
    joint_min = trajectories.min()
    joint_max = trajectories.max()
    trajectories_normalized = 2 * (trajectories - joint_min) / (joint_max - joint_min) - 1
    
    print(f"  Trajectory shape: {trajectories.shape}")
    print(f"  Joint range: [{joint_min:.3f}, {joint_max:.3f}] -> [-1, 1]")
    print(f"  Average throw distance: {np.mean(throw_distances):.2f}m")
    
    return trajectories_normalized, (joint_min, joint_max), throw_distances

def train_flow_matching(trajectories, num_epochs=150, batch_size=32, lr=3e-4):
    """Train flow matching model on collected trajectories"""
    print(f"\nTraining flow matching model...")
    print(f"  Trajectories: {trajectories.shape}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    
    # Create dataset and dataloader
    train_dataset = TrajectoryDataset(trajectories)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Define model architecture
    num_joints = trajectories.shape[1]
    num_timesteps = trajectories.shape[2]
    
    upsampler = FCFlattenUpsampler(
        input_shape=(num_joints, num_timesteps), 
        output_shape=(1, 28, 28)
    ).to(device)
    
    downsampler = FCDownsampler(
        input_shape=(1, 28, 28), 
        output_shape=(num_joints, num_timesteps)
    ).to(device)
    
    model = UNetModel(
        dim=(1, 28, 28),
        num_channels=64,
        num_res_blocks=2,
        num_classes=10,
        class_cond=True
    ).to(device)
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(
        list(model.parameters()) +
        list(upsampler.parameters()) +
        list(downsampler.parameters()),
        lr=lr
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Flow matcher
    FM = ConditionalFlowMatcher(sigma=0.0)
    
    # Training loop
    loss_history = []
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        for batch_traj in train_loader:
            optimizer.zero_grad()
            
            x1_traj = batch_traj.to(device)
            dummy_label = torch.zeros((x1_traj.size(0),), dtype=torch.long, device=device)
            x0_traj = torch.randn_like(x1_traj)  # Noise trajectories
            
            # Sample time and get flow
            t, xt_traj, ut_traj = FM.sample_location_and_conditional_flow(x0_traj, x1_traj)
            
            # Forward pass through up-sampler, UNet, down-sampler
            xt_img = upsampler(xt_traj)
            vt_img = model(t, xt_img, dummy_label)
            vt_traj = downsampler(vt_img)
            
            # Flow matching loss
            flow_loss = torch.mean((vt_traj - ut_traj) ** 2)
            
            # Reconstruction loss (ensure up/down sampling preserves trajectories)
            x1_img = upsampler(x1_traj)
            recon_traj = downsampler(x1_img)
            recon_loss = torch.mean((recon_traj - x1_traj) ** 2)
            
            # Combined loss
            loss = flow_loss + 1.0 * recon_loss
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        scheduler.step()
        avg_loss = np.mean(epoch_losses)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    print(f"Training complete! Final loss: {loss_history[-1]:.6f}")
    
    return model, upsampler, downsampler, loss_history

def generate_trajectories(model, upsampler, downsampler, num_samples=5, num_joints=5, num_timesteps=100):
    """Generate new joint trajectories from noise using trained flow matching model"""
    print(f"\nGenerating {num_samples} new trajectories...")
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # Sample random noise trajectories
        noise_trajs = torch.randn((num_samples, num_joints, num_timesteps), device=device)
        
        # Define ODE function for flow matching
        def traj_ode_func(t, x_flat):
            x = x_flat.view(1, num_joints, num_timesteps)
            x_img = upsampler(x)
            dummy_label = torch.zeros((1,), dtype=torch.long, device=device)
            v_img = model(t, x_img, dummy_label)
            v_traj = downsampler(v_img)
            return v_traj.view(-1)
        
        # Generate trajectories by solving ODE
        t_sample = torch.linspace(0, 1, 2, device=device)
        generated_trajs = []
        
        for i in range(num_samples):
            x0 = noise_trajs[i].unsqueeze(0)  # (1, num_joints, num_timesteps)
            
            traj_sample = torchdiffeq.odeint(
                traj_ode_func,
                x0.view(-1),
                t_sample,
                atol=1e-4,
                rtol=1e-4,
                method="dopri5"
            )
            
            final_traj = traj_sample[-1].view(1, num_joints, num_timesteps).cpu().numpy()
            generated_trajs.append(final_traj)
        
        generated_trajs = np.concatenate(generated_trajs, axis=0)
    
    print(f"Generated trajectories shape: {generated_trajs.shape}")
    return generated_trajs

def denormalize_trajectories(trajectories, joint_range):
    """Convert normalized trajectories back to joint angle range"""
    joint_min, joint_max = joint_range
    return (trajectories + 1) / 2 * (joint_max - joint_min) + joint_min

def visualize_trajectories(trajectories, title="Joint Trajectories", joint_names=None):
    """Visualize joint angle trajectories"""
    if joint_names is None:
        joint_names = ["Rotation_R", "Pitch_R", "Elbow_R", "Wrist_Pitch_R", "Wrist_Roll_R"]
    
    num_trajs = min(5, len(trajectories))  # Show at most 5 trajectories
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    plt.figure(figsize=(15, 10))
    
    for joint_idx in range(trajectories.shape[1]):
        plt.subplot(2, 3, joint_idx + 1)
        
        for traj_idx in range(num_trajs):
            plt.plot(trajectories[traj_idx, joint_idx, :], 
                    color=colors[traj_idx], alpha=0.7, linewidth=2,
                    label=f'Traj {traj_idx+1}' if joint_idx == 0 else '')
        
        plt.title(f'{joint_names[joint_idx]}')
        plt.xlabel('Time Step')
        plt.ylabel('Angle (rad)')
        plt.grid(True, alpha=0.3)
        
        if joint_idx == 0:
            plt.legend()
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def save_generated_trajectories(trajectories, joint_range, save_path="generated_trajectories.npz"):
    """Save generated trajectories for robot control"""
    joint_min, joint_max = joint_range
    
    # Denormalize trajectories
    denorm_trajectories = denormalize_trajectories(trajectories, joint_range)
    
    # Save with metadata
    np.savez(save_path,
             trajectories=denorm_trajectories,
             joint_range=(joint_min, joint_max),
             joint_names=["Rotation_R", "Pitch_R", "Elbow_R", "Wrist_Pitch_R", "Wrist_Roll_R"],
             description="Generated joint angle trajectories for robotic arm throwing")
    
    print(f"Saved {len(trajectories)} generated trajectories to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Flow Matching for Robotic Arm Throwing Trajectories")
    parser.add_argument("--exp_name", type=str, default="so_arm_throwing", help="Experiment name")
    parser.add_argument("--ckpt", type=int, default=700, help="Checkpoint number to load")
    parser.add_argument("--collect_episodes", type=int, default=100, help="Number of episodes to collect")
    parser.add_argument("--min_success_distance", type=float, default=5.0, help="Minimum throw distance for success")
    parser.add_argument("--train_epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--num_generate", type=int, default=10, help="Number of trajectories to generate")
    parser.add_argument("--save_path", type=str, default="generated_trajectories.npz", help="Path to save generated trajectories")
    parser.add_argument("--skip_collection", action="store_true", help="Skip trajectory collection (load from file)")
    parser.add_argument("--data_path", type=str, default="collected_trajectories.npz", help="Path to load/save collected data")
    
    args = parser.parse_args()
    
    print("="*80)
    print("Flow Matching for Robotic Arm Throwing Trajectories")
    print("="*80)
    
    # Initialize Genesis
    gs.init(logging_level="warning")
    
    if not args.skip_collection:
        # Step 1: Load trained RL model
        print("\n1. Loading trained RL model...")
        env, policy, env_cfg = load_trained_model(args.exp_name, args.ckpt)
        
        # Step 2: Collect successful trajectories
        print("\n2. Collecting trajectories...")
        trajectories, joint_range, throw_distances = collect_trajectories(
            env, policy, args.collect_episodes, args.min_success_distance
        )
        
        # Save collected data
        np.savez(args.data_path,
                 trajectories=trajectories,
                 joint_range=joint_range,
                 throw_distances=throw_distances)
        print(f"Saved collected data to: {args.data_path}")
        
        # Visualize collected trajectories
        print("\n3. Visualizing collected trajectories...")
        visualize_trajectories(trajectories, "Collected Successful Throwing Trajectories")
        
    else:
        # Load previously collected data
        print(f"\nLoading collected data from: {args.data_path}")
        data = np.load(args.data_path)
        trajectories = data['trajectories']
        joint_range = tuple(data['joint_range'])
        throw_distances = data['throw_distances']
        print(f"Loaded {len(trajectories)} trajectories")
    
    # Step 3: Train flow matching model
    print("\n4. Training flow matching model...")
    model, upsampler, downsampler, loss_history = train_flow_matching(
        trajectories, args.train_epochs, args.batch_size, args.lr
    )
    
    # Visualize training loss
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title("Flow Matching Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()
    
    # Step 4: Generate new trajectories
    print("\n5. Generating new trajectories...")
    generated_trajs = generate_trajectories(
        model, upsampler, downsampler, 
        num_samples=args.num_generate,
        num_joints=trajectories.shape[1],
        num_timesteps=trajectories.shape[2]
    )
    
    # Denormalize for visualization and saving
    generated_trajs_denorm = denormalize_trajectories(generated_trajs, joint_range)
    original_trajs_denorm = denormalize_trajectories(trajectories[:5], joint_range)
    
    # Visualize results
    print("\n6. Visualizing results...")
    visualize_trajectories(original_trajs_denorm, "Original Successful Trajectories")
    visualize_trajectories(generated_trajs_denorm, "Generated Trajectories from Flow Matching")
    
    # Step 5: Save generated trajectories
    print("\n7. Saving generated trajectories...")
    save_generated_trajectories(generated_trajs, joint_range, args.save_path)
    
    print("\n" + "="*80)
    print("Flow matching training complete!")
    print(f"Generated {len(generated_trajs)} new throwing trajectories")
    print(f"Trajectories saved to: {args.save_path}")
    print("These trajectories can be used for robot control or further analysis.")
    print("="*80)

if __name__ == "__main__":
    main() 