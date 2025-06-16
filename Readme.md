# Robotic Arm Throwing System (LE Thrower)

Project at Lerobot HACKATHON Munich 2025\
In 30 hours build from scratch

Roboflow Team\
ZewenYang\
DianYu\
BingkunHuang \
ChiZhang \
KunTao\
Contact: imdian.yu@mytum.de


https://github.com/user-attachments/assets/c0d1ebe4-1561-49ee-9938-164782f44fb3



A comprehensive reinforcement learning and flow matching framework for training robotic arms to perform long-distance throwing tasks with precise targeting capabilities.

## 🎯 Overview

This system trains a 5-DOF robotic arm to throw objects to specific target positions [100m, 50m] using:
- **Reinforcement Learning (RL)**: Train throwing policies with PPO
- **Flow Matching**: Generate diverse throwing trajectories from noise
- **Evaluation & Analysis**: Assess performance and record trajectories
- **Real Robot Integration**: Export trajectories for physical robot control

### Key Features
- ✅ **Long-distance throwing** (1.5 meters) 
- ✅ **Precise targeting** with right-forward bias
- ✅ **Joint-specific rewards** encouraging Rotation_R usage
- ✅ **Trajectory visualization** and analysis
- ✅ **Flow matching generation** for trajectory diversity
- ✅ **Real robot compatibility** with exported control sequences

## 📁 System Components

### 1. **le_env.py** - Environment & Simulation
The core Genesis-based simulation environment for robotic arm throwing.

**Key Features:**
- 5-DOF robotic arm control (Rotation_R, Pitch_R, Elbow_R, Wrist_Pitch_R, Wrist_Roll_R)
- Physics-based ball throwing simulation
- Comprehensive reward system for long-distance throwing
- Fixed target position [100, 50] meters
- Freeze period for ball stabilization
- Real-time throw detection and landing analysis

**Reward Components:**
- `throw_distance_reward` (5000x): Distance maximization
- `forward_direction_reward` (3000x): Forward accuracy  
- `rotation_r_usage_reward` (2000x): Encourage shoulder rotation
- `target_alignment_reward` (2500x): Target direction alignment
- `velocity_magnitude_reward` (1200x): Throw velocity optimization

### 2. **le_train.py** - RL Training Pipeline
PPO-based reinforcement learning training for throwing policies.

**Training Features:**
- Multi-environment parallel training (16,384 environments)
- Adaptive learning with curriculum-free approach
- Fixed P-gain control (no domain randomization)
- Comprehensive logging and visualization
- Periodic checkpoint saving
- WandB integration support

**Usage:**
```bash
# Basic training
python le_train.py -e so_arm_throwing --max_iterations 700

# Training with visualization  
python le_train.py -e so_arm_throwing --vis --max_iterations 700

# Training with WandB logging
python le_train.py -e so_arm_throwing --wandb --max_iterations 700
```

### 3. **le_eval.py** - Evaluation & Recording
Comprehensive evaluation system for trained policies with trajectory recording.

**Evaluation Features:**
- Policy performance assessment (10+ episodes)
- Joint trajectory visualization (overlapping plots)
- Position target recording (P targets)
- Real robot control sequence export
- Success rate analysis and statistics
- Throw distance and accuracy metrics

**Recording Modes:**
- **Target Mode**: NN output joint angles for robot control
- **State Mode**: Actual joint positions for trajectory analysis

**Usage:**
```bash
# Basic evaluation
python le_eval.py -e so_arm_throwing --ckpt 700

# Record target joint angles for robot control
python le_eval.py -e so_arm_throwing --ckpt 700 --record --record_type target --record_episode 5

# Record actual joint states for analysis
python le_eval.py -e so_arm_throwing --ckpt 700 --record --record_type state --record_episode 5
```

**Outputs:**
- `robot_control_sequence_{type}_ep{N}.json`: Detailed trajectory data
- `robot_control_script_{type}_ep{N}.py`: Executable robot control script
- `joint_trajectories_episode_0.png`: Joint angle visualizations

### 4. **le_fm.py** - Flow Matching Generation
Advanced trajectory generation using flow matching for creating diverse throwing motions.

**Flow Matching Features:**
- Collect successful trajectories from trained RL policies
- UNet-based flow matching architecture
- Joint trajectory generation from noise
- Trajectory normalization and denormalization
- Integration with torchcfm framework

**Pipeline:**
1. **Data Collection**: Extract successful throwing trajectories
2. **Preprocessing**: Normalize joint angles to [-1, 1] range
3. **Training**: Flow matching with reconstruction loss
4. **Generation**: Create new trajectories from random noise
5. **Export**: Save for robot control or further analysis

**Usage:**
```bash
# Full pipeline: collect, train, generate
python le_fm.py --ckpt 700 --collect_episodes 100 --train_epochs 150

# Skip collection, use saved data
python le_fm.py --skip_collection --data_path collected_trajectories.npz

# Generate more diverse trajectories
python le_fm.py --skip_collection --num_generate 20
```

## 🚀 Quick Start

### Prerequisites
```bash
# Install Genesis physics engine
pip install genesis-world

# Install RSL-RL for reinforcement learning
pip install rsl-rl-lib==2.2.4

# Install flow matching dependencies
pip install torchcfm torchdiffeq

# Additional dependencies
pip install matplotlib numpy torch
```

### 1. Training a Throwing Policy
```bash
cd examples/le
python le_train.py -e so_arm_throwing --max_iterations 700
```

### 2. Evaluating Performance
```bash
python le_eval.py -e so_arm_throwing --ckpt 700
```

### 3. Recording Robot Control Sequences
```bash
# Record target angles for robot control
python le_eval.py -e so_arm_throwing --ckpt 700 --record --record_type target --record_episode 0
```

### 4. Generating Diverse Trajectories
```bash
# Train flow matching model and generate trajectories
python le_fm.py --ckpt 700 --collect_episodes 50 --train_epochs 100
```

## 📊 Results & Analysis

### Training Metrics
- **Episode Reward**: Target 1000+ for successful long-distance throws
- **Throw Distance**: 100+ meters achievable with optimized policies
- **Direction Accuracy**: <15° deviation from target direction
- **Success Rate**: 80%+ for well-trained policies

### Trajectory Analysis
The system provides comprehensive trajectory analysis including:
- Individual joint motion patterns
- Throw velocity and timing analysis
- Landing position accuracy
- Joint usage statistics (especially Rotation_R)

### Real Robot Integration
Generated trajectories can be directly deployed on physical robots:
```python
# Example robot control usage
robot.send_action(target_position)  # From exported JSON sequences
```

## 🔧 Configuration

### Environment Configuration
```python
env_cfg = {
    "num_actions": 5,  # 5 joints
    "control_freq": 50,  # 50Hz control
    "episode_length_s": 5.0,  # 5 second episodes
    "kp": 50.0,  # Fixed P gain
    "torque_limit": 35.0,  # Newton-meters
}
```

### Reward Configuration
```python
reward_cfg = {
    "target_throw_position": [100.0, 50.0],  # Target position
    "reward_scales": {
        "throw_distance_reward": 5000.0,
        "rotation_r_usage_reward": 2000.0,
        "target_alignment_reward": 2500.0,
    }
}
```

## 📈 Advanced Usage

### Custom Target Positions
Modify target positions in environment configuration:
```python
command_cfg = {
    "target_position": [120.0, 80.0],  # Custom target
}
```

### Multi-Stage Training
```bash
# Stage 1: Basic throwing (300 iterations)
python le_train.py -e stage1_throwing --max_iterations 300

# Stage 2: Precision training (400 iterations) 
python le_train.py -e stage2_precision --max_iterations 400
```

### Batch Evaluation
```bash
# Evaluate multiple checkpoints
for ckpt in 100 200 300 400 500; do
    python le_eval.py -e so_arm_throwing --ckpt $ckpt
done
```

## 🤖 Real Robot Deployment

### Generated Control Scripts
The system generates ready-to-use robot control scripts:

```python
# Auto-generated robot control script
robot_cfg = SO101FollowerConfig(
    port="/dev/ttyACM0",
    use_degrees=True
)

robot = make_robot_from_config(robot_cfg)
robot.connect()

for target_position in joint_sequence:
    robot.send_action(target_position)
    time.sleep(0.02)  # 50Hz control
```

### Joint Mapping
| Simulation Joint | Real Robot Joint |
|------------------|------------------|
| Rotation_R | shoulder_pan.pos |
| Pitch_R | shoulder_lift.pos |
| Elbow_R | elbow_flex.pos |
| Wrist_Pitch_R | wrist_flex.pos |
| Wrist_Roll_R | wrist_roll.pos |

## 📝 File Structure

```
examples/le/
├── Le_Readme.md              # This documentation
├── le_env.py                  # Environment & simulation
├── le_train.py                # RL training pipeline
├── le_eval.py                 # Evaluation & recording
├── le_fm.py                   # Flow matching generation
├── logs/                      # Training logs & checkpoints
│   └── so_arm_throwing/
│       ├── model_700.pt       # Trained policy
│       ├── cfgs.pkl           # Configuration
│       └── tb/                # TensorBoard logs
├── *.json                     # Exported robot sequences
├── *.npz                      # Trajectory datasets
└── *.png                      # Visualization plots
```

## 🔍 Troubleshooting

### Common Issues

1. **Genesis Import Error**
   ```bash
   pip install genesis-world
   export PYTHONPATH=$PYTHONPATH:/path/to/genesis
   ```

2. **CUDA Memory Issues**
   ```bash
   # Reduce number of environments
   python le_train.py -B 8192  # Instead of 16384
   ```

3. **No Successful Episodes**
   ```bash
   # Lower success threshold
   python le_fm.py --min_success_distance 3.0
   ```

4. **Visualization Issues**
   ```bash
   # Enable GUI backend
   export DISPLAY=:0
   python le_eval.py --visualize
   ```

## 📚 References

- **Genesis Physics**: High-performance physics simulation:https://github.com/Genesis-Embodied-AI/Genesis
- **RSL-RL**: Reinforcement learning framework:https://github.com/leggedrobotics/rsl_rl
- **TorchCFM**: Flow matching implementation:https://github.com/atong01/conditional-flow-matching

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-reward`)
3. Commit changes (`git commit -am 'Add new reward function'`)
4. Push to branch (`git push origin feature/new-reward`)
5. Create Pull Request

## 📄 License

This project is part of the Genesis robotics framework. See the main Genesis repository for licensing information.

---

**Happy Throwing! 🎯🤖**

For questions and support, please refer to the Genesis documentation or open an issue in the repository. 
