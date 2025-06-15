# XLeRobot MuJoCo Simulation Controller

A Python-Mujoco keyboard controller for the XLeRobot 

## Overview

This controller provides keyboard controller for mobile chassis movement for omni-wheels and dual arm joint motion

## Installation

1. **Clone or navigate to the XLeRobot repository:**
   ```bash
   cd /path/to/XLeRobot/simulation/mujoco
   ```

2. **Install dependencies:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

   Key dependencies include:
   - `mujoco==3.3.0` - Physics simulation engine
   - `mujoco-python-viewer==0.1.4` - 3D visualization
   - `glfw` - Keyboard input handling

## Usage

### Basic Execution

```bash
python xlerobot_mujoco.py
```

The simulation will start with:
- A 3D viewer window showing the robot
- Real-time control interface
- On-screen control instructions

### Control Scheme

#### 🚗 Chassis Movement (Omnidirectional)
| Action | Key | Description |
|--------|-----|-------------|
| Forward | `Home` | Move in +X direction |
| Backward | `End` | Move in -X direction |
| Left | `Delete` | Move in +Y direction |
| Right | `Page Down` | Move in -Y direction |
| Rotate CCW | `Insert` | Rotate counter-clockwise (+Z) |
| Rotate CW | `Page Up` | Rotate clockwise (-Z) |

#### 🦾 Left Arm Control
| Joint | Positive | Negative | Description |
|-------|----------|----------|-------------|
| Joint 1 | `Q` | `A` | Shoulder rotation |
| Joint 2 | `W` | `S` | Shoulder elevation |
| Joint 3 | `E` | `D` | Elbow rotation |

#### 🦾 Right Arm Control
| Joint | Positive | Negative | Description |
|-------|----------|----------|-------------|
| Joint 1 | `U` | `J` | Shoulder rotation |
| Joint 2 | `I` | `K` | Shoulder elevation |
| Joint 3 | `O` | `L` | Elbow rotation |

### Control Behavior

- **Chassis Movement**: Incremental velocity control with smooth acceleration/deceleration
- **Arm Joints**: Direct position control with small incremental steps (0.005 rad)
- **Real-time Feedback**: Live display of commanded vs actual values
- **Smooth Dynamics**: Automatic velocity decay when keys are released

## Architecture

### Class Structure

```python
class XLeRobotController:
    def __init__(mjcf_path)      # Initialize simulation and viewer
    def update_feedback()        # Read sensor data from simulation
    def update_keyboards()       # Process keyboard input via GLFW
    def update_reference()       # Compute control commands
    def update_control()         # Apply commands to actuators
    def render_ui()              # Update 3D visualization and overlays
    def run()                    # Main control loop
    def cleanup()                # Graceful shutdown
```

### Control Flow

1. **Initialization**: Load MJCF model, setup viewer and camera
2. **Main Loop**:
   - Read joint positions and velocities
   - Process keyboard inputs
   - Update control references
   - Apply controls to simulation
   - Step physics simulation
   - Render visualization (at 60 Hz)
3. **Cleanup**: Close viewer on exit