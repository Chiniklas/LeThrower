# XLeRobot Playground (0.1.5)
[![en](https://img.shields.io/badge/lang-en-red.svg)](sim.md)
[![中文](https://img.shields.io/badge/lang-中文-green.svg)](sim_CN.md)
> [!NOTE] 
> Built upon [ManiSkill](https://maniskill.readthedocs.io/en/latest/index.html), [RoboCasa](https://github.com/robocasa/robocasa), [ReplicaCAD](https://aihabitat.org/datasets/replica_cad/), and [AI2Thor](https://github.com/allenai/ai2thor). Special thanks to Stone Tao, the main leader on ManiSkill3, for his detailed discussions that helped me get familiar with ManiSkill so quickly.

> [!NOTE] 
> See here for the [Step-by-Step Guide](sim_guide.md)

https://github.com/user-attachments/assets/c9d664e0-4bba-4eec-bfe2-14fd4755e677


## What We Have / What You Can Do in Simulation

- **Keyboard base movement and joint control** 
- **Keyboard base movement and end effector control** with built-in analytical IK function (this is what we used for the main demos)
- **Keyboard base movement and VR-based end effector control** (Oculus Quest required):
  - Choose between VR joystick or pure hand tracking
  - Currently supports position control only
  - Pose control and gripper control based on hand gestures will be updated soon
- **Many research opportunities** to explore: MPC, imitation learning, data collection, RL, VLA, etc.

All of these demos can be run in multiple different scenes as you wish.

## About Teleop Demos in Simulation

💡 **Key Points:**

- All demos are implemented in `demo_ee_control.py` using 3 different scenes in ManiSkill, utilizing my newly developed hybrid joint-ee controller specifically designed for the SO100 arm
- All tasks were completed in one take - I only removed periods with no operation (completely still screen) and manual camera angle adjustments
- All robot actions are controlled by keyboard unless stated otherwise.
- Currently no camera integration yet. The only input used for control is the interactive interface exactly shown in the demo video. Camera integration will be updated in the URDF and corresponding code in XLeRobot 0.2.0
- For convenience of running directly in ManiSkill, I directly replaced the Fetch robot's corresponding URDF model and robot settings, naming XLeRobot as "fetch". In the future, when ManiSkill officially supports XLeRobot, it can be called directly as an independent robot in simulation
  - The current Fetch base only has forward/backward and steering (2 degrees of freedom), and is relatively difficult to modify, so we temporarily use it as is. However, in reality, XLeRobot's omnidirectional wheel base has 3 degrees of freedom (forward/backward, left/right, rotation), making movement more convenient and flexible
- In simulation, unstable grasping and high-frequency vibrations when grasping objects may occur due to collision model parameter settings, which may cause sudden object displacement in videos. This does not occur in reality (though reality currently doesn't have the precise and stable angle control available in simulation)

## About the Hybrid Joint-EE Controller

The controller decouples traditional IK into different components that can be either directly controlled or solved via simple analytical solutions (law of cosines, which we learned in high school).

### Horizontal Plane Control
- Uses **polar coordinates** instead of Cartesian coordinates, allowing direct control by joint1
- **Advantages of polar coordinates over Cartesian coordinates:**
  - More intuitive for human control, simplifying keyboard control and future VR control difficulty
  - Beneficial for task planning due to rotational symmetry - can reduce search space and state space from 3D to 2D in many cases
  - Provides symmetry for AI large model training, only needing to focus on the vertical plane. Building training datasets in polar coordinate form can greatly improve VLA model training efficiency and policy generalization ability
    - SO2 rotational symmetry has been proven superior to traditional methods in many papers, such as Dian Wang's series of work during his PhD on using Equivariant Models to improve RL, VLA ([Equivariant Diffusion Policy](https://equidiff.github.io/)), and grasp optimization algorithms to improve the efficiency and generalization of robot learning

- If you still want to use Cartesian coordinates rather than polar coordinates, just add 1-2 lines of code based on this transformation formula:
  - `ee_pose[0] = sqrt(x^2 + y^2)`, `ee_pose[1] = z`, `target_joint1 = arctan2(x,y)`

### Vertical Plane Control
- Without considering robot orientation control, the vertical plane becomes a 2-joint linkage, which has a unique analytical solution under SO100's angle>0 constraint. Can be solved in one step using the simple law of cosines from high school

### Tip Orientation
- Since it's 5DOF, yaw cannot be changed
- SO100's 5th joint directly corresponds to roll
- SO100's 4th joint directly corresponds to pitch (note: when calculating pitch, need to compensate for the previous vertical plane height, but still has a unique analytical solution)

## Hardware Setup Versions

XLeRobot provides 4 versions of hardware setup:

1. **Basic fixed arm bases (back)** - Arms mounted on the back of the cart to fully utilize storage space
2. **Fixed arm bases (front)** - Arms mounted on the front of the cart for better dual-arm manipulation ability, but cannot utilize cart storage
3. **Moving arm bases on the z-axis (front)** - For potential applications
4. **Moving arm bases on the z-axis (back)** - For potential applications

> **Note:** The 2 fixed base versions will be completed in the next month's release of XLeRobot 0.2.0. Moving base versions are for potential applications with no specific plans yet.



## Limitations

You can see that for the basic fixed base version, there are still many places it cannot reach. 

## Dig Deeper

### Advanced Manipulation Strategies

Exploring the manipulation strategies used in teleoperation and how we can utilize these high-level strategies to fully unleash XLeRobot's potential:

#### Mobile Base Integration
- **Compensating arm limitations**: The mobile base's additional degrees of freedom and driving force can compensate for the SO100 arm's limitations (5DOF, small range, low payload ~1.2kg)
- **Example - Opening refrigerator doors**: Although the arm payload is only about 1.2kg, when the arm is extended and gripping tightly while moving the cart backward, the pulling force limitation becomes the wheel motor torque limit, gripper friction limit, and tire-ground friction limit - all of which are generally higher than the arm's payload limit

#### Non-Prehensile Manipulation
- **Beyond pick and place**: Utilizing non-grasping manipulation can greatly improve efficiency, robustness, and physical limits of XLeRobot. In household manipulation tasks, there's definitely more than just pick and place
- **Understanding the physical world**: The robot needs to fully understand the physical world and predict/reason about possible physical interactions through intuitive physics (world model, which aligns with Yann LeCun's ideas)
- **Optimization assistance**: Before robots develop complete intelligence, we can use cost function/energy function-based optimization/MPC/model-based RL to assist with complex planning. Later, robots can design relevant functions based on their understanding of the physical world

#### Advanced Techniques
- **Collision-based grasping**: In cluttered or jammed environments (tightly packed bottles, densely arranged books on shelves), robots need to use collisions to change the environment to achieve more convenient object distributions for grasping, sometimes also using compliance to guide themselves to target positions
- **Strategy-level error correction**: For example, immediately switching to a different grasping direction when a bottle tips over, or switching to another bottle when the current one is temporarily unreachable

### Research Contributions
- **"Caging in Time"** (accepted by IJRR): Provides methods for robust object manipulation, especially for non-prehensile manipulation
- **Rice RobotPI Lab**: Our lab's main research track is funnel-based manipulation

## Links
- [Dian Wang's Homepage](https://www.dianwang.io/)
- [Equivariant Diffusion Policy](https://equidiff.github.io/)
