# Step-by-step Guide for XLeRobot Playground
[![en](https://img.shields.io/badge/lang-en-red.svg)](sim_guide.md)
[![中文](https://img.shields.io/badge/lang-中文-green.svg)](sim_guide_CN.md)

Here's the full video for keyboard EE control

https://github.com/user-attachments/assets/b8e630bd-1133-4941-acd1-d974f60098ff


## Prerequisites

- Ubuntu operating system
- Basic familiarity with terminal commands
- [Miniconda](https://docs.anaconda.com/free/miniconda/index.html) (recommended)

## Installation

### 1. Create a Conda Environment

If you already have a `lerobot` conda environment, you can use that. Otherwise, create a new environment:

```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
```

> **Note**: It's recommended to have ManiSkill and lerobot code in the same environment for future sim2real deployment.

### 2. Install ManiSkill

Follow the [official installation instructions](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html) for ManiSkill.

```bash
# Basic installation
pip install mani-skill

# Download scene dataset
python -m mani_skill.utils.download_asset "ReplicaCAD"
```

#### 2.1 If the dataset downloading goes wrong, use:

```bash
git clone https://huggingface.co/datasets/haosulab/ReplicaCAD
```
to directly download the dataset from huggingface. And then folder contents should go to

```bash
~/.maniskill/data/scene_datasets/replica_cad_dataset
```

#### 2.2 Familiarize yourself with ManiSkill using the [quickstart guide](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/quickstart.html) and [demo scripts](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/quickstart.html). Try this command to test whether you have successfully installed Maniskill:
```bash
python -m mani_skill.examples.demo_random_action -e "ReplicaCAD_SceneManipulation-v1" \
  --render-mode="human" --shader="rt-fast"
```
And you should be looking at something like this:
![image](https://github.com/user-attachments/assets/c7509843-f037-4f37-9b1c-e7cad939037c)


### 3. Additional Dependencies

```bash
pip install pygame
```

### 4. Replace Robot Files

Navigate to the ManiSkill package folder in your conda environment:

```bash
cd ~/miniconda3/envs/lerobot/lib/python3.10/site-packages/mani_skill
```

Replace the fetch robot code and assets with the XLeRobot files:

1. Download simulation/Maniskill.zip

2. Replace the files in /agents and /assets:
![image](https://github.com/user-attachments/assets/2675fb26-0302-45ec-a994-d4133ce8c239)
![image](https://github.com/user-attachments/assets/5a85d244-b342-45f5-bfa3-72f1ce11c83a)


3. Add control code to /examples:
![image](https://github.com/user-attachments/assets/654556ab-473f-44d2-8ff7-107c346882c6)


## Usage

> [!NOTE] 
> The name of the files may be slightly different, check and change if you need.
### Joint Control

Run the joint control demo with:

```bash
python -m mani_skill.examples.XLeRobot_demo_joint_ctrl -e "ReplicaCAD_SceneManipulation-v1"   --render-mode="human" --shader="rt-fast" -c "pd_joint_delta_pos_dual_arm"
```

### End Effector Control

Run the end effector control demo with:

```bash
python -m mani_skill.examples.XLeRobot_demo_ee_ctrl -e "ReplicaCAD_SceneManipulation-v1"   --render-mode="human" --shader="rt-fast" -c "pd_joint_delta_pos_dual_arm"
```

### 5. Rename the links

If you encounter this error:
![image](https://github.com/user-attachments/assets/c81569a3-5c4f-4ba6-99d9-65d84937e767)
Navigate to the corresponding script
![image](https://github.com/user-attachments/assets/afda5567-3dfa-4e04-997f-4b5eff0dd1bc)
and change "torso_lift_link" to "head_camera_link"
![image](https://github.com/user-attachments/assets/05b52683-5e50-47fc-9cf7-9c021927db18)

### 6. More Scenes

You can use various environments with XLeRobot following [this](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/scenes.html):

- `ReplicaCAD_SceneManipulation-v1` (default)
- AI2THOR scenes
- Robocasa Kitchen counter scenes
- `OpenCabinetDrawer-v1`
![image](https://github.com/user-attachments/assets/767683be-c090-4fd7-9cfe-05fd2b4559c6)


### 7. VR Integration

To enable VR with Oculus:

1. Install Oculus reader according to the (official documentation)[https://github.com/rail-berkeley/oculus_reader]
2. Replace the reader.py file with the provided version in `codes/VR/reader.py`:
3. Update the path of the Oculus reader folder in your configuration:
![image](https://github.com/user-attachments/assets/f05fae0f-9641-4704-bac7-dea9aa4f0092)
4. After putting the control code to `examples/`, run the end effector control demo with:

```bash
python -m mani_skill.examples.XLeRobot_demo_VR_ee_ctrl -e "ReplicaCAD_SceneManipulation-v1"   --render-mode="human" --shader="rt-fast" -c "pd_joint_delta_pos_dual_arm"
```

### 8. Custom Hardware Integration

To use different hardware versions:

1. Modify the URDF path in `agents/robots/fetch.py` to follow the files in `assets/`
2. You may need to adjust the size of `qpos` depending on your hardware configuration
![image](https://github.com/user-attachments/assets/01c5568a-46ac-4d74-95e1-c66994a72d19)

## Troubleshooting

### Common Errors

- Navigate to the script and change the link name
- Navigate to the script and comment out the avoid collision list


  
