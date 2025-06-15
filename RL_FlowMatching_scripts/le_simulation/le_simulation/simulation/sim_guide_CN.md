# XLeRobot Playground 安装运行教程
[![en](https://img.shields.io/badge/lang-en-red.svg)](sim_guide.md)
[![中文](https://img.shields.io/badge/lang-中文-green.svg)](sim_guide_CN.md)

这是实现键盘EE控制的完整教程视频，内容和下面的图文教程一样

https://github.com/user-attachments/assets/b8e630bd-1133-4941-acd1-d974f60098ff


## 前提条件

- Ubuntu操作系统
- 基本的终端命令熟悉程度
- [Miniconda](https://docs.anaconda.com/free/miniconda/index.html)（推荐）

## 安装

### 1. 创建Conda环境

如果你已经有`lerobot`conda环境，可以直接使用。否则，创建一个新环境：

```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
```

> **注意**：建议将ManiSkill和lerobot代码放在同一环境中，以便将来进行sim2real部署。

### 2. 安装ManiSkill

按照[官方安装指南](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html)安装ManiSkill。

```bash
# 基本安装
pip install mani-skill

# 下载场景数据集
python -m mani_skill.utils.download_asset "ReplicaCAD"
```

#### 2.1 如果数据集下载出错，请使用：

```bash
git clone https://huggingface.co/datasets/haosulab/ReplicaCAD
```
直接从huggingface下载数据集。然后将文件夹内容放置到

```bash
~/.maniskill/data/scene_datasets/replica_cad_dataset
```

#### 2.2 使用[快速入门指南](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/quickstart.html)和[示例脚本](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/quickstart.html)熟悉ManiSkill。使用以下命令测试是否成功安装了Maniskill：
```bash
python -m mani_skill.examples.demo_random_action -e "ReplicaCAD_SceneManipulation-v1" \
  --render-mode="human" --shader="rt-fast"
```
你应该能看到类似这样的画面：
![image](https://github.com/user-attachments/assets/c7509843-f037-4f37-9b1c-e7cad939037c)


### 3. 附加依赖

```bash
pip install pygame
```

### 4. 替换机器人文件

导航到conda环境中的ManiSkill包文件夹：

```bash
cd ~/miniconda3/envs/lerobot/lib/python3.10/site-packages/mani_skill
```

用XLeRobot文件替换fetch机器人代码和资源：

1. 下载simulation/Maniskill.zip

2. 替换/agents和/assets中的文件：
![image](https://github.com/user-attachments/assets/2675fb26-0302-45ec-a994-d4133ce8c239)
![image](https://github.com/user-attachments/assets/5a85d244-b342-45f5-bfa3-72f1ce11c83a)


3. 将控制代码添加到/examples：
![image](https://github.com/user-attachments/assets/654556ab-473f-44d2-8ff7-107c346882c6)


## 使用方法

> [!NOTE] 
> 文件名可能略有不同，如有需要请检查并更改。
### 关节控制

使用以下命令运行关节控制演示：

```bash
python -m mani_skill.examples.XLeRobot_demo_joint_ctrl -e "ReplicaCAD_SceneManipulation-v1"   --render-mode="human" --shader="rt-fast" -c "pd_joint_delta_pos_dual_arm"
```

### 末端执行器控制

使用以下命令运行末端执行器控制演示：

```bash
python -m mani_skill.examples.XLeRobot_demo_ee_ctrl -e "ReplicaCAD_SceneManipulation-v1"   --render-mode="human" --shader="rt-fast" -c "pd_joint_delta_pos_dual_arm"
```

### 5. 重命名链接

如果遇到以下错误：
![image](https://github.com/user-attachments/assets/c81569a3-5c4f-4ba6-99d9-65d84937e767)
导航到相应的脚本
![image](https://github.com/user-attachments/assets/afda5567-3dfa-4e04-997f-4b5eff0dd1bc)
并将"torso_lift_link"更改为"head_camera_link"
![image](https://github.com/user-attachments/assets/05b52683-5e50-47fc-9cf7-9c021927db18)

### 6. 更多场景

你可以按照[这个指南](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/scenes.html)在XLeRobot中使用各种环境：

- `ReplicaCAD_SceneManipulation-v1`（默认）
- AI2THOR场景
- Robocasa厨房台面场景
- `OpenCabinetDrawer-v1`
![image](https://github.com/user-attachments/assets/767683be-c090-4fd7-9cfe-05fd2b4559c6)


### 7. VR集成

要启用Oculus VR：

1. 根据[官方文档](https://github.com/rail-berkeley/oculus_reader)安装Oculus读取器
2. 用`codes/VR/reader.py`中提供的版本替换reader.py文件：
3. 在配置中更新Oculus读取器文件夹的路径：
![image](https://github.com/user-attachments/assets/f05fae0f-9641-4704-bac7-dea9aa4f0092)
4. 将控制代码放入`examples/`后，使用以下命令运行末端执行器控制演示：

```bash
python -m mani_skill.examples.XLeRobot_demo_VR_ee_ctrl -e "ReplicaCAD_SceneManipulation-v1"   --render-mode="human" --shader="rt-fast" -c "pd_joint_delta_pos_dual_arm"
```

### 8. 自定义硬件集成

要使用不同的硬件版本：

1. 修改`agents/robots/fetch.py`中的URDF路径，以匹配`assets/`中的文件
2. 根据硬件配置，可能需要调整`qpos`的大小
![image](https://github.com/user-attachments/assets/01c5568a-46ac-4d74-95e1-c66994a72d19)

## 故障排除

### 常见错误

- 导航到脚本并更改链接名称
- 导航到脚本并注释掉避障碰撞列表
