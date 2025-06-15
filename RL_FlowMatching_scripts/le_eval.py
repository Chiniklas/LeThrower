"""
le_eval.py - 机械臂投掷任务评估和记录脚本

主要功能:
1. 评估训练好的投掷策略
2. 记录目标关节角度序列用于真实机器人控制
3. 可视化投掷结果

使用示例：
# 基本评估（10个episodes）
python le_eval.py -e so_arm_throwing --ckpt 700

# 记录第5个episode的目标关节角度序列（NN输出，用于控制）
python le_eval.py -e so_arm_throwing --ckpt 700 --record --record_type target --record_episode 5

# 记录第5个episode的实际关节位置状态序列（执行状态，用于分析）
python le_eval.py -e so_arm_throwing --ckpt 700 --record --record_type state --record_episode 5

# 指定固定目标位置并记录目标角度
python le_eval.py -e so_arm_throwing --ckpt 700 --target_x 1.0 --target_y 0.5 --record --record_type target --record_episode 0

输出文件：
- robot_control_sequence_{type}_ep{N}.json: 详细的关节角度数据（JSON格式）
- robot_control_script_{type}_ep{N}.py: 可执行的机器人控制/分析脚本

记录类型说明：
- target: 记录NN输出计算的目标关节角度，可直接用于机器人控制
- state: 记录机器人执行后的实际关节位置，用于轨迹分析和性能评估

目标关节角度格式：
{
    "shoulder_pan.pos": 0.0,      # 对应仿真中的Rotation_R关节
    "shoulder_lift.pos": -20.0,   # 对应仿真中的Pitch_R关节
    "elbow_flex.pos": 90.0,       # 对应仿真中的Elbow_R关节  
    "wrist_flex.pos": 0.0,        # 对应仿真中的Wrist_Pitch_R关节
    "wrist_roll.pos": 0.0,        # 对应仿真中的Wrist_Roll_R关节
    "gripper.pos": 0.0,           # 默认gripper控制（0-100）
}
"""
"""
le_eval.py - Robotic Arm Throwing Task Evaluation and Recording Script

Main Features:
1. Evaluate trained throwing strategies
2. Record target joint angle sequences for real robot control
3. Visualize throwing results

Usage Examples:
# Basic evaluation (10 episodes)
python le_eval.py -e so_arm_throwing --ckpt 700

# Record target joint angle sequence for episode 5 (NN output, for control)
python le_eval.py -e so_arm_throwing --ckpt 700 --record --record_type target --record_episode 5

# Record actual joint position state sequence for episode 5 (execution state, for analysis)
python le_eval.py -e so_arm_throwing --ckpt 700 --record --record_type state --record_episode 5

# Specify fixed target position and record target angles
python le_eval.py -e so_arm_throwing --ckpt 700 --target_x 1.0 --target_y 0.5 --record --record_type target --record_episode 0

Output Files:
- robot_control_sequence_{type}_ep{N}.json: Detailed joint angle data (JSON format)
- robot_control_script_{type}_ep{N}.py: Executable robot control/analysis script

Recording Type Description:
- target: Records NN output calculated target joint angles, can be directly used for robot control
- state: Records actual joint positions after robot execution, for trajectory analysis and performance evaluation

Target Joint Angle Format:
{
    "shoulder_pan.pos": 0.0,      # Corresponds to Rotation_R joint in simulation
    "shoulder_lift.pos": -20.0,   # Corresponds to Pitch_R joint in simulation
    "elbow_flex.pos": 90.0,       # Corresponds to Elbow_R joint in simulation
    "wrist_flex.pos": 0.0,        # Corresponds to Wrist_Pitch_R joint in simulation
    "wrist_roll.pos": 0.0,        # Corresponds to Wrist_Roll_R joint in simulation
    "gripper.pos": 0.0,           # Default gripper control (0-100)
}
"""

import argparse
import os
import pickle
import torch
import numpy as np
from importlib import metadata
import json
import time

# 检查rsl_rl版本
try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("请卸载 'rsl_rl' 并安装 'rsl-rl-lib==2.2.4'.") from e

from rsl_rl.runners import OnPolicyRunner
import genesis as gs
from le_env import SoArmEnv

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

def convert_to_json_serializable(obj):
    """
    递归转换对象中的numpy类型为JSON可序列化的Python原生类型
    """
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist() if obj.numel() > 1 else obj.item()
    else:
        return obj

def map_joint_angles_to_robot(sim_angles):
    """
    将仿真关节角度映射到真实机器人关节角度
    
    仿真关节顺序: ["Rotation_R", "Pitch_R", "Elbow_R", "Wrist_Pitch_R", "Wrist_Roll_R"]
    真实机器人关节: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
    """
    # 将弧度转换为角度
    sim_angles_deg = np.degrees(sim_angles)
    
    # 映射关节角度（根据实际机器人配置调整）
    robot_angles = {
        "shoulder_pan.pos": float(sim_angles_deg[0]),    # Rotation_R -> shoulder_pan
        "shoulder_lift.pos": float(sim_angles_deg[1]),   # Pitch_R -> shoulder_lift  
        "elbow_flex.pos": float(sim_angles_deg[2]),      # Elbow_R -> elbow_flex
        "wrist_flex.pos": float(sim_angles_deg[3]),      # Wrist_Pitch_R -> wrist_flex
        "wrist_roll.pos": float(sim_angles_deg[4]),      # Wrist_Roll_R -> wrist_roll
        "gripper.pos": 0.0,  # 默认gripper位置
    }
    
    return robot_angles

def generate_robot_control_script(joint_sequence, output_path, episode_info, record_type):
    """
    生成可以直接控制真实机器人的Python脚本 - 角度方向训练版本
    """
    data_type_desc = {
        "target": "目标关节角度序列 - 神经网络输出计算的控制命令",
        "state": "实际关节位置状态序列 - 机器人执行后的真实关节位置"
    }[record_type]
    
    usage_note = {
        "target": "注意: 此数据为目标位置命令，可直接用于机器人控制",
        "state": "注意: 此数据为实际执行状态，用于轨迹复现时需要转换为位置控制命令"
    }[record_type]
    
    script_content = '''#!/usr/bin/env python3
"""
自动生成的机器人控制脚本 - 角度方向训练
基于仿真episode的{}

Episode信息:
- 目标位置: {}
- 前向偏差: {:.1f}°
- 投掷距离: {:.3f}m
- Episode奖励: {:.3f}
- 总步数: {}
- 控制频率: 50Hz (每步0.02秒)
- 数据类型: {}
- 方向成功: {}
- 距离成功: {}
- 综合成功: {}

{}

关节映射:
- shoulder_pan.pos:  仿真Rotation_R -> 机器人shoulder_pan
- shoulder_lift.pos: 仿真Pitch_R -> 机器人shoulder_lift  
- elbow_flex.pos:    仿真Elbow_R -> 机器人elbow_flex
- wrist_flex.pos:    仿真Wrist_Pitch_R -> 机器人wrist_flex
- wrist_roll.pos:    仿真Wrist_Roll_R -> 机器人wrist_roll
"""

import time
from lerobot.common.robots import make_robot_from_config
from lerobot.common.robots.so101_follower.so101_follower import SO101FollowerConfig

# 创建配置
robot_cfg = SO101FollowerConfig(
    port="/dev/ttyACM0",         # 端口根据实际情况修改
    id="follower_arm_1",
    use_degrees=True             # 使用角度控制（推荐）
)

# 目标关节角度序列（每个元素是一个时间步的目标关节角度）
joint_sequence = {}

def main():
    # 初始化并连接
    robot = make_robot_from_config(robot_cfg)
    robot.connect()
    
    try:
        print("🤖 开始执行角度方向训练记录的关节角度序列")
        print(f"总步数: {{len(joint_sequence)}}")
        print(f"预计执行时间: {{len(joint_sequence) * 0.02:.1f}}秒")
        print("\\n关节角度范围预览:")
        if len(joint_sequence) > 0:
            first_pos = joint_sequence[0]
            print(f"  shoulder_pan:  {{first_pos['shoulder_pan.pos']:.1f}}°")
            print(f"  shoulder_lift: {{first_pos['shoulder_lift.pos']:.1f}}°") 
            print(f"  elbow_flex:    {{first_pos['elbow_flex.pos']:.1f}}°")
            print(f"  wrist_flex:    {{first_pos['wrist_flex.pos']:.1f}}°")
            print(f"  wrist_roll:    {{first_pos['wrist_roll.pos']:.1f}}°")
            print(f"  gripper:       {{first_pos['gripper.pos']:.1f}}")
        
        print("\\n按Enter开始执行...")
        input()
        
        for step, target_position in enumerate(joint_sequence):
            print(f"\\r步数: {{step+1}}/{{len(joint_sequence)}}", end="", flush=True)
            
            # 发送目标位置
            robot.send_action(target_position)
            
            # 等待下一步（仿真频率50Hz = 0.02秒）
            time.sleep(0.02)
        
        print("\\n✅ 角度方向序列执行完成！")
        print("等待5秒让机器人稳定...")
        time.sleep(5.0)
        
    except KeyboardInterrupt:
        print("\\n⚠️  用户中断执行")
    except Exception as e:
        print(f"\\n❌ 执行错误: {{e}}")
    finally:
        robot.disconnect()
        print("🔌 机器人连接已断开")

if __name__ == "__main__":
    main()
'''.format(
        data_type_desc,
        "固定前向目标", "[100, 0]",
        np.degrees(episode_info["direction_error"]),
        episode_info["throw_distance"],
        episode_info["episode_reward"],
        len(joint_sequence), record_type.upper(),
        '✅' if episode_info["direction_success"] else '❌',
        '✅' if episode_info["distance_success"] else '❌',
        '✅' if episode_info["overall_success"] else '❌',
        usage_note, repr(joint_sequence)
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    # 使脚本可执行
    os.chmod(output_path, 0o755)

def visualize_throws(target_positions, landing_positions, circle_centers, target_radius, save_path):
    """可视化投掷结果，支持圆形目标区域"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制机械臂基座位置
    base = Circle((0, 0), 0.1, color='black', label='机械臂基座')
    ax.add_patch(base)
    
    # 绘制目标圆形区域（如果有的话）
    if circle_centers is not None and target_radius > 0:
        for i, center in enumerate(circle_centers):
            circle = Circle((center[0], center[1]), target_radius, 
                          fill=False, color='blue', linestyle='--', alpha=0.7)
            ax.add_patch(circle)
            # 标记圆心
            ax.scatter(center[0], center[1], c='blue', marker='+', s=200, alpha=0.8)
    
    # 绘制目标位置和实际落点
    for i, (target, landing) in enumerate(zip(target_positions, landing_positions)):
        # 目标位置（绿色）
        ax.scatter(target[0], target[1], c='green', marker='x', s=100, alpha=0.8)
        # 实际落点（红色）
        ax.scatter(landing[0], landing[1], c='red', marker='o', s=50, alpha=0.8)
        # 连线
        ax.plot([target[0], landing[0]], [target[1], landing[1]], 
                'gray', alpha=0.3, linewidth=1)
    
    # 添加图例
    legend_handles = [
        mpatches.Patch(color='black', label='机械臂基座'),
        mpatches.Patch(color='green', label='目标位置'),
        mpatches.Patch(color='red', label='实际落点')
    ]
    if circle_centers is not None and target_radius > 0:
        legend_handles.extend([
            mpatches.Patch(color='blue', label='目标圆心'),
            mpatches.Patch(color='blue', label=f'目标区域 (半径{target_radius:.1f}m)', fill=False)
        ])
    
    ax.legend(handles=legend_handles, loc='upper right')
    
    # 设置坐标轴
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-3.0, 3.0)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('机械臂投掷结果可视化')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def visualize_angle_throws(target_angles, landing_positions, base_pos, save_path):
    """可视化角度方向投掷结果"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制机械臂基座位置
    base = Circle((base_pos[0], base_pos[1]), 0.1, color='black', label='机械臂基座')
    ax.add_patch(base)
    
    # 绘制投掷结果
    for i, (target_angle, landing) in enumerate(zip(target_angles, landing_positions)):
        # 计算目标方向上的一个点（用于显示目标方向）
        target_distance = 3.0  # 显示3米距离的目标方向
        target_x = base_pos[0] + target_distance * np.sin(target_angle)
        target_y = base_pos[1] + target_distance * np.cos(target_angle)
        
        # 绘制目标方向线
        ax.plot([base_pos[0], target_x], [base_pos[1], target_y], 
                'g--', alpha=0.6, linewidth=2, label='目标方向' if i == 0 else '')
        
        # 绘制目标方向点
        ax.scatter(target_x, target_y, c='green', marker='x', s=150, alpha=0.8)
        
        # 绘制实际落点
        ax.scatter(landing[0], landing[1], c='red', marker='o', s=80, alpha=0.8)
        
        # 绘制从基座到落点的连线
        ax.plot([base_pos[0], landing[0]], [base_pos[1], landing[1]], 
                'r-', alpha=0.4, linewidth=1)
        
        # 计算实际角度
        direction_vector = landing - base_pos
        actual_angle = np.arctan2(direction_vector[0], direction_vector[1])
        
        # 将实际角度规范化到[0, 2π]范围，与目标角度范围匹配
        if actual_angle < 0:
            actual_angle += 2 * np.pi
        
        # 计算角度误差（考虑周期性）
        angle_error = abs(actual_angle - target_angle)
        angle_error = min(angle_error, 2 * np.pi - angle_error)
        
        # 添加角度标注（仅前几个）
        if i < 3:
            mid_x = (base_pos[0] + landing[0]) / 2
            mid_y = (base_pos[1] + landing[1]) / 2
            ax.annotate(f'误差:{np.degrees(angle_error):.1f}°', 
                       (mid_x, mid_y), fontsize=8, alpha=0.7)
    
    # 添加距离圆圈参考
    for radius in [1, 2, 3]:
        circle = Circle((base_pos[0], base_pos[1]), radius, 
                       fill=False, color='gray', linestyle=':', alpha=0.3)
        ax.add_patch(circle)
        ax.text(base_pos[0] + radius, base_pos[1], f'{radius}m', 
               fontsize=8, alpha=0.5)
    
    # 添加图例
    legend_handles = [
        mpatches.Patch(color='black', label='机械臂基座'),
        mpatches.Patch(color='green', label='目标方向'),
        mpatches.Patch(color='red', label='实际落点'),
        mpatches.Patch(color='gray', label='距离参考圆')
    ]
    
    ax.legend(handles=legend_handles, loc='upper right')
    
    # 设置坐标轴
    ax.set_xlim(-4.0, 4.0)
    ax.set_ylim(-4.0, 4.0)  # 扩大Y轴范围以显示后方投掷
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('机械臂角度方向投掷结果可视化 - 反向投掷模式')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def visualize_distance_throws(landing_positions, base_pos, save_path):
    """可视化远距离前向投掷结果"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制机械臂基座位置
    base = Circle((base_pos[0], base_pos[1]), 0.1, color='black', label='机械臂基座')
    ax.add_patch(base)
    
    # 绘制前向方向指示线（+Y轴方向）
    forward_line_length = 10.0  # 10米长的前向指示线
    ax.arrow(base_pos[0], base_pos[1], 0, forward_line_length, 
             head_width=0.3, head_length=0.5, fc='blue', ec='blue', 
             alpha=0.7, linewidth=3, label='前向方向 (+Y)')
    
    # 绘制目标区域（100米远，50米偏右的区域）
    target_circle = Circle((base_pos[0] + 50, base_pos[1] + 100), 5.0, 
                          fill=False, color='green', linestyle='--', 
                          alpha=0.8, linewidth=2, label='目标区域 [100,50]')
    ax.add_patch(target_circle)
    
    # 绘制投掷结果
    for i, landing in enumerate(landing_positions):
        # 绘制实际落点
        ax.scatter(landing[0], landing[1], c='red', marker='o', s=80, alpha=0.8)
        
        # 绘制从基座到落点的连线
        ax.plot([base_pos[0], landing[0]], [base_pos[1], landing[1]], 
                'r-', alpha=0.4, linewidth=1)
        
        # 计算投掷距离
        distance = np.linalg.norm(landing - base_pos)
        
        # 添加距离标注（仅前几个）
        if i < 3:
            mid_x = (base_pos[0] + landing[0]) / 2
            mid_y = (base_pos[1] + landing[1]) / 2
            ax.annotate(f'{distance:.1f}m', 
                       (mid_x, mid_y), fontsize=8, alpha=0.7,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    # 添加距离圆圈参考
    for radius in [10, 20, 50, 100]:
        circle = Circle((base_pos[0], base_pos[1]), radius, 
                       fill=False, color='gray', linestyle=':', alpha=0.3)
        ax.add_patch(circle)
        ax.text(base_pos[0] + radius * 0.7, base_pos[1] + radius * 0.7, f'{radius}m', 
               fontsize=8, alpha=0.5)
    
    # 添加图例
    legend_handles = [
        mpatches.Patch(color='black', label='机械臂基座'),
        mpatches.Patch(color='blue', label='前向方向 (+Y)'),
        mpatches.Patch(color='green', label='目标区域 (100m)'),
        mpatches.Patch(color='red', label='实际落点'),
        mpatches.Patch(color='gray', label='距离参考圆')
    ]
    
    ax.legend(handles=legend_handles, loc='upper right')
    
    # 设置坐标轴
    max_distance = max([np.linalg.norm(pos - base_pos) for pos in landing_positions])
    axis_limit = max(max_distance * 1.2, 120)  # 至少显示120米
    
    ax.set_xlim(-axis_limit * 0.3, axis_limit * 0.3)
    ax.set_ylim(-axis_limit * 0.1, axis_limit)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('机械臂远距离偏右前方投掷结果可视化 [100,50]')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def visualize_joint_trajectories(joint_trajectories, control_freq, save_path, episode_info, position_targets=None):
    """Visualize joint angle trajectories over time - overlapping plot with P targets"""
    if not joint_trajectories:
        print("⚠️  No joint trajectory data available for visualization")
        return
    
    # Joint name mapping
    joint_names = [
        "Rotation_R (Shoulder Pan)",
        "Pitch_R (Shoulder Lift)", 
        "Elbow_R (Elbow Flex)",
        "Wrist_Pitch_R (Wrist Flex)",
        "Wrist_Roll_R (Wrist Roll)"
    ]
    joint_colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Create time axis
    time_steps = np.arange(len(joint_trajectories)) / control_freq  # Convert to seconds
    
    # Extract angle data for each joint (convert from radians to degrees)
    joint_data = np.array(joint_trajectories)  # [timesteps, joints]
    joint_data_deg = np.degrees(joint_data)
    
    # Create figure with two subplots: one for overlapping trajectories, one for P targets
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    fig.suptitle(f'Episode 0 Joint Trajectories - Long-distance Forward Throw\n'
                f'Throw Distance: {episode_info.get("throw_distance", 0):.2f}m, '
                f'Forward Deviation: {np.degrees(episode_info.get("direction_error", 0)):.1f}°, '
                f'Throw Velocity: {episode_info.get("throw_velocity", 0):.2f}m/s', 
                fontsize=14, fontweight='bold')
    
    # Plot 1: All joint angles overlapping
    ax1.set_title('Joint Angle Trajectories (Actual Positions)', fontsize=12, fontweight='bold')
    
    for i in range(5):  # 5 joints
        ax1.plot(time_steps, joint_data_deg[:, i], 
                color=joint_colors[i], linewidth=2, 
                label=f'{joint_names[i]}', alpha=0.8)
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Angle (deg)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add throw detection marker if available
    throw_step = episode_info.get("throw_step", None)
    if throw_step is not None:
        throw_time = throw_step / control_freq
        ax1.axvline(x=throw_time, color='red', linestyle='--', alpha=0.7, 
                   label=f'Throw Detection ({throw_time:.2f}s)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Calculate and display joint motion statistics
    motion_stats = []
    for i, name in enumerate(joint_names):
        min_angle = np.min(joint_data_deg[:, i])
        max_angle = np.max(joint_data_deg[:, i])
        range_angle = max_angle - min_angle
        mean_angle = np.mean(joint_data_deg[:, i])
        motion_stats.append(f"{name.split('(')[0].strip()}: Range={range_angle:.1f}°, Mean={mean_angle:.1f}°")
    
    # Add statistics text box
    stats_text = "Joint Motion Statistics:\n" + "\n".join([f"• {stat}" for stat in motion_stats])
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    # Plot 2: Position targets (P targets) - actual targets sent to robot
    ax2.set_title('Joint Position Targets (Policy Output + Default Positions)', fontsize=12, fontweight='bold')
    
    if position_targets is not None and len(position_targets) > 0:
        # Use actual recorded position targets
        target_data = np.array(position_targets)  # [timesteps, joints]
        target_data_deg = np.degrees(target_data)
        
        # Ensure time_steps and target_data have the same length
        min_length = min(len(time_steps), len(target_data_deg))
        time_steps_targets = time_steps[:min_length]
        target_data_deg = target_data_deg[:min_length]
        
        for i in range(5):  # 5 joints
            ax2.plot(time_steps_targets, target_data_deg[:, i], 
                    color=joint_colors[i], linewidth=2, linestyle='--',
                    label=f'{joint_names[i]} Target', alpha=0.8)
    else:
        # Fallback: Simulate position targets using smoothed version of joint trajectories
        print("⚠️  No position target data available, using simulated targets")
        
        # Use simple moving average for fallback
        window_size = 5
        for i in range(5):  # 5 joints
            if len(joint_data_deg) >= window_size:
                smoothed_trajectory = np.convolve(joint_data_deg[:, i], 
                                                np.ones(window_size)/window_size, mode='same')
            else:
                smoothed_trajectory = joint_data_deg[:, i]
            
            ax2.plot(time_steps, smoothed_trajectory, 
                    color=joint_colors[i], linewidth=2, linestyle='--',
                    label=f'{joint_names[i]} Target (Simulated)', alpha=0.8)
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Target Angle (deg)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add throw detection marker to P target plot as well
    if throw_step is not None:
        ax2.axvline(x=throw_time, color='red', linestyle='--', alpha=0.7, 
                   label=f'Throw Detection ({throw_time:.2f}s)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add episode performance summary
    performance_text = f"""Episode Performance Summary:
• Total Duration: {len(joint_trajectories) / control_freq:.2f} seconds
• Control Frequency: {control_freq} Hz
• Total Steps: {len(joint_trajectories)}
• Throw Distance: {episode_info.get("throw_distance", 0):.3f} m
• Forward Deviation: {np.degrees(episode_info.get("direction_error", 0)):.1f}°
• Throw Velocity: {episode_info.get("throw_velocity", 0):.2f} m/s
• Episode Reward: {episode_info.get("episode_reward", 0):.1f}
• Direction Success: {'Yes' if episode_info.get("direction_success", False) else 'No'}
• Distance Success: {'Yes' if episode_info.get("distance_success", False) else 'No'}
• Overall Success: {'Yes' if episode_info.get("overall_success", False) else 'No'}"""
    
    ax2.text(0.02, 0.02, performance_text, transform=ax2.transAxes, 
            verticalalignment='bottom', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Joint angle trajectory visualization saved at: {save_path}")

def get_cfgs():
    """获取评估配置 - 远距离前向投掷模式"""
    # 环境配置（从训练脚本同步）
    env_cfg = {
        "num_actions": 5,  # 5个关节
        "robot_file": "/home/nvidiapc/dodo/Genesis/genesis/assets/xml/le_simulation/simulation/single_le_box.xml",
        "joint_names": [
            "Rotation_R",
            "Pitch_R", 
            "Elbow_R",
            "Wrist_Pitch_R",
            "Wrist_Roll_R"
        ],
        "default_joint_angles": [0.0, 3.14, 0.0, 0.0, 3.14],  # 保持初始关节位置
        "kp": 50.0,  # 固定位置增益
        "kd": 5.0,   # 速度增益
        "torque_limit": 35.0,  # 力矩限制
        "episode_length_s": 5.0,  # episode长度
        "control_freq": 50,  # 控制频率50Hz
        "action_scale": 0.7,  # 动作缩放
        "clip_actions": 1.2,  # 动作裁剪范围
        "freeze_duration_s": 0.4,  # 冻结时间
    }
    
    # 观测配置 - 远距离前向投掷
    obs_cfg = {
        "num_obs": 5 + 5,  # 关节位置(5) + 关节速度(5)，移除目标角度
        "obs_scales": {
            "dof_pos": 1.0,      # 关节位置
            "dof_vel": 0.05,     # 关节速度
        },
    }
    
    # 奖励配置 - 远距离前向投掷训练（激进模式，与训练脚本一致）
    reward_cfg = {
        "target_throw_position": [100.0, 0.0],  # 目标投掷位置
        "distance_tolerance": 5.0,  # 距离容忍度
        "direction_tolerance": 0.3,  # 方向容忍度
        "reward_scales": {
            # 主要奖励 - 远距离投掷（大幅增强）
            "throw_distance_reward": 5000.0,  # 投掷距离奖励（越远越好）- 增强2.5倍
            "forward_direction_reward": 3000.0,  # 前向方向奖励 - 增强2倍
            "throw_success": 800.0,  # 基本投掷成功奖励 - 增强
            "velocity_magnitude_reward": 1200.0,  # 球初始速度奖励 - 增强50%
            
            # 特殊奖励 - 鼓励Rotation_R关节使用以瞄准[100, 50]
            "rotation_r_usage_reward": 2000.0,  # 鼓励使用Rotation_R关节进行侧向瞄准
            "target_alignment_reward": 2500.0,  # 奖励瞄准正确目标方向[100, 50]
            
            # 移除约束，鼓励所有关节自由运动
            # "wrist_roll_lock": 0.0,  # 完全移除手腕滚转锁定，允许自由扭转
            "joint_vel_penalty": 0.005,  # 大幅减少关节速度惩罚，鼓励快速运动
            "action_smoothness": 0.05,   # 减少动作平滑度惩罚，允许更激进的动作
            "energy_penalty": 0.002,   # 减少能耗惩罚，鼓励大力投掷
        },
    }
    
    # 命令配置 - 固定偏右前方投掷目标
    command_cfg = {
        "num_commands": 0,  # 不需要命令，目标固定
        "target_position": [100.0, 50.0],  # 固定目标：100米远，50米偏右（鼓励Rotation_R使用）
        "training_mode": "max_distance_forward",  # 训练模式标识
    }
    
    return env_cfg, obs_cfg, reward_cfg, command_cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="so_arm_throwing")
    parser.add_argument("--ckpt", type=int, default=700, help="检查点编号，-1表示最终模型")
    parser.add_argument("--num_episodes", type=int, default=10, help="评估的episode数量")
    # 移除角度参数，因为目标固定为前向投掷
    parser.add_argument("--visualize", action="store_true", default=True, help="可视化投掷结果")
    parser.add_argument("--record", action="store_true", help="记录关节角度序列用于真实机器人控制")
    parser.add_argument("--record_type", type=str, default="target", choices=["target", "state"], 
                       help="记录类型: 'target'=目标关节角度(NN输出), 'state'=实际关节位置状态")
    parser.add_argument("--record_episode", type=int, default=0, help="指定要记录的episode编号（0-based）")
    args = parser.parse_args()
    
    # 初始化Genesis
    gs.init(logging_level="warning")
    
    # 加载配置
    log_dir = f"../../logs/{args.exp_name}"  # 修正路径：从examples/le/到Genesis根目录
    if not os.path.exists(log_dir):
        print(f"❌ 日志目录不存在: {log_dir}")
        print(f"请先运行训练: python le_train.py -e {args.exp_name}")
        return
    
    try:
        # 对于角度方向训练，使用内置配置而不是从文件加载
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
        
        # 尝试从训练配置文件获取train_cfg（如果存在）
        try:
            _, _, _, _, train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
        except:
            # 如果加载失败，使用默认的train_cfg
            train_cfg = {
                "policy": {
                    "activation": "elu",
                    "actor_hidden_dims": [256, 128, 64],
                    "critic_hidden_dims": [256, 128, 64],
                    "init_noise_std": 1.5,
                    "class_name": "ActorCritic",
                },
                "algorithm": {
                    "class_name": "PPO",
                    "clip_param": 0.2,
                    "desired_kl": 0.01,
                    "entropy_coef": 0.03,
                    "gamma": 0.97,
                    "lam": 0.95,
                    "learning_rate": 0.0008,
                    "max_grad_norm": 1.0,
                    "num_learning_epochs": 5,
                    "num_mini_batches": 4,
                    "schedule": "adaptive",
                    "use_clipped_value_loss": True,
                    "value_loss_coef": 1.0,
                },
                "init_member_classes": {},
                "runner": {
                    "checkpoint": -1,
                    "experiment_name": args.exp_name,
                    "load_run": -1,
                    "log_interval": 1,
                    "max_iterations": 700,
                    "record_interval": -1,
                    "resume": False,
                    "resume_path": None,
                    "run_name": "",
                },
                "runner_class_name": "OnPolicyRunner",
                "num_steps_per_env": 32,
                "save_interval": 50,
                "empirical_normalization": None,
                "seed": 1,
                "logger": "tensorboard",
                "tensorboard_subdir": "tb",
            }
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return
    
    # 显示当前配置信息
    print(f"\n{'='*60}")
    print(f"远距离前向投掷评估配置 - {args.exp_name}")
    print(f"{'='*60}")
    print(f"Episode数量: {args.num_episodes}")
    print(f"控制频率: {env_cfg['control_freq']} Hz")
    print(f"Episode长度: {env_cfg['episode_length_s']} 秒")
    print(f"观测维度: {obs_cfg['num_obs']} (关节状态)")
    print(f"P增益: {env_cfg['kp']} (固定值)")
    
    if args.record:
        record_type_desc = "目标关节角度(NN输出)" if args.record_type == "target" else "实际关节位置状态"
        print(f"🎬 记录模式: 启用 (记录episode {args.record_episode}, 类型: {record_type_desc})")
    
    target_pos = command_cfg["target_position"]
    print(f"投掷目标: [{target_pos[0]:.0f}, {target_pos[1]:.0f}]米 (远距离前向投掷)")
    print(f"训练模式: {command_cfg['training_mode']}")
    print(f"方向容忍度: {reward_cfg['direction_tolerance']:.2f} 弧度 ({np.degrees(reward_cfg['direction_tolerance']):.1f}°)")
    print(f"距离容忍度: {reward_cfg['distance_tolerance']:.1f}m")
    print(f"{'='*60}")
    
    # 创建环境（单个环境用于评估）
    env = SoArmEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )
    
    # 添加红色球体作为可视化指示器
    print("🔴 添加可视化指示器球体到位置 [1, 0, 1.2]")
    try:
        # 使用MJCF方式创建球体
        indicator_ball = env.scene.add_entity(
            gs.morphs.MJCF(
                xml_string="""
                <mujoco>
                    <worldbody>
                        <body name="indicator_ball" pos="1.0 0.0 1.2">
                            <geom type="sphere" size="0.05" rgba="1 0 0 1" contype="0" conaffinity="0"/>
                        </body>
                    </worldbody>
                </mujoco>
                """
            )
        )
        print("✅ 红色指示器球体添加完成 (MJCF方式)")
    except Exception as e:
        print(f"⚠️  MJCF方式失败，尝试Box方式: {e}")
        try:
            # 备用方案：使用Box创建小立方体作为指示器
            indicator_ball = env.scene.add_entity(
                gs.morphs.Box(
                    pos=(1.0, 0.0, 1.2),
                    size=(0.05, 0.05, 0.05),
                    color=(1.0, 0.0, 0.0, 1.0),  # 红色
                    fixed=True
                )
            )
            print("✅ 红色指示器立方体添加完成 (Box方式)")
        except Exception as e2:
            print(f"❌ 指示器添加失败: {e2}")
            print("💡 继续运行，但没有可视化指示器")
    
    # 加载训练器和策略
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    
    # 加载模型
    ckpt_name = f"model_{args.ckpt}.pt" if args.ckpt >= 0 else "model_final.pt"
    model_path = os.path.join(log_dir, ckpt_name)
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        # 尝试查找可用的检查点
        available_ckpts = []
        for f in os.listdir(log_dir):
            if f.startswith("model_") and f.endswith(".pt"):
                available_ckpts.append(f)
        if available_ckpts:
            print(f"可用的检查点: {', '.join(available_ckpts)}")
        return
    
    print(f"📁 加载模型: {model_path}")
    runner.load(model_path)
    
    # 获取推理策略
    policy = runner.get_inference_policy(device=gs.device)
    
    # 评估统计 - 远距离前向投掷
    episode_rewards = []
    throw_distances = []
    direction_errors = []  # 前向方向误差统计
    throw_velocities = []  # 投掷速度统计
    landing_positions = []
    success_rates = []
    
    # 记录变量
    recorded_joint_sequence = []
    recorded_episode_info = {}
    
    # Episode 0的关节轨迹记录（总是记录，用于可视化）
    episode_0_joint_trajectory = []
    episode_0_info = {}
    
    print(f"\n🚀 开始评估 {args.num_episodes} 个episodes...")
    
    # 主评估循环
    for episode in range(args.num_episodes):
        obs, _ = env.reset()
        
        # 固定目标：100米远，正前方
        target_position = command_cfg["target_position"]
        
        episode_reward = 0
        max_ball_height = env.ball_init_pos[2]
        throw_detected = False
        throw_velocity = 0.0
        throw_step = None
        
        # 当前episode的关节角度序列
        current_episode_joints = []
        # Episode 0的原始关节角度轨迹（弧度）
        current_episode_joint_trajectory = []
        # Episode 0的位置目标轨迹（P目标，弧度）
        current_episode_position_targets = []
        
        print(f"\n📍 Episode {episode + 1}:")
        print(f"  投掷目标: [{target_position[0]:.0f}, {target_position[1]:.0f}]米 (远距离前向投掷)")
        print(f"  当前P增益: {env_cfg['kp']} (固定值)")
        
        if args.record and episode == args.record_episode:
            record_desc = "目标关节角度" if args.record_type == "target" else "实际关节位置状态"
            print(f"  🎬 正在记录此episode的{record_desc}序列...")
        
        if episode == 0:
            print(f"  📈 正在记录Episode 0的关节角度轨迹用于可视化...")
        
        with torch.no_grad():
            while True:
                # 获取动作
                actions = policy(obs)
                
                # 记录关节角度序列（如果是指定的记录episode）
                if args.record and episode == args.record_episode:
                    if args.record_type == "target":
                        # 记录目标关节角度（NN输出计算的目标位置）
                        is_frozen = env.freeze_counter[0] > 0
                        if is_frozen:
                            # 冻结期间使用默认姿态
                            joint_angles = env.default_dof_pos.cpu().numpy()
                        else:
                            # 正常期间使用策略动作计算目标位置
                            joint_angles = (actions[0] * env.env_cfg["action_scale"] + env.default_dof_pos).cpu().numpy()
                        
                        # 映射到真实机器人关节角度并记录
                        robot_angles = map_joint_angles_to_robot(joint_angles)
                        current_episode_joints.append(robot_angles)
                
                # 执行动作
                obs, rews, dones, infos = env.step(actions)
                
                # 记录Episode 0的关节轨迹（原始弧度值）和位置目标
                if episode == 0:
                    current_joint_angles = env.dof_pos[0].cpu().numpy()
                    current_episode_joint_trajectory.append(current_joint_angles.copy())
                    
                    # 记录位置目标（P目标）- 计算发送给机器人的实际目标位置
                    is_frozen = env.freeze_counter[0] > 0
                    if is_frozen:
                        # 冻结期间使用默认姿态作为目标
                        position_targets = env.default_dof_pos.cpu().numpy()
                    else:
                        # 正常期间使用策略动作计算目标位置
                        position_targets = (actions[0] * env.env_cfg["action_scale"] + env.default_dof_pos).cpu().numpy()
                    
                    current_episode_position_targets.append(position_targets.copy())
                
                # 记录实际关节位置状态（在step之后，获取执行后的实际状态）
                if args.record and episode == args.record_episode and args.record_type == "state":
                    # 记录实际关节位置状态
                    actual_joint_angles = env.dof_pos[0].cpu().numpy()
                    
                    # 映射到真实机器人关节角度并记录
                    robot_angles = map_joint_angles_to_robot(actual_joint_angles)
                    current_episode_joints.append(robot_angles)
                episode_reward += rews[0].item()
                
                # 更新最大球高度和投掷检测
                current_ball_height = env.ball_pos[0, 2].item()
                max_ball_height = max(max_ball_height, current_ball_height)
                if not throw_detected and env.throw_detected[0]:
                    throw_detected = True
                    throw_step = env.episode_length_buf[0].item()
                    # 记录投掷速度
                    throw_velocity = env.throw_velocity_magnitude[0].item()
                    print(f"  🏀 投掷检测! 步数: {env.episode_length_buf[0]}, 投掷速度: {throw_velocity:.2f}m/s")
                
                # 打印状态信息（每50步）
                if env.episode_length_buf[0] % 50 == 0:
                    ball_pos = env.ball_pos[0].cpu().numpy()
                    freeze_remaining = max(0, env.freeze_counter[0].item() * env.dt)
                    status = f"冻结中({freeze_remaining:.1f}s)" if freeze_remaining > 0 else "运行中"
                    print(f"  步数: {env.episode_length_buf[0]}, 球位置: ({ball_pos[0]:.2f}, {ball_pos[1]:.2f}, {ball_pos[2]:.2f}), 状态: {status}")
                
                # Episode结束
                if dones[0]:
                    # 获取落点位置
                    landing_pos = env.ball_landing_pos[0].cpu().numpy()
                    landing_positions.append(landing_pos.copy())
                    
                    # 计算投掷距离（从基座到落点）
                    base_pos = env.base_pos[:2].cpu().numpy()
                    throw_dist = np.linalg.norm(landing_pos - base_pos)
                    throw_distances.append(throw_dist)
                    
                    # 计算前向方向偏差
                    direction_vector = landing_pos - base_pos
                    direction_norm = np.linalg.norm(direction_vector)
                    
                    if direction_norm > 0.1:  # 避免除零
                        # 归一化方向向量
                        normalized_direction = direction_vector / direction_norm
                        # 前向方向是+Y轴
                        forward_direction = np.array([0.0, 1.0])
                        # 计算与前向方向的角度偏差
                        cos_similarity = np.dot(normalized_direction, forward_direction)
                        direction_error = np.arccos(np.clip(cos_similarity, -1.0, 1.0))  # 0到π的角度偏差
                    else:
                        direction_error = np.pi  # 最大偏差
                    
                    direction_errors.append(float(direction_error))
                    throw_velocities.append(throw_velocity)
                    
                    # 计算成功率（基于方向和距离）
                    direction_tolerance = reward_cfg.get("direction_tolerance", 0.3)
                    distance_tolerance = reward_cfg.get("distance_tolerance", 5.0)
                    direction_success = direction_error <= direction_tolerance
                    distance_success = throw_dist >= 1.0  # 至少投掷1米
                    overall_success = direction_success and distance_success
                    success_rates.append(overall_success)
                    
                    print(f"  📍 球落地位置: ({landing_pos[0]:.3f}, {landing_pos[1]:.3f})")
                    print(f"  📐 投掷距离: {throw_dist:.3f}m")
                    print(f"  📐 前向偏差: {np.degrees(direction_error):.1f}° (容忍度: {np.degrees(direction_tolerance):.1f}°)")
                    print(f"  🚀 投掷速度: {throw_velocity:.2f}m/s")
                    print(f"  📈 最大球高度: {max_ball_height:.3f}m")
                    print(f"  🏆 Episode奖励: {episode_reward:.2f}")
                    print(f"  🎯 方向成功: {'✅' if direction_success else '❌'}")
                    print(f"  🎯 距离成功: {'✅' if distance_success else '❌'} (≥1.0m)")
                    print(f"  🎯 综合成功: {'✅' if overall_success else '❌'}")
                    print(f"  🎲 投掷检测: {'是' if throw_detected else '否'}")
                    
                    # 保存Episode 0的轨迹信息
                    if episode == 0:
                        episode_0_joint_trajectory = current_episode_joint_trajectory
                        episode_0_position_targets = current_episode_position_targets
                        episode_0_info = {
                            "episode": episode,
                            "target_position": target_position,
                            "direction_error": float(direction_error),
                            "throw_velocity": float(throw_velocity),
                            "landing_pos": landing_pos.tolist(),
                            "episode_reward": float(episode_reward),
                            "throw_distance": float(throw_dist),
                            "max_ball_height": float(max_ball_height),
                            "throw_detected": bool(throw_detected),
                            "throw_step": throw_step,
                            "total_steps": len(current_episode_joint_trajectory),
                            "direction_success": bool(direction_success),
                            "distance_success": bool(distance_success),
                            "overall_success": bool(overall_success),
                            "position_targets": episode_0_position_targets  # 添加位置目标数据
                        }
                        print(f"  📈 已记录Episode 0关节轨迹: {len(current_episode_joint_trajectory)} 步")
                        print(f"  📈 已记录Episode 0位置目标: {len(current_episode_position_targets)} 步")
                    
                    # 保存记录的关节序列
                    if args.record and episode == args.record_episode:
                        recorded_joint_sequence = current_episode_joints
                        recorded_episode_info = {
                            "episode": episode,
                            "target_position": target_position,
                            "direction_error": float(direction_error),
                            "throw_velocity": float(throw_velocity),
                            "landing_pos": landing_pos.tolist(),
                            "episode_reward": float(episode_reward),
                            "throw_distance": float(throw_dist),
                            "max_ball_height": float(max_ball_height),
                            "throw_detected": bool(throw_detected),
                            "total_steps": len(current_episode_joints),
                            "direction_success": bool(direction_success),
                            "distance_success": bool(distance_success),
                            "overall_success": bool(overall_success)
                        }
                        record_desc = "目标关节角度" if args.record_type == "target" else "实际关节位置状态"
                        print(f"  🎬 已记录 {len(current_episode_joints)} 步{record_desc}序列")
                    
                    episode_rewards.append(episode_reward)
                    break
    
    # 保存记录的关节序列
    if args.record and recorded_joint_sequence:
        # 保存为JSON格式
        record_data = {
            "episode_info": recorded_episode_info,
            "joint_sequence": recorded_joint_sequence,
            "record_type": args.record_type,
            "record_description": {
                "target": "目标关节角度序列 - 神经网络输出计算的目标位置，发送给机器人的控制命令",
                "state": "实际关节位置状态序列 - 机器人执行后的实际关节位置，反映真实运动轨迹"
            }[args.record_type],
            "joint_mapping": {
                "simulation": ["Rotation_R", "Pitch_R", "Elbow_R", "Wrist_Pitch_R", "Wrist_Roll_R"],
                "robot": ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos"]
            },
            "training_mode": "max_distance_forward",
            "direction_info": {
                "target_position": recorded_episode_info["target_position"],
                "direction_error_rad": recorded_episode_info["direction_error"],
                "direction_error_deg": np.degrees(recorded_episode_info["direction_error"]),
                "throw_velocity": recorded_episode_info["throw_velocity"],
                "direction_tolerance_rad": reward_cfg.get("direction_tolerance", 0.3),
                "direction_tolerance_deg": np.degrees(reward_cfg.get("direction_tolerance", 0.3)),
            },
            "control_frequency": env_cfg["control_freq"],
            "timestamp": time.strftime("%Y%m%d_%H%M%S")
        }
        
        # 转换为JSON可序列化格式
        record_data = convert_to_json_serializable(record_data)
        
        record_json_path = os.path.join(log_dir, f"robot_control_sequence_{args.record_type}_ep{args.record_episode}.json")
        with open(record_json_path, 'w', encoding='utf-8') as f:
            json.dump(record_data, f, indent=2, ensure_ascii=False)
        
        # 生成可执行的Python脚本
        script_path = os.path.join(log_dir, f"robot_control_script_{args.record_type}_ep{args.record_episode}.py")
        generate_robot_control_script(recorded_joint_sequence, script_path, recorded_episode_info, args.record_type)
        
        record_type_desc = {
            "target": "目标关节角度序列(NN输出)",
            "state": "实际关节位置状态序列"
        }[args.record_type]
        
        print(f"\n🎬 {record_type_desc}记录完成!")
        print(f"📄 JSON数据保存至: {record_json_path}")
        print(f"🐍 Python脚本保存至: {script_path}")
        print(f"📊 记录统计:")
        print(f"  - Episode: {recorded_episode_info['episode']}")
        print(f"  - 记录类型: {args.record_type.upper()} ({record_type_desc})")
        print(f"  - 总步数: {recorded_episode_info['total_steps']}")
        print(f"  - 执行时间: {recorded_episode_info['total_steps'] * 0.02:.1f}秒")
        print(f"  - 目标位置: {recorded_episode_info['target_position']}")
        print(f"  - 前向偏差: {np.degrees(recorded_episode_info['direction_error']):.1f}°")
        print(f"  - 投掷速度: {recorded_episode_info['throw_velocity']:.2f}m/s")
        print(f"  - 投掷距离: {recorded_episode_info['throw_distance']:.3f}m")
        print(f"  - Episode奖励: {recorded_episode_info['episode_reward']:.3f}")
        print(f"  - 方向成功: {'✅' if recorded_episode_info['direction_success'] else '❌'}")
        print(f"  - 距离成功: {'✅' if recorded_episode_info['distance_success'] else '❌'}")
        print(f"  - 综合成功: {'✅' if recorded_episode_info['overall_success'] else '❌'}")
        
        if args.record_type == "target":
            print(f"💡 使用方法 (远距离前向投掷控制):")
            print(f"  - 直接运行脚本: python {script_path}")
            print(f"  - 数据可直接用于机器人远距离前向投掷控制")
        else:
            print(f"💡 使用方法 (投掷轨迹分析):")
            print(f"  - 分析轨迹: python {script_path}")
            print(f"  - 数据用于投掷轨迹分析，复现时需要转换为控制命令")
            
        print(f"  - 或加载JSON数据进行自定义处理")
        print(f"📝 关节角度格式样例:")
        if recorded_joint_sequence:
            sample_angles = recorded_joint_sequence[0]
            print(f"  {{")
            for key, value in sample_angles.items():
                print(f"    '{key}': {value:.1f},")
            print(f"  }}")
    
    # 打印统计结果
    print("\n" + "="*60)
    print("📊 远距离前向投掷评估结果汇总:")
    print("="*60)
    print(f"平均Episode奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"平均前向偏差: {np.degrees(np.mean(direction_errors)):.1f}° ± {np.degrees(np.std(direction_errors)):.1f}°")
    print(f"平均投掷距离: {np.mean(throw_distances):.3f} ± {np.std(throw_distances):.3f}m")
    print(f"平均投掷速度: {np.mean(throw_velocities):.2f} ± {np.std(throw_velocities):.2f}m/s")
    print(f"最小前向偏差: {np.degrees(np.min(direction_errors)):.1f}°")
    print(f"最大前向偏差: {np.degrees(np.max(direction_errors)):.1f}°")
    print(f"最大投掷距离: {np.max(throw_distances):.3f}m")
    
    # 成功率统计
    overall_success_rate = np.mean(success_rates) * 100
    direction_tolerance_deg = np.degrees(reward_cfg.get("direction_tolerance", 0.3))
    print(f"综合成功率: {overall_success_rate:.1f}% (前向偏差<{direction_tolerance_deg:.1f}° + 距离≥1.0m)")
    
    # 单项成功率
    direction_successes = [err <= reward_cfg.get("direction_tolerance", 0.3) for err in direction_errors]
    distance_successes = [dist >= 1.0 for dist in throw_distances]
    
    print(f"前向准确率: {np.mean(direction_successes) * 100:.1f}% (偏差<{direction_tolerance_deg:.1f}°)")
    print(f"距离成功率: {np.mean(distance_successes) * 100:.1f}% (距离≥1.0m)")
    
    # 前向偏差分布统计
    print(f"\n前向偏差分布:")
    print(f"  <5°:  {np.sum(np.array(direction_errors) < np.radians(5)) / len(direction_errors) * 100:.1f}%")
    print(f"  <10°: {np.sum(np.array(direction_errors) < np.radians(10)) / len(direction_errors) * 100:.1f}%")
    print(f"  <15°: {np.sum(np.array(direction_errors) < np.radians(15)) / len(direction_errors) * 100:.1f}%")
    print(f"  <20°: {np.sum(np.array(direction_errors) < np.radians(20)) / len(direction_errors) * 100:.1f}%")
    
    # 可视化结果
    if args.visualize:
        viz_path = os.path.join(log_dir, "distance_throw_results.png")
        base_pos = np.array([0.0, 0.0])  # 机器人基座位置
        visualize_distance_throws(landing_positions, base_pos, viz_path)
        print(f"\n🎨 远距离前向投掷可视化结果保存在: {viz_path}")
    
    # 保存详细结果
    results = {
        "episode_rewards": episode_rewards,
        "direction_errors": direction_errors,
        "throw_distances": throw_distances,
        "throw_velocities": throw_velocities,
        "landing_positions": landing_positions,
        "success_rates": success_rates,
        "throw_statistics": {
            "mean_direction_error_deg": float(np.degrees(np.mean(direction_errors))),
            "std_direction_error_deg": float(np.degrees(np.std(direction_errors))),
            "min_direction_error_deg": float(np.degrees(np.min(direction_errors))),
            "max_direction_error_deg": float(np.degrees(np.max(direction_errors))),
            "mean_throw_distance": float(np.mean(throw_distances)),
            "max_throw_distance": float(np.max(throw_distances)),
            "mean_throw_velocity": float(np.mean(throw_velocities)),
            "max_throw_velocity": float(np.max(throw_velocities)),
            "direction_success_rate": float(np.mean([err <= reward_cfg.get("direction_tolerance", 0.3) for err in direction_errors])),
            "distance_success_rate": float(np.mean([dist >= 1.0 for dist in throw_distances])),
            "overall_success_rate": float(np.mean(success_rates)),
        },
        "config": {
            "direction_tolerance_deg": float(np.degrees(reward_cfg.get("direction_tolerance", 0.3))),
            "target_position": command_cfg["target_position"],
            "training_mode": command_cfg["training_mode"],
            "num_episodes": args.num_episodes,
        }
    }
    
    # 转换为JSON可序列化格式
    results = convert_to_json_serializable(results)
    
    results_path = os.path.join(log_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"📄 详细结果保存在: {results_path}")
    print("="*60)

    # 可视化关节角度轨迹（如果有Episode 0数据）
    if episode_0_joint_trajectory and episode_0_info:
        print(f"\n📈 生成Episode 0关节角度轨迹可视化...")
        joint_trajectory_path = os.path.join(log_dir, "joint_trajectories_episode_0.png")
        # 传递位置目标数据（如果有的话）
        position_targets = episode_0_info.get("position_targets", None)
        visualize_joint_trajectories(episode_0_joint_trajectory, env_cfg["control_freq"], joint_trajectory_path, episode_0_info, position_targets)
    else:
        print(f"\n⚠️  未找到Episode 0的关节轨迹数据，跳过关节角度轨迹可视化")

if __name__ == "__main__":
    main()