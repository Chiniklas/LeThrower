"""keyboard control for XLeRobot, python mujoco"""

import time
import mujoco_viewer
import numpy as np
import glfw
import mujoco


class XLeRobotController:
    def __init__(self, mjcf_path):
        """Initialize the XLeRobot controller with the given MJCF model path."""
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.render_freq = 60  # Hz
        self.render_interval = 1.0 / self.render_freq
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.camera.trackbodyid = self.model.body("chassis").id
        self.camera.distance = 3.0
        self.camera.azimuth = 90.0
        self.camera.elevation = -30.0
        self.camera.lookat = np.array([0.0, 0.0, 0.0])

        self.abs_vel = np.array([1, 1, 1])
        self.chassis_ref_vel = np.zeros(3)
        self.qCmd = np.zeros(self.model.nu)
        self.qdCmd = np.zeros(self.model.nu)
        self.qFb = np.zeros(self.model.nu)
        self.qdFb = np.zeros(self.model.nu)
        self.last_render_time = time.time()
        self.kp = 1

        self.key_states = {
            "home": False,  # Forward (+x)
            "end": False,  # Backward (-x)
            "delete": False,  # Leftward (+y)
            "page_down": False,  # Rightward (-y)
            "insert": False,  # Rotate CCW (+z)
            "page_up": False,  # Rotate CW (-z)
            # Left arm controls
            "q": False,  # Left arm joint 1 positive
            "a": False,  # Left arm joint 1 negative
            "w": False,  # Left arm joint 2 positive
            "s": False,  # Left arm joint 2 negative
            "e": False,  # Left arm joint 3 positive
            "d": False,  # Left arm joint 3 negative
            # Right arm controls
            "u": False,  # Right arm joint 1 positive
            "j": False,  # Right arm joint 1 negative
            "i": False,  # Right arm joint 2 positive
            "k": False,  # Right arm joint 2 negative
            "o": False,  # Right arm joint 3 positive
            "l": False,  # Right arm joint 3 negative
        }

    def update_feedback(self):
        """Calculate current yaw angle from quaternion"""
        self.qFb = self.data.qpos
        self.qdFb = self.data.qvel

    def update_keyboards(self):
        """Check key states using GLFW directly from viewer window"""
        try:
            window = self.viewer.window
            if window is None:
                return

            key_map = {
                "home": glfw.KEY_HOME,
                "end": glfw.KEY_END,
                "delete": glfw.KEY_DELETE,
                "page_down": glfw.KEY_PAGE_DOWN,
                "insert": glfw.KEY_INSERT,
                "page_up": glfw.KEY_PAGE_UP,
                # Left arm keys
                "q": glfw.KEY_Q,
                "a": glfw.KEY_A,
                "w": glfw.KEY_W,
                "s": glfw.KEY_S,
                "e": glfw.KEY_E,
                "d": glfw.KEY_D,
                # Right arm keys
                "u": glfw.KEY_U,
                "j": glfw.KEY_J,
                "i": glfw.KEY_I,
                "k": glfw.KEY_K,
                "o": glfw.KEY_O,
                "l": glfw.KEY_L,
            }

            for key_name, glfw_key in key_map.items():
                self.key_states[key_name] = glfw.get_key(window, glfw_key) == glfw.PRESS

        except Exception:
            pass

    def update_reference(self):
        # X-direction (forward/backward)
        yaw = self.qFb[2]
        rotmz = np.array(
            [
                [np.cos(yaw), np.sin(yaw), 0],
                [-np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1],
            ]
        )
        chassis_vel = rotmz @ self.qdFb[0:3]

        self.chassis_ref_vel = np.zeros(3)
        if self.key_states["home"]:
            self.chassis_ref_vel[0] = self.abs_vel[0]
        elif self.key_states["end"]:
            self.chassis_ref_vel[0] = -self.abs_vel[0]
        if self.key_states["delete"]:
            self.chassis_ref_vel[1] = self.abs_vel[1]
        elif self.key_states["page_down"]:
            self.chassis_ref_vel[1] = -self.abs_vel[1]

        if self.key_states["insert"]:
            self.chassis_ref_vel[2] = self.abs_vel[2]
        elif self.key_states["page_up"]:
            self.chassis_ref_vel[2] = -self.abs_vel[2]

        k_p = 10
        k_p_rot = 100
        self.qdCmd[0] = self.chassis_ref_vel[0] * np.cos(yaw) + \
                        self.chassis_ref_vel[1] * np.cos(yaw + 1.5708) + \
                        k_p * (self.chassis_ref_vel[0] - chassis_vel[0]) * np.cos(yaw) + \
                        k_p * (self.chassis_ref_vel[1] - chassis_vel[1]) * np.cos(yaw + 1.5708)
        self.qdCmd[1] = self.chassis_ref_vel[0] * np.sin(yaw) + \
                        self.chassis_ref_vel[1] * np.sin(yaw + 1.5708) + \
                        k_p * (self.chassis_ref_vel[0] - chassis_vel[0]) * np.sin(yaw) + \
                        k_p * (self.chassis_ref_vel[1] - chassis_vel[1]) * np.sin(yaw + 1.5708)
        self.qdCmd[2] = self.chassis_ref_vel[2] + k_p_rot * (self.chassis_ref_vel[2] - chassis_vel[2])

        radius = 0.1
        vel2wheel_matrix = np.array(
            [[0, 1, -radius], [-np.sqrt(3) * 0.5, -0.5, -radius], [np.sqrt(3) * 0.5, -0.5, -radius]]
        )
        coe_vel_to_wheel = 20
        self.qCmd[15:18] = coe_vel_to_wheel * np.dot(vel2wheel_matrix, chassis_vel)
        self.qdCmd[2] = np.clip(self.qdCmd[2], -1.0, 1.0)

        # Left arm joint control (qCmd[3:9])
        arm_step = 0.005

        # Left arm joint 1
        if self.key_states["q"]:
            self.qCmd[3] += arm_step
        elif self.key_states["a"]:
            self.qCmd[3] -= arm_step

        # Left arm joint 2
        if self.key_states["w"]:
            self.qCmd[4] += arm_step
        elif self.key_states["s"]:
            self.qCmd[4] -= arm_step

        # Left arm joint 3
        if self.key_states["e"]:
            self.qCmd[5] += arm_step
        elif self.key_states["d"]:
            self.qCmd[5] -= arm_step

        # Right arm joint control (qCmd[9:15])
        # Right arm joint 1
        if self.key_states["u"]:
            self.qCmd[9] += arm_step
        elif self.key_states["j"]:
            self.qCmd[9] -= arm_step

        # Right arm joint 2
        if self.key_states["i"]:
            self.qCmd[10] += arm_step
        elif self.key_states["k"]:
            self.qCmd[10] -= arm_step

        # Right arm joint 3
        if self.key_states["o"]:
            self.qCmd[11] += arm_step
        elif self.key_states["l"]:
            self.qCmd[11] -= arm_step

        # Keep other joints at zero
        self.qCmd[6:9] = 0.0  # Left arm joints 4-6
        self.qCmd[12:15] = 0.0  # Right arm joints 4-6

    def update_control(self):
        self.qdCmd[0:3] = self.kp * self.qdCmd[0:3]
        self.data.ctrl[:3] = self.qdCmd[:3]
        self.data.ctrl[3:] = self.qCmd[3:]

    def render_ui(self):
        current_time = time.time()

        if current_time - self.last_render_time >= self.render_interval:
            self.viewer.cam = self.camera
            self.viewer._overlay[mujoco.mjtGridPos.mjGRID_TOPLEFT] = [
                f"Time: {self.data.time:.3f} sec",
                "",
            ]
            self.viewer._overlay[mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT] = [
                "=== CHASSIS MOVEMENT (INCREMENTAL) ===\n"
                "Forward/Backward   (+x/-x): press Home/End\n"
                "Leftward/Rightward (+y/-y): press Delete/Page Down\n"
                "Rotate CCW/CW      (+z/-z): press Insert/Page Up\n"
                "\n=== LEFT ARM CONTROLS ===\n"
                "Joint1: q(+)/a(-)    Joint2: w(+)/s(-)    Joint3: e(+)/d(-)\n"
                "\n=== RIGHT ARM CONTROLS ===\n"
                "Joint1: u(+)/j(-)    Joint2: i(+)/k(-)    Joint3: o(+)/l(-)\n"
                f"\ncommand: Chassis Vel: [{self.qdCmd[0]:.2f}, {self.qdCmd[1]:.2f}, {self.qdCmd[2]:.2f}]\n"
                f"feedback: Chassis Vel: [{self.qdFb[0]:.2f}, {self.qdFb[1]:.2f}, {self.qdFb[2]:.2f}]\n"
                f"Left Arm: [{self.qCmd[3]:.2f}, {self.qCmd[4]:.2f}, {self.qCmd[5]:.2f}]\n"
                f"Right Arm: [{self.qCmd[9]:.2f}, {self.qCmd[10]:.2f}, {self.qCmd[11]:.2f}]",
                "",
            ]

            self.viewer.render()
            self.last_render_time = current_time

    def run(self):
        """Main control loop for XLeRobot keyboard control."""
        print("Starting XLeRobot keyboard Controller...")

        while self.viewer.is_alive:
            self.update_feedback()
            self.update_keyboards()
            self.update_reference()
            self.update_control()
            mujoco.mj_step(self.model, self.data)
            self.render_ui()
            time.sleep(0.002)

        self.cleanup()

    def cleanup(self):
        self.viewer.close()
        print("XLeRobot controller stopped.")


def main():
    try:
        mjcf_path = "../scene.xml"
        controller = XLeRobotController(mjcf_path)
        controller.run()
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt, shutting down...")


if __name__ == "__main__":
    main()
