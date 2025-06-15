from oculus_reader.FPS_counter import FPSCounter
from oculus_reader.buttons_parser import parse_buttons
import numpy as np
import threading
import time
import os
from ppadb.client import Client as AdbClient
import sys

def eprint(*args, **kwargs):
    RED = "\033[1;31m"  
    sys.stderr.write(RED)
    print(*args, file=sys.stderr, **kwargs)
    RESET = "\033[0;0m"
    sys.stderr.write(RESET)

class OculusReader:
    def __init__(self,
            ip_address=None,
            port = 5555,
            APK_name='com.rail.oculus.teleop',
            print_FPS=False,
            run=True,
            print_positions=False  # New parameter to enable position printing
        ):
        self.running = False
        self.last_transforms = {}
        self.last_buttons = {}
        self.last_pinch_state = {'l': False, 'r': False}  # Track previous pinch states
        self._lock = threading.Lock()
        self.tag = 'wE9ryARX'

        self.ip_address = ip_address
        self.port = port
        self.APK_name = APK_name
        self.print_FPS = print_FPS
        self.print_positions = print_positions  # Store the parameter
        if self.print_FPS:
            self.fps_counter = FPSCounter()

        self.device = self.get_device()
        self.install(verbose=False)
        if run:
            self.run()

    def __del__(self):
        self.stop()

    def run(self):
        self.running = True
        self.device.shell('am start -n "com.rail.oculus.teleop/com.rail.oculus.teleop.MainActivity" -a android.intent.action.MAIN -c android.intent.category.LAUNCHER')
        self.thread = threading.Thread(target=self.device.shell, args=("logcat -T 0", self.read_logcat_by_line))
        self.thread.start()
        
        # Start position printing thread if enabled
        if self.print_positions:
            self.position_thread = threading.Thread(target=self.print_positions_thread)
            self.position_thread.daemon = True  # Make thread exit when main program exits
            self.position_thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()

    def get_network_device(self, client, retry=0):
        try:
            client.remote_connect(self.ip_address, self.port)
        except RuntimeError:
            os.system('adb devices')
            client.remote_connect(self.ip_address, self.port)
        device = client.device(self.ip_address + ':' + str(self.port))

        if device is None:
            if retry==1:
                os.system('adb tcpip ' + str(self.port))
            if retry==2:
                eprint('Make sure that device is running and is available at the IP address specified as the OculusReader argument `ip_address`.')
                eprint('Currently provided IP address:', self.ip_address)
                eprint('Run `adb shell ip route` to verify the IP address.')
                exit(1)
            else:
                self.get_device(client=client, retry=retry+1)
        return device

    def get_usb_device(self, client):
        try:
            devices = client.devices()
        except RuntimeError:
            os.system('adb devices')
            devices = client.devices()
        for device in devices:
            if device.serial.count('.') < 3:
                return device
        eprint('Device not found. Make sure that device is running and is connected over USB')
        eprint('Run `adb devices` to verify that the device is visible.')
        exit(1)

    def get_device(self):
        # Default is "127.0.0.1" and 5037
        client = AdbClient(host="127.0.0.1", port=5037)
        if self.ip_address is not None:
            return self.get_network_device(client)
        else:
            return self.get_usb_device(client)

    def install(self, APK_path=None, verbose=True, reinstall=False):
        try:
            installed = self.device.is_installed(self.APK_name)
            if not installed or reinstall:
                if APK_path is None:
                    APK_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'APK', 'teleop-debug.apk')
                success = self.device.install(APK_path, test=True, reinstall=reinstall)
                installed = self.device.is_installed(self.APK_name)
                if installed and success:
                    print('APK installed successfully.')
                else:
                    eprint('APK install failed.')
            elif verbose:
                print('APK is already installed.')
        except RuntimeError:
            eprint('Device is visible but could not be accessed.')
            eprint('Run `adb devices` to verify that the device is visible and accessible.')
            eprint('If you see "no permissions" next to the device serial, please put on the Oculus Quest and allow the access.')
            exit(1)

    def uninstall(self, verbose=True):
        try:
            installed = self.device.is_installed(self.APK_name)
            if installed:
                success = self.device.uninstall(self.APK_name)
                installed = self.device.is_installed(self.APK_name)
                if not installed and success:
                    print('APK uninstall finished.')
                    print('Please verify if the app disappeared from the list as described in "UNINSTALL.md".')
                    print('For the resolution of this issue, please follow https://github.com/Swind/pure-python-adb/issues/71.')
                else:
                    eprint('APK uninstall failed')
            elif verbose:
                print('APK is not installed.')
        except RuntimeError:
            eprint('Device is visible but could not be accessed.')
            eprint('Run `adb devices` to verify that the device is visible and accessible.')
            eprint('If you see "no permissions" next to the device serial, please put on the Oculus Quest and allow the access.')
            exit(1)

    @staticmethod
    def process_data(string):
        try:
            transforms_string, buttons_string = string.split('&')
        except ValueError:
            return None, None
        split_transform_strings = transforms_string.split('|')
        transforms = {}
        for pair_string in split_transform_strings:
            transform = np.empty((4,4))
            pair = pair_string.split(':')
            if len(pair) != 2:
                continue
            left_right_char = pair[0] # is r or l
            transform_string = pair[1]
            values = transform_string.split(' ')
            c = 0
            r = 0
            count = 0
            for value in values:
                if not value:
                    continue
                transform[r][c] = float(value)
                c += 1
                if c >= 4:
                    c = 0
                    r += 1
                count += 1
            if count == 16:
                transforms[left_right_char] = transform
        buttons = parse_buttons(buttons_string)
        return transforms, buttons

    def extract_data(self, line):
        output = ''
        if self.tag in line:
            try:
                output += line.split(self.tag + ': ')[1]
            except ValueError:
                pass
        return output

    def get_transformations_and_buttons(self):
        with self._lock:
            return self.last_transforms, self.last_buttons

    def read_logcat_by_line(self, connection):
        file_obj = connection.socket.makefile()
        while self.running:
            try:
                line = file_obj.readline().strip()
                data = self.extract_data(line)
                if data:
                    transforms, buttons = OculusReader.process_data(data)
                    with self._lock:
                        self.last_transforms, self.last_buttons = transforms, buttons
                        
                        # Check for pinch gestures
                        if buttons:
                            for hand in ['l', 'r']:
                                if hand in buttons:
                                    # Check if 'pinch' is in the buttons dictionary
                                    current_pinch = buttons[hand].get('pinch', False)
                                    # Detect pinch activation (transition from False to True)
                                    if current_pinch and not self.last_pinch_state[hand]:
                                        hand_name = "Left" if hand == 'l' else "Right"
                                        print(f"{hand_name} hand pinch detected")
                                    # Update the pinch state
                                    self.last_pinch_state[hand] = current_pinch
                        
                    if self.print_FPS:
                        self.fps_counter.getAndPrintFPS()
            except UnicodeDecodeError:
                pass
        file_obj.close()
        connection.close()

    def visualize_transforms(self, duration=None, scale=0.1):
        """
        Visualize the controller transforms in real-time 3D space.
        
        Args:
            duration: Time in seconds to run visualization (None for infinite)
            scale: Scale factor for orientation vectors
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.animation as animation

        # Create the figure and 3D axis
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Initialize empty plots for left and right controllers
        left_pos, = ax.plot([], [], [], 'bo', markersize=10, label='Left Controller')  # Blue dot
        right_pos, = ax.plot([], [], [], 'ro', markersize=10, label='Right Controller')  # Red dot
        
        # Initialize orientation vectors (x, y, z axes for each controller) with RGB colors
        # Red for x-axis, Green for y-axis, Blue for z-axis
        axis_colors = ['r', 'g', 'b']  # RGB for xyz
        left_axes = [ax.plot([], [], [], color=c)[0] for c in axis_colors]
        right_axes = [ax.plot([], [], [], color=c)[0] for c in axis_colors]
        
        # Set axis limits and labels
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Oculus Controller Transforms')
        
        # Add legend with additional entries for axes
        ax.plot([], [], [], 'r-', label='X axis')
        ax.plot([], [], [], 'g-', label='Y axis')
        ax.plot([], [], [], 'b-', label='Z axis')
        ax.legend()
        
        # Set initial view angle
        ax.view_init(elev=30, azim=45)
        
        start_time = time.time()
        
        def extract_transform_data(transform_matrix):
            """Extract position and orientation from 4x4 transform matrix"""
            if transform_matrix is None:
                return None, None
            
            # Position is the translation part (first 3 elements of 4th column)
            position = transform_matrix[:3, 3]
            
            # Orientation axes (first 3 columns, each representing x, y, z direction)
            x_axis = transform_matrix[:3, 0] * scale
            y_axis = transform_matrix[:3, 1] * scale
            z_axis = transform_matrix[:3, 2] * scale
            
            return position, (x_axis, y_axis, z_axis)
        
        def update(frame):
            """Update function for animation"""
            transforms, _ = self.get_transformations_and_buttons()
            
            if duration and time.time() - start_time > duration:
                animation_obj.event_source.stop()
                plt.close(fig)
                return
            
            # Update left controller
            if 'l' in transforms:
                position, axes = extract_transform_data(transforms['l'])
                if position is not None:
                    left_pos.set_data([position[0]], [position[1]])
                    left_pos.set_3d_properties([position[2]])
                    
                    # Update orientation vectors (RGB for XYZ axes)
                    for i, axis in enumerate(axes):
                        left_axes[i].set_data([position[0], position[0]+axis[0]], 
                                              [position[1], position[1]+axis[1]])
                        left_axes[i].set_3d_properties([position[2], position[2]+axis[2]])
            
            # Update right controller
            if 'r' in transforms:
                position, axes = extract_transform_data(transforms['r'])
                if position is not None:
                    right_pos.set_data([position[0]], [position[1]])
                    right_pos.set_3d_properties([position[2]])
                    
                    # Update orientation vectors (RGB for XYZ axes)
                    for i, axis in enumerate(axes):
                        right_axes[i].set_data([position[0], position[0]+axis[0]], 
                                               [position[1], position[1]+axis[1]])
                        right_axes[i].set_3d_properties([position[2], position[2]+axis[2]])
            
            # Adjust axis limits dynamically if controllers move outside current view
            if 'l' in transforms or 'r' in transforms:
                all_positions = []
                if 'l' in transforms:
                    all_positions.append(transforms['l'][:3, 3])
                if 'r' in transforms:
                    all_positions.append(transforms['r'][:3, 3])
                
                if all_positions:
                    positions = np.vstack(all_positions)
                    min_pos = positions.min(axis=0) - scale
                    max_pos = positions.max(axis=0) + scale
                    
                    margin = 0.3  # Add extra margin
                    ax.set_xlim([min_pos[0]-margin, max_pos[0]+margin])
                    ax.set_ylim([min_pos[1]-margin, max_pos[1]+margin])
                    ax.set_zlim([min_pos[2]-margin, max_pos[2]+margin])
            
            return left_pos, right_pos, *left_axes, *right_axes
        
        # Create animation - target 60fps
        interval = 1000/60  # milliseconds
        animation_obj = animation.FuncAnimation(
            fig, update, interval=interval, blit=True, cache_frame_data=False)
        
        plt.tight_layout()
        plt.show()

    def print_positions_thread(self):
        """Print controller positions (X,Y,Z) every 0.1 seconds"""
        while self.running:
            transforms, _ = self.get_transformations_and_buttons()
            
            if 'l' in transforms:
                pos_l = transforms['l'][:3, 3]  # Extract position (first 3 elements of 4th column)
                print(f"Left controller: X={pos_l[0]:.3f}, Y={pos_l[1]:.3f}, Z={pos_l[2]:.3f}")
            
            if 'r' in transforms:
                pos_r = transforms['r'][:3, 3]  # Extract position (first 3 elements of 4th column)
                print(f"Right controller: X={pos_r[0]:.3f}, Y={pos_r[1]:.3f}, Z={pos_r[2]:.3f}")
            
            time.sleep(0.1)  # Sleep for 0.1 seconds


def main():
    oculus_reader = OculusReader(print_positions=True)  # Enable position printing

    try:
        # Start visualization
        oculus_reader.visualize_transforms()
    except KeyboardInterrupt:
        print("Visualization stopped by user")
    finally:
        oculus_reader.stop()


if __name__ == '__main__':
    main()
