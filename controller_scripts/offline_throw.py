import ollama
import speech_recognition as sr
import pyttsx3
import re
import json
import time
from pathlib import Path
from lerobot.common.robots import make_robot_from_config
from lerobot.common.robots.so101_follower.so101_follower import SO101FollowerConfig

# Initialize components
class VoiceControlledRobot:
    def __init__(self):
        # Initialize voice components
        self.cline = ollama.Client()
        self.engine = pyttsx3.init()
        self.recognizer = sr.Recognizer()
        
        # Robot control parameters
        self.json_path = Path("/home/chi/lerobot/lerobot/robot_control_sequence_state_ep1.json")
        self.step_delay = 0.001
        self.flag = None  # Will be set based on voice input
        
        # Initialize robot (connection happens when needed)
        self.robot_cfg = SO101FollowerConfig(
            port="/dev/ttyACM0",
            id="follower_arm_1",
            use_degrees=True
        )
        self.robot = None

    def print_mic_device_index(self):
        """List available microphone devices"""
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print(f"device_index: {index}, device_name: {name}")

    def text_to_speech(self, text):
        """Convert text to speech and play"""
        self.engine.say(text)
        self.engine.runAndWait()

    def speech_to_text(self, model="whisper", language="en", device_index=3, filename="output.txt"):
        """Convert speech to text using specified model"""
        with sr.Microphone(device_index=device_index) as source:
            print("Adjusting for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Starting to listen...")

            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)

                if model == "google":
                    text = self.recognizer.recognize_google(audio, language="en-US")
                elif model == "sphinx":
                    text = self.recognizer.recognize_sphinx(audio, language=language)
                elif model == "whisper":
                    text = self.recognizer.recognize_whisper(audio, language="en")
                else:
                    raise ValueError(f"Unsupported model: {model}")

                print(f"You said: {text}")
                self.text_to_txt(text, filename=filename)
                return text

            except sr.UnknownValueError:
                print("Could not understand audio")
                return None
            except sr.RequestError as e:
                print(f"Request error: {e}")
                return None
            except sr.WaitTimeoutError:
                print("Listening timed out, no speech detected")
                return None

    def text_to_txt(self, text, filename="output.txt"):
        """Save text to file"""
        with open(filename, "a", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")

    def parse_ollama_response(self, response):
        """Extract color and motion from Ollama response"""
        try:
            # Try to parse as JSON
            data = json.loads(response)
            color = data.get("color", "").lower()
            motion = data.get("motion", "")
            return color, motion
        except json.JSONDecodeError:
            # If not JSON, try to find color in text
            color = ""
            if "red" in response.lower():
                color = "red"
            elif "blue" in response.lower():
                color = "blue"
            return color, ""

    def ollama_reply(self, prompt, model="roboflow"):
        """Generate response using Ollama model"""
        llm_response_object = self.cline.generate(model=model, prompt=prompt)
        cleaned_response = re.sub(r'<think>.*?</think>', '', llm_response_object.response, flags=re.DOTALL)
        
        # Parse the response to get color and motion
        color, motion = self.parse_ollama_response(cleaned_response)
        
        if color:
            self.flag = color
            print(f"Detected color: {color}, setting flag")
        
        return cleaned_response

    def run_robot_sequence(self):
        """Execute the robot movement sequence based on the flag"""
        if not self.flag:
            print("No flag set, cannot run robot sequence")
            return False

        # Load trajectory data
        with open(self.json_path, "r") as f:
            data = json.load(f)

        joint_sequence = data["joint_sequence"]

        # Apply offset based on flag
        if self.flag == "red":
            offset = 12
        elif self.flag == "blue":
            offset = -15
        else:
            offset = 0

        # Define target joint positions
        # homing position
        target_joint_pos = {
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": -90.0,
            "elbow_flex.pos": -90.0,
            "wrist_flex.pos": 0.0,
            "wrist_roll.pos": 0.0,
            "gripper.pos": 0.0,
        }

        # Apply offsets to joint sequence out of calibration
        joint_sequence = [
            {k: v + offset if k == "shoulder_pan.pos" else v for k, v in joint_cmd.items()}
            for joint_cmd in joint_sequence
        ]

        joint_sequence = [
            {k: -1*(v) if k == "shoulder_lift.pos" else v for k, v in joint_cmd.items()}
            for joint_cmd in joint_sequence
        ]

        joint_sequence = [
            {k: 1*(v-90) if k == "elbow_flex.pos" else v for k, v in joint_cmd.items()}
            for joint_cmd in joint_sequence
        ]

        joint_sequence = [
            {k: 0*(v + 90) if k == "wrist_roll.pos" else v for k, v in joint_cmd.items()}
            for joint_cmd in joint_sequence
        ]

        # Connect to robot
        if not self.robot:
            self.robot = make_robot_from_config(self.robot_cfg)
            self.robot.connect()

        try:
            print(f"Loading {len(joint_sequence)} trajectory points, starting execution...")
            joint_sequence = joint_sequence[:10]  # Take first 10 steps for demo
            
            # Execute main sequence
            for i, joint_cmd in enumerate(joint_sequence):
                self.robot.send_action(joint_cmd)
                print(f"Step {i + 1}/{len(joint_sequence)}:", joint_cmd)
                time.sleep(self.step_delay)
            print("Trajectory playback complete!")

            time.sleep(1.0)

            # Interpolate to target joint position
            current_joint_pos = joint_sequence[-1].copy()
            alpha = 0.1  # interpolation factor
            tolerance = 0.5  # degrees

            while True:
                interpolated_joint_pos = {}
                max_diff = 0
                for joint, target in target_joint_pos.items():
                    current = current_joint_pos[joint]
                    new_val = current + (target - current) * alpha
                    interpolated_joint_pos[joint] = new_val
                    max_diff = max(max_diff, abs(new_val - target))
                
                self.robot.send_action(interpolated_joint_pos)
                print("Interpolating:", interpolated_joint_pos)
                time.sleep(self.step_delay)
                current_joint_pos = interpolated_joint_pos
                
                if max_diff < tolerance:
                    break

            print("Reached target joint position!")
            return True

        except Exception as e:
            print(f"Error during robot operation: {e}")
            return False
        finally:
            if self.robot:
                self.robot.disconnect()
                self.robot = None

    def main_loop(self):
        """Main interaction loop"""
        # Optional: Print microphone device indices
        self.print_mic_device_index()
        
        while True:
            print("\nListening for command...")
            user_input = self.speech_to_text(model="whisper", device_index=9, language="en")
            
            if user_input:
                print(f"Processing: {user_input}")
                reply = self.ollama_reply(user_input)
                
                print(f"Ollama response: {reply}")
                self.text_to_speech(reply)
                
                # If we detected a color, run the robot sequence
                if self.flag:
                    print(f"Executing robot sequence for {self.flag}")
                    success = self.run_robot_sequence()
                    if success:
                        self.text_to_speech(f"Completed {self.flag} box movement")
                    else:
                        self.text_to_speech("Robot operation failed")
            else:
                print("No valid input captured, listening again...")

            # Ask to continue
            choice = input("Continue listening? (y/n): ")
            if choice.lower() != 'y':
                break

if __name__ == "__main__":
    vcr = VoiceControlledRobot()
    vcr.main_loop()