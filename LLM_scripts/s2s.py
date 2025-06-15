import ollama
import speech_recognition as sr
import pyttsx3
import re

# Initialize Ollama client
cline = ollama.Client()

# Initialize text-to-speech engine
engine = pyttsx3.init()

def print_mic_device_index():
    """List available microphone devices"""
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"device_index: {index}, device_name: {name}")

def text_to_speech(text):
    """Convert text to speech and play"""
    engine.say(text)
    engine.runAndWait()

def speech_to_text(model="whisper", language="en", device_index=3, filename="output.txt"):
    """Convert speech to text using specified model"""
    r = sr.Recognizer()
    with sr.Microphone(device_index=device_index) as source:
        print("Adjusting for ambient noise...")
        r.adjust_for_ambient_noise(source, duration=1)
        print("Starting to listen...")

        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=10)  # Set timeout and phrase time limit

            if model == "google":
                text = r.recognize_google(audio, language="en-US")
            elif model == "sphinx":
                text = r.recognize_sphinx(audio, language=language)
            elif model == "whisper":
                text = r.recognize_whisper(audio, language="en")
            else:
                raise ValueError(f"Unsupported model: {model}")

            print(f"You said: {text}")
            text_to_txt(text, filename=filename)
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

def text_to_txt(text, filename="output.txt"):
    """Save text to file"""
    with open(filename, "a", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")

def ollama_reply(prompt, model="roboflow"):
    """Generate response using Ollama model"""
    llm_response_object = cline.generate(model=model, prompt=prompt)

    cleaned_response = re.sub(r'<think>.*?</think>', '', llm_response_object.response, flags=re.DOTALL)

    # ... (rest of your ollama_reply function) ...
    return cleaned_response # Or whatever your function returns


if __name__ == "__main__":
    # Optional: Print microphone device indices
    print_mic_device_index()
    while True:
        user_input = speech_to_text(model="whisper", device_index=9, language="en")
        if user_input:
            reply = ollama_reply(user_input)
            print(f"Ollama response: {reply}")
            text_to_speech(reply)
        else:
            print("No valid input captured, listening again...")

        # Ask to continue
        choice = input("Continue listening? (y/n): ")
        if choice.lower() != 'y':
            break 