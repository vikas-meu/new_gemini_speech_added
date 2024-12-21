from camera_manager import CameraManager
import cv2
import mediapipe as mp
import serial
import time
import speech_recognition as sr
import pyttsx3
import threading
from headtracker import HeadTrackingSystem
import pyzbar.pyzbar as pyzbar  # For QR and Barcode detection
from ultralytics import YOLO  # For object detection
import google.generativeai as genai

# Serial communication with Arduino
arduino = serial.Serial('COM12', 9600)
time.sleep(2)

# Configure the Gemini API
genai.configure(api_key="AIzaSyB4_jlX2uQd3LLlS0WVVp-c1jiY9B5F6Jw")
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Mediapipe Hand and Face detection
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)

# Speech recognition and text-to-speech setup
recognizer = sr.Recognizer()
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust the speaking rate

# Gemini speech-related functions
def ask_gemini(question):
    """Send a question to Gemini and get the response."""
    try:
        response = gemini_model.generate_content(question)
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

def recognize_speech():
    """Recognize speech input from the user."""
    with sr.Microphone() as mic:
        print("Listening for your question...")
        try:
            recognizer.adjust_for_ambient_noise(mic, duration=0.2)
            audio = recognizer.listen(mic, timeout=5)
            question = recognizer.recognize_google(audio)
            print(f"You said: {question}")
            return question
        except sr.UnknownValueError:
            return "Sorry, I didn't understand that. Please try again."
        except sr.RequestError as e:
            return f"Could not request results; {e}"

# General TTS function
def speak(text):
    """Convert text to speech."""
    engine.say(text)
    engine.runAndWait()

# Check for predefined speech commands
def check_keywords(text):
    """Check if text contains any speech command keywords."""
    for command, details in speech_commands.items():
        if any(keyword in text for keyword in details["keywords"]):
            return command, details["response"]
    return None, None

def confirm_shutdown():
    """Ask for confirmation to shut down."""
    speak("We are terminating the program. See you soon sir.")
    time.sleep(1)
    arduino.write(f"terminate_flag".encode())
    with sr.Microphone() as mic:
        try:
            recognizer.adjust_for_ambient_noise(mic, duration=0.2)
            print("Listening for confirmation...")
            audio = recognizer.listen(mic, timeout=5)
            confirmation = recognizer.recognize_google(audio).lower()
            if "ok" in confirmation:
                speak("Shutting down. Goodbye!")
                return True
            elif "no" in confirmation:
                speak("Cancellation confirmed. Resuming operations.")
                return False
        except sr.WaitTimeoutError:
            print("Confirmation timed out. Resuming operations.")
        except sr.UnknownValueError:
            print("Could not understand the confirmation.")
            arduino.write(f"terminate_flag".encode())
    return True

# Initialize shared camera
camera = CameraManager(camera_index=0)
camera.start()

# Load YOLO model
model = YOLO("yolov5su.pt")

# Keywords to detect
speech_commands = {
    "what_can_you_see": {
        "keywords": ["what can you see", "what is in front of you"],
        "response": "Analyzing the scene. Please wait.",
    },
    "shutdown": {
        "keywords": ["shutdown", "terminate", "close program", "exit"],
        "response": "Shutting down request received. Confirming now.",
    },
    "gemini": {
        "keywords": ["hey buddy", "buddy", "i have a question", "question here", "i want to ask you", "buddy i have a question", "hey", "hey"],
        "response": "Sure, please ask your question for Gemini.",
    },
    "love": {
        "keywords": ["i love you", "i love", "love u", "love you", "i like you"],
        "response": "That's so sweet! Sending love your way."
    },
    "start_up": {
        "keywords": ["wake up", "wake up buddy", "let's start buddy", "let's build something"],
        "response": "Always at your service sir.. what you want to build today??."
    },
    "good_bye": {
        "keywords": ["bye", "goodbye", "see you", "see ya"],
        "response": "Goodbye! Take care."
    },
    "warn": {
        "keywords": ["battery low", "get away", "just backoff", "we are in danger"],
        "response": "Warning received. Be cautious."
    },
    "bye": {
        "keywords": ["good night", "feeling sleepy", "i need rest", "going to bed"],
        "response": "ok have a sweet dreams, good night."
    },
    "good_morning": {
        "keywords": ["good morning", "good morning buddy", "morning buddy", "morning"],
        "response": "hi,, good morning, sir  what't we have to start to build today ."
    },
    "listen": {
        "keywords": ["buddy", "are you there", "hey are you there", "listen to me"],
        "response": "yes sir,  always listening for your commands."
    },
    "llo": {
        "keywords": ["what are you looking at", "what", "hey what are you up", "look here"],
        "response": "i am looking at you sir, for your commands"
    },
     
    "read_data": {
        "keywords": ["please read this data","read this","scan and tell"],
        "response": "Scanning the visible QR or barcode now."
    },
    "nikhil": {
        "keywords": ["nikhil ","nikhil","who is nikhil"],
        "response": "land a very very very big laand"
    },
    # Add other commands as necessary
}

# Initialize head-tracking system
head_tracking_system = HeadTrackingSystem('COM5')
head_tracking_active = True
head_tracking_thread = threading.Thread(target=head_tracking_system.start)
head_tracking_thread.start()

# Main loop
while True:
    frame = camera.get_frame()
    if frame is None:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # QR/Barcode detection
    decoded_objects = pyzbar.decode(frame)
    if decoded_objects:
        for obj in decoded_objects:
            data = obj.data.decode("utf-8")
            print(f"Detected: {data}")
            speak(f"Detected data: {data}")
            cv2.rectangle(frame, (obj.rect.left, obj.rect.top),
                          (obj.rect.left + obj.rect.width, obj.rect.top + obj.rect.height), (0, 255, 0), 2)
            cv2.putText(frame, data, (obj.rect.left, obj.rect.top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Mediapipe processing
    if not head_tracking_active:
        results_face = face_detection.process(rgb_frame)

    # Speech recognition
    with sr.Microphone() as mic:
        try:
            recognizer.adjust_for_ambient_noise(mic, duration=0.2)
            print("Listening...")
            audio = recognizer.listen(mic, timeout=2)
            text = recognizer.recognize_google(audio).lower()
            print(f"Recognized: {text}")

            # Check for speech commands
            command, response = check_keywords(text)
            if command:
                print(f"Command recognized: {command}")
                arduino.write(f"{command}\n".encode())
                speak(response)

            if command == "what_can_you_see":
                fresh_frame = camera.get_frame()  # Capture a new frame from the camera
                if fresh_frame is not None:
                    results = model.predict(source=fresh_frame, show=False)
                    objects_detected = []
                    for result in results:
                         for box in result.boxes:
                           class_name = model.names[int(box.cls[0])]
                           confidence = box.conf[0] * 100
                           objects_detected.append(f"{class_name} ({confidence:.2f}%)")
                    if objects_detected:
                             detected_objects_str = ", ".join(objects_detected)
                             speak(f"I can see: {detected_objects_str}")
                    else:
                      speak("I can't detect any recognizable objects right now.")
               
            if command == "gemini":
                    question = recognize_speech()
                    if question:
                        answer = ask_gemini(question)
                        print(f"Gemini's Answer: {answer}")
                        speak(answer)

            if command == "shutdown":
                    if confirm_shutdown():
                        break

        except sr.WaitTimeoutError:
            print("Speech recognition timed out.")
        except sr.UnknownValueError:
            print("Could not understand the audio.")

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
camera.stop()
cv2.destroyAllWindows()
arduino.close()
head_tracking_system.stop()
