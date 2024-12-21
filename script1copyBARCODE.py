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

# Serial communication with Arduino
arduino = serial.Serial('COM12', 9600)
time.sleep(2)

# Mediapipe Hand and Face detection
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face.FaceDetection(min_detection_confidence=0.7)

# Speech recognition and text-to-speech setup
recognizer = sr.Recognizer()
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust the speaking rate

def speak(text):
    """Convert text to speech."""
    engine.say(text)
    engine.runAndWait()

def check_keywords(text):
    """Check if text contains any speech command keywords."""
    for command, details in speech_commands.items():
        if any(keyword in text for keyword in details["keywords"]):
            return command, details["response"]
    return None, None

# Initialize shared camera
camera = CameraManager(camera_index=0)
camera.start()

# Keywords to detect
speech_commands = {
    "hi_hello": {
        "keywords": ["hi", "hello", "whatsapp", "hola", "ohayo"],
        "response": "hi sir"
    },
    "love": {
        "keywords": ["i love you", "i love", "i love u", "heart", "i like you"],
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
        "keywords": ["please read this data", "read this", "scan and tell"],
        "response": "Scanning the visible QR or barcode now."
    },
}

# Initialize head-tracking system
head_tracking_system = HeadTrackingSystem('COM4')
head_tracking_active = True
head_tracking_thread = threading.Thread(target=head_tracking_system.start)
head_tracking_thread.start()

# Main loop
while True:
    frame = camera.get_frame()
    if frame is None:
        continue

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
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

                # Special case for "read_data"
                if command == "read_data" and decoded_objects:
                    for obj in decoded_objects:
                        data = obj.data.decode("utf-8")
                        speak(f"Visible data: {data}")
        except sr.WaitTimeoutError:
            print("Speech recognition timed out.")
        except sr.UnknownValueError:
            print("Could not understand the audio.")

    # Display the frame
    #cv2.imshow("Interactive System", frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
camera.stop()
cv2.destroyAllWindows()
arduino.close()
head_tracking_system.stop()
