import cv2
from cvzone.HandTrackingModule import HandDetector
import speech_recognition as sr
from camera_manager import CameraManager  # Ensure this is implemented correctly

class FingerCount:
    def __init__(self, camera_index=0):
        self.detector = HandDetector(detectionCon=0.8, maxHands=1)
        self.camera = CameraManager(camera_index)  # Initialize CameraManager with index
        self.count = 0
        self.stop_signal = False
        self.recognizer = sr.Recognizer()

    def listen_for_stop(self):
        """Listen for the 'ok stop' command."""
        while not self.stop_signal:
            try:
                with sr.Microphone() as mic:
                    self.recognizer.adjust_for_ambient_noise(mic, duration=0.2)
                    audio = self.recognizer.listen(mic, timeout=3)
                    text = self.recognizer.recognize_google(audio).lower()
                    print(f"Recognized (finger counting): {text}")
                    if "ok stop" in text:
                        self.stop_signal = True
                        print("Stopping finger count...")
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                print(f"Speech recognition error: {e}")

    def count_fingers(self):
        """Start finger counting."""
        print("Starting finger count...")
        self.stop_signal = False

        # Start a thread for speech recognition to listen for "ok stop"
 

        while not self.stop_signal:
            frame = self.camera.get_frame()  # Get frame from CameraManager
            frame = cv2.flip(frame, 1)

            # Detect hands
            hands, img = self.detector.findHands(frame)

            if hands:
                lmList = hands[0]["lmList"]  # Extract landmarks for the detected hand
                finger_up = self.detector.fingersUp(lmList)

                # Count fingers
                self.count = sum(finger_up)
                #cv2.putText(frame, f'Finger count: {self.count}', (20, 460), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

            # Display the frame
            #cv2.imshow("Finger Count Detection", frame)

            # Exit on pressing 'k' key
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
         
        self.camera.release()  # Ensure the camera is released
        cv2.destroyAllWindows()
        return self.count 
