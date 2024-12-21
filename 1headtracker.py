from camera_manager import CameraManager
import cv2
import mediapipe as mp
import pyfirmata
import time


class HeadTrackingSystem:
    def __init__(self, port):
        self.port = port
        self.board = pyfirmata.Arduino(self.port)
        self.camera = CameraManager(camera_index=0)
        self.running = False

        # Servo initialization
        self.servo_x = self.board.get_pin('d:9:s')
        self.servo_y = self.board.get_pin('d:10:s')
        self.servo_x.write(90)
        self.servo_y.write(90)

        # Mediapipe initialization
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def smooth_servo_move(self, servo, start_angle, end_angle):
        step = 1 if end_angle > start_angle else -1
        for angle in range(int(start_angle), int(end_angle), step):
            servo.write(angle)
            time.sleep(0.005)

    def start(self):
        self.camera.start()
        self.running = True
        while self.running:
            frame = self.camera.get_frame()
            if frame is None:
                continue

            img = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(img_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Extract landmarks
                    nose_tip = face_landmarks.landmark[1]
                    img_height, img_width, _ = img.shape
                    cx_nose = int(nose_tip.x * img_width)
                    cy_nose = int(nose_tip.y * img_height)

                    # Map to servo positions
                    target_x = 90 + (cx_nose - img_width // 2) // 10
                    target_y = 90 + (cy_nose - img_height // 2) // 10

                    self.smooth_servo_move(self.servo_x, self.servo_x.read(), target_x)
                    self.smooth_servo_move(self.servo_y, self.servo_y.read(), target_y)

    
                if cv2.waitKey(1) & 0xFF == ord('q'):
                  break
    
    def stop(self):
        self.running = False
        self.camera.stop()
        self.board.exit()
