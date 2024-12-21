import cv2
import threading

class CameraManager:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(self.camera_index)
        self.lock = threading.Lock()
        self.running = False
        self.frame = None
        self.thread = None

    def _capture_frames(self):
        while self.running:
            with self.lock:
                ret, frame = self.cap.read()
                if ret:
                    self.frame = frame

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._capture_frames)
            self.thread.start()

    def get_frame(self):
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
        return None

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        self.cap.release()
        
 