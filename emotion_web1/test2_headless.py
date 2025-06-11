import cv2
import requests
import sounddevice as sd
import scipy.io.wavfile as wav
import io
import threading
import time
import logging
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput
import numpy as np
import os
from queue import Queue
import concurrent.futures

# Disable display for headless operation
os.environ['DISPLAY'] = ':0'

FLASK_SERVER = 'http://localhost:5000'
AUDIO_RATE = 16000
AUDIO_DURATION = 3  # 3 seconds
SEND_INTERVAL = 3  # Interval between sending data in seconds
IMAGE_SEND_FREQUENCY = 0.08  # Send image every 0.08 seconds (12.5 FPS) for real-time

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Shared variables for threads
frame_lock = threading.Lock()
latest_frame = None
trip_id_global = None
running = False

# Queues for better thread communication
image_queue = Queue(maxsize=50)  # Buffer for images to send
audio_queue = Queue(maxsize=10)  # Buffer for audio to send

logging.basicConfig(level=logging.INFO)

def send_image_worker():
    """Dedicated worker thread for sending images"""
    while running:
        try:
            if not image_queue.empty():
                face_img, trip_id = image_queue.get(timeout=0.1)
                _, img_encoded = cv2.imencode('.jpg', face_img)
                files = {'file': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}
                data = {'trip_id': trip_id}
                
                for attempt in range(2):
                    try:
                        res = requests.post(f'{FLASK_SERVER}/upload_image', files=files, data=data, timeout=2)
                        if res.ok:
                            logging.info("[Image] Sent successfully")
                            break
                        else:
                            logging.error(f"[Image] Server error (Attempt {attempt + 1}): {res.status_code}")
                    except Exception as e:
                        logging.error(f"[Image] Connection error (Attempt {attempt + 1}): {e}")
                    time.sleep(0.5)
                    
                image_queue.task_done()
            else:
                time.sleep(0.01)  # Short sleep when queue is empty
        except Exception as e:
            logging.error(f"Error in image worker: {e}")
            time.sleep(0.1)

def send_audio_worker():
    """Dedicated worker thread for sending audio"""
    while running:
        try:
            if not audio_queue.empty():
                audio_data, trip_id = audio_queue.get(timeout=0.1)
                wav_buffer = io.BytesIO()
                wav.write(wav_buffer, AUDIO_RATE, audio_data)
                wav_buffer.seek(0)
                files = {'file': ('audio.wav', wav_buffer, 'audio/wav')}
                
                for attempt in range(3):
                    try:
                        res = requests.post(f'{FLASK_SERVER}/upload_audio', files=files, timeout=10)
                        if res.ok:
                            logging.info("[Audio] Sent successfully and processed by server")
                            break
                        else:
                            logging.error(f"[Audio] Server error (Attempt {attempt + 1}): {res.status_code}")
                    except Exception as e:
                        logging.error(f"[Audio] Connection error (Attempt {attempt + 1}): {e}")
                    time.sleep(1)
                    
                audio_queue.task_done()
            else:
                time.sleep(0.01)  # Short sleep when queue is empty
        except Exception as e:
            logging.error(f"Error in audio worker: {e}")
            time.sleep(0.1)

def check_start_signal():
    for attempt in range(3):  # Retry logic
        try:
            res = requests.get(f'{FLASK_SERVER}/start_signal', timeout=5)
            if res.ok:
                return res.json()
            else:
                logging.error(f"[Signal] Server error (Attempt {attempt + 1}): {res.status_code}")
        except Exception as e:
            logging.error(f"[Signal] Connection error (Attempt {attempt + 1}): {e}")
        time.sleep(1)
    logging.error("[Signal] Unable to connect after 3 attempts.")
    return None

def check_trip_status():
    try:
        response = requests.get(f"{FLASK_SERVER}/start_signal")
        if response.status_code == 200:
            return response.json().get("start", False)
        else:
            logging.error(f"Failed to check trip status: {response.status_code}")
            return False
    except Exception as e:
        logging.error(f"Error checking trip status: {e}")
        return False

class CameraThread(threading.Thread):
    def __init__(self):
        super().__init__()
        # Initialize Picamera2 for Raspberry Pi
        self.picam2 = Picamera2()
        
        # Configure camera for emotion detection (optimized for headless)
        config = self.picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            controls={"FrameDurationLimits": (33333, 33333)}  # 30 FPS
        )
        self.picam2.configure(config)
        self.picam2.start()
        
        logging.info("Picamera2 initialized successfully (headless mode)")
        self.running = True

    def run(self):
        global latest_frame, running
        while self.running and running:
            try:
                # Capture frame from Picamera2
                frame = self.picam2.capture_array()
                
                # Convert from RGB to BGR for OpenCV compatibility
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                with frame_lock:
                    latest_frame = frame_bgr.copy()
                
                # No display in headless mode
                # cv2.imshow() is removed for headless operation
                    
            except Exception as e:
                logging.error(f"Error capturing frame: {e}")
                time.sleep(0.1)
                
        self.picam2.stop()
        # No cv2.destroyAllWindows() needed in headless mode

    def stop(self):
        self.running = False

class FaceDetectionThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        global latest_frame, trip_id_global, running
        while self.running and running:
            if latest_frame is None or trip_id_global is None:
                time.sleep(IMAGE_SEND_FREQUENCY)
                continue
                
            with frame_lock:
                frame = latest_frame.copy()
            
            # Process frame for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            
            if len(faces) > 0:
                # If face detected, send the face image
                for (x, y, w, h) in faces:
                    face = gray[y:y+h, x:x+w]
                    face = cv2.equalizeHist(face)
                    face = cv2.resize(face, (48, 48))
                    
                    # Add to queue instead of direct sending
                    if not image_queue.full():
                        image_queue.put((face, trip_id_global))
            else:
                # If no face detected, send the full frame for continuous monitoring
                full_frame = cv2.resize(gray, (48, 48))
                full_frame = cv2.equalizeHist(full_frame)
                
                # Add to queue instead of direct sending
                if not image_queue.full():
                    image_queue.put((full_frame, trip_id_global))
            
            time.sleep(IMAGE_SEND_FREQUENCY)

    def stop(self):
        self.running = False

class AudioThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        global running
        while self.running and running:
            try:
                logging.info(f"Recording audio for {AUDIO_DURATION} seconds...")
                audio = sd.rec(int(AUDIO_DURATION * AUDIO_RATE), samplerate=AUDIO_RATE, channels=1, dtype='int16')
                sd.wait()
                logging.info("Audio recording completed, processing...")
                
                # Add to queue instead of direct sending
                if not audio_queue.full():
                    audio_queue.put((audio, trip_id_global))
                else:
                    logging.warning("Audio queue is full, skipping audio recording")
                    
            except Exception as e:
                logging.error(f"Error in audio recording: {e}")
                
            time.sleep(SEND_INTERVAL)

    def stop(self):
        self.running = False

def main():
    global running, trip_id_global

    logging.info("Starting Raspberry Pi Emotion Detection System (Headless Mode)")

    while True:
        logging.info("Waiting for start signal from server...")
        signal_data = check_start_signal()
        if signal_data and signal_data.get('start'):
            trip_id_global = signal_data.get('trip_id')
            trip_duration = signal_data.get('trip_duration')
            logging.info(f"Starting Trip ID {trip_id_global} for {trip_duration} minutes.")
            running = True

            # Start worker threads for sending data
            image_worker = threading.Thread(target=send_image_worker, daemon=True)
            audio_worker = threading.Thread(target=send_audio_worker, daemon=True)
            
            # Start main processing threads
            camera_thread = CameraThread()
            face_thread = FaceDetectionThread()
            audio_thread = AudioThread()

            # Start all threads
            image_worker.start()
            audio_worker.start()
            camera_thread.start()
            face_thread.start()
            audio_thread.start()

            start_time = time.time()
            trip_duration_seconds = trip_duration * 60

            while True:
                elapsed_time = time.time() - start_time

                # Check if the trip duration has ended
                if elapsed_time >= trip_duration_seconds:
                    logging.info("Trip duration has ended. Stopping data transmission.")
                    break

                # Check server signal to ensure trip is still active
                if not check_trip_status():
                    logging.info("Server indicates trip has ended. Stopping data transmission.")
                    break

                time.sleep(1)

            # End trip session
            running = False
            
            # Stop main threads
            camera_thread.stop()
            face_thread.stop()
            audio_thread.stop()

            # Wait for main threads to finish
            camera_thread.join()
            face_thread.join()
            audio_thread.join()
            
            # Wait for queues to be processed
            image_queue.join()
            audio_queue.join()

            logging.info("Trip session ended, returning to wait for signal...")

        else:
            time.sleep(3)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Stopped by user.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        logging.info("System will restart in 10 seconds...")
        time.sleep(10)
        main()  # Restart the system 