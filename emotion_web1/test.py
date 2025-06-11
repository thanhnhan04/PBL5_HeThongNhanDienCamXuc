import cv2
import requests
import sounddevice as sd
import scipy.io.wavfile as wav
import io
import threading
import time
import logging

FLASK_SERVER = 'http://localhost:5000'
AUDIO_RATE = 16000
AUDIO_DURATION = 3  # 10 seconds
SEND_INTERVAL = 3  # Interval between sending data in seconds
IMAGE_SEND_FREQUENCY = 0.05  # Send image every 0.05 seconds (20 FPS) for real-time

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Shared variables for threads
frame_lock = threading.Lock()
latest_frame = None
trip_id_global = None
running = False

logging.basicConfig(level=logging.INFO)

def send_image(face_img, trip_id):
    _, img_encoded = cv2.imencode('.jpg', face_img)
    files = {'file': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}
    data = {'trip_id': trip_id}
    for attempt in range(2):  # Reduced retry attempts from 3 to 2
        try:
            res = requests.post(f'{FLASK_SERVER}/upload_image', files=files, data=data, timeout=2)  # Reduced timeout from 5 to 2 seconds
            if res.ok:
                logging.info("[Image] Sent successfully")
                return
            else:
                logging.error(f"[Image] Server error (Attempt {attempt + 1}): {res.status_code}")
        except Exception as e:
            logging.error(f"[Image] Connection error (Attempt {attempt + 1}): {e}")
        time.sleep(0.5)  # Reduced sleep time from 1 to 0.5 seconds

def send_audio():
    for attempt in range(3):  # Retry logic
        try:
            logging.info(f"Recording audio for {AUDIO_DURATION} seconds...")
            audio = sd.rec(int(AUDIO_DURATION * AUDIO_RATE), samplerate=AUDIO_RATE, channels=1, dtype='int16')
            sd.wait()
            logging.info("Audio recording completed, processing...")
            wav_buffer = io.BytesIO()
            wav.write(wav_buffer, AUDIO_RATE, audio)
            wav_buffer.seek(0)
            files = {'file': ('audio.wav', wav_buffer, 'audio/wav')}
            res = requests.post(f'{FLASK_SERVER}/upload_audio', files=files, timeout=10)
            if res.ok:
                logging.info("[Audio] Sent successfully and processed by server")
                return
            else:
                logging.error(f"[Audio] Server error (Attempt {attempt + 1}): {res.status_code}")
        except Exception as e:
            logging.error(f"[Audio] Connection error (Attempt {attempt + 1}): {e}")
        time.sleep(1)

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
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logging.error("Unable to open webcam.")
            raise RuntimeError("Cannot open webcam")
        self.running = True

    def run(self):
        global latest_frame, running
        while self.running and running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            with frame_lock:
                latest_frame = frame.copy()
            cv2.imshow("Real-Time Emotion Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
                break
        self.cap.release()
        cv2.destroyAllWindows()

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
                time.sleep(IMAGE_SEND_FREQUENCY)  # Use configurable frequency
                continue
            with frame_lock:
                frame = latest_frame.copy()
            
            # Always send the frame regardless of face detection for more continuous data
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            
            if len(faces) > 0:
                # If face detected, send the face image
                for (x, y, w, h) in faces:
                    face = gray[y:y+h, x:x+w]
                    face = cv2.equalizeHist(face)
                    face = cv2.resize(face, (48, 48))
                    threading.Thread(target=send_image, args=(face, trip_id_global)).start()
            else:
                # If no face detected, send the full frame for continuous monitoring
                full_frame = cv2.resize(gray, (48, 48))
                full_frame = cv2.equalizeHist(full_frame)
                threading.Thread(target=send_image, args=(full_frame, trip_id_global)).start()
            
            time.sleep(IMAGE_SEND_FREQUENCY)  # Use configurable frequency for consistent timing

    def stop(self):
        self.running = False

class AudioThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        global running
        while self.running and running:
            send_audio()
            time.sleep(SEND_INTERVAL)  # Send audio at regular intervals

    def stop(self):
        self.running = False

def main():
    global running, trip_id_global

    while True:
        logging.info("Waiting for start signal from server...")
        signal_data = check_start_signal()
        if signal_data and signal_data.get('start'):
            trip_id_global = signal_data.get('trip_id')
            trip_duration = signal_data.get('trip_duration')
            logging.info(f"Starting Trip ID {trip_id_global} for {trip_duration} minutes.")
            running = True

            camera_thread = CameraThread()
            face_thread = FaceDetectionThread()
            audio_thread = AudioThread()

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
            camera_thread.stop()
            face_thread.stop()
            audio_thread.stop()

            camera_thread.join()
            face_thread.join()
            audio_thread.join()

            logging.info("Trip session ended, returning to wait for signal...")

        else:
            time.sleep(3)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Stopped by user.")
