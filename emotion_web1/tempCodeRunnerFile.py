import cv2
import requests
import sounddevice as sd
import scipy.io.wavfile as wav
import io
import threading
import time

FLASK_SERVER = 'http://localhost:5000'
AUDIO_RATE = 16000
AUDIO_DURATION = 10  # 10 seconds

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Shared variables for threads
frame_lock = threading.Lock()
latest_frame = None
trip_id_global = None
running = False

def send_image(face_img, trip_id):
    _, img_encoded = cv2.imencode('.jpg', face_img)
    files = {'file': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}
    data = {'trip_id': trip_id}
    for attempt in range(3):  # Retry logic
        try:
            res = requests.post(f'{FLASK_SERVER}/upload_image', files=files, data=data, timeout=5)
            if res.ok:
                print("[Image] Sent successfully")
                return
            else:
                print(f"[Image] Server error (Attempt {attempt + 1}):", res.status_code)
        except Exception as e:
            print(f"[Image] Connection error (Attempt {attempt + 1}):", e)
        time.sleep(1)

def send_audio():
    for attempt in range(3):  # Retry logic
        try:
            print("  Recording audio for 10 seconds...")
            audio = sd.rec(int(AUDIO_DURATION * AUDIO_RATE), samplerate=AUDIO_RATE, channels=1, dtype='int16')
            sd.wait()
            wav_buffer = io.BytesIO()
            wav.write(wav_buffer, AUDIO_RATE, audio)
            wav_buffer.seek(0)
            files = {'file': ('audio.wav', wav_buffer, 'audio/wav')}
            res = requests.post(f'{FLASK_SERVER}/upload_audio', files=files, timeout=10)
            if res.ok:
                print("[Audio] Sent successfully")
                return
            else:
                print(f"[Audio] Server error (Attempt {attempt + 1}):", res.status_code)
        except Exception as e:
            print(f"[Audio] Connection error (Attempt {attempt + 1}):", e)
        time.sleep(1)

def check_start_signal():
    for attempt in range(3):  # Retry logic
        try:
            res = requests.get(f'{FLASK_SERVER}/start_signal', timeout=5)
            if res.ok:
                return res.json()
            else:
                print(f"[Signal] Server error (Attempt {attempt + 1}):", res.status_code)
        except Exception as e:
            print(f"[Signal] Connection error (Attempt {attempt + 1}):", e)
        time.sleep(1)
    print("[Signal] Unable to connect after 3 attempts.")
    return None

def check_send_audio_signal():
    try:
        res = requests.get(f'{FLASK_SERVER}/send_audio_signal', timeout=5)
        if res.status_code == 404:
            print("[Signal] Endpoint '/send_audio_signal' not found. Please check the server implementation.")
            return False
        elif res.ok:
            json_data = res.json()
            return json_data.get('send_audio', False)
        else:
            print("[Signal] Server error:", res.status_code)
            return False
    except Exception as e:
        print("[Signal] Connection error:", e)
        return False

class CameraThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Unable to open webcam.")
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
                time.sleep(0.1)
                continue
            with frame_lock:
                frame = latest_frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.equalizeHist(face)
                face = cv2.resize(face, (48, 48))
                threading.Thread(target=send_image, args=(face, trip_id_global)).start()
            time.sleep(0.3)  # Adjust detection frequency (3 times per second)

class AudioThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        global running
        while self.running and running:
            if check_send_audio_signal():
                send_audio()
            time.sleep(1)  # Check every second

def main():
    global running, trip_id_global

    while True:
        print("Waiting for start signal from server...")
        signal_data = check_start_signal()
        if signal_data and signal_data.get('start'):
            trip_id_global = signal_data.get('trip_id')
            trip_duration = signal_data.get('trip_duration')
            print(f"Starting Trip ID {trip_id_global} for {trip_duration} minutes.")
            running = True

            camera_thread = CameraThread()
            face_thread = FaceDetectionThread()
            audio_thread = AudioThread()

            camera_thread.start()
            face_thread.start()
            audio_thread.start()

            start_time = time.time()
            while time.time() - start_time < trip_duration * 60 and running:
                time.sleep(1)

            # End trip session
            running = False
            camera_thread.stop()

            camera_thread.join()
            face_thread.running = False
            face_thread.join()
            audio_thread.running = False
            audio_thread.join()

            print("Trip session ended, returning to wait for signal...")

        else:
            time.sleep(3)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped by user.")
