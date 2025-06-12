import cv2
import requests
import sounddevice as sd
import scipy.io.wavfile as wav
import io
import threading
import time
import logging
from picamera2 import Picamera2
from queue import Queue


FLASK_SERVER = 'http://192.168.137.74:5000'
AUDIO_RATE = 16000
AUDIO_DURATION = 3
SEND_INTERVAL = 3
IMAGE_SEND_FREQUENCY = 0.08

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')


frame_lock = threading.Lock()
latest_frame = None
trip_id_global = None
running = False

image_queue = Queue(maxsize=50)
audio_queue = Queue(maxsize=10)

logging.basicConfig(level=logging.INFO)


def send_image_worker():
    while True:
        if not image_queue.empty():
            face_img, trip_id = image_queue.get()
            _, img_encoded = cv2.imencode('.jpg', face_img)
            files = {'file': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}
            data = {'trip_id': trip_id}
            try:
                res = requests.post(f'{FLASK_SERVER}/upload_image', files=files, data=data, timeout=5)
                if res.ok:
                    logging.info("[Image] Sent")
                else:
                    logging.warning(f"[Image] Server error: {res.status_code}")
            except Exception as e:
                logging.warning(f"[Image] Failed: {e}")
            image_queue.task_done()
        else:
            time.sleep(0.01)


def send_audio_worker():
    while True:
        if not audio_queue.empty():
            audio_data, trip_id = audio_queue.get()
            buf = io.BytesIO()
            wav.write(buf, AUDIO_RATE, audio_data)
            buf.seek(0)
            files = {'file': ('audio.wav', buf, 'audio/wav')}
            try:
                res = requests.post(f'{FLASK_SERVER}/upload_audio', files=files, timeout=10)
                if res.ok:
                    logging.info("[Audio] Sent")
                else:
                    logging.warning(f"[Audio] Server error: {res.status_code}")
            except Exception as e:
                logging.warning(f"[Audio] Failed: {e}")
            audio_queue.task_done()
        else:
            time.sleep(0.01)


def check_start_signal():
    try:
        res = requests.get(f'{FLASK_SERVER}/start_signal', timeout=5)
        if res.ok:
            return res.json()
    except Exception as e:
        logging.warning(f"[Signal] Check failed: {e}")
    return None


def check_trip_status():
    try:
        res = requests.get(f'{FLASK_SERVER}/start_signal', timeout=3)
        if res.ok:
            return res.json().get("start", False)
    except:
        pass
    return False

class CameraThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            controls={"FrameDurationLimits": (33333, 33333)}
        )
        self.picam2.configure(config)
        self.picam2.start()
        self.running = True

    def run(self):
        global latest_frame
        while self.running:
            try:
                frame = self.picam2.capture_array()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                with frame_lock:
                    latest_frame = frame_bgr.copy()
                cv2.imshow("Camera (press q to quit)", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break
            except Exception as e:
                logging.error(f"[Camera] Error: {e}")
        self.picam2.stop()
        cv2.destroyAllWindows()


class FaceDetectionThread(threading.Thread):
    def run(self):
        global running, trip_id_global, latest_frame
        while True:
            if running and latest_frame is not None and trip_id_global is not None:
                with frame_lock:
                    frame = latest_frame.copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    face_img = gray[y:y+h, x:x+w]
                else:
                    face_img = gray
                face_img = cv2.resize(cv2.equalizeHist(face_img), (48, 48))
                if not image_queue.full():
                    image_queue.put((face_img, trip_id_global))
            time.sleep(IMAGE_SEND_FREQUENCY)


class AudioThread(threading.Thread):
    def run(self):
        global running, trip_id_global
        while True:
            if running and trip_id_global:
                try:
                    logging.info("Recording audio…")
                    audio = sd.rec(int(AUDIO_RATE * AUDIO_DURATION),
                                   samplerate=AUDIO_RATE,
                                   channels=1, dtype='int16')
                    sd.wait()
                    if not audio_queue.full():
                        audio_queue.put((audio, trip_id_global))
                except Exception as e:
                    logging.error(f"[Audio] Record error: {e}")
            time.sleep(SEND_INTERVAL)

def main():
    global running, trip_id_global
    cam_thread = CameraThread()
    cam_thread.start()

    threading.Thread(target=send_image_worker, daemon=True).start()
    threading.Thread(target=send_audio_worker, daemon=True).start()

    face_thread = FaceDetectionThread()
    audio_thread = AudioThread()
    face_thread.daemon = True
    audio_thread.daemon = True
    face_thread.start()
    audio_thread.start()

    try:
        while cam_thread.running:
            sig = check_start_signal()
            if sig and sig.get("start"):
                trip_id_global = sig["trip_id"]
                duration = sig["trip_duration"] * 60
                logging.info(f"===> Trip started (ID: {trip_id_global}, duration: {duration//60}m)")

                running = True
                trip_start_time = time.time()

              
                while time.time() - trip_start_time < duration:
                    if not check_trip_status():
                        logging.info("Trip stopped early by server.")
                        break
                    time.sleep(1)

                running = False
                logging.info("===> Trip ended, back to waiting mode.\n")
                trip_id_global = None
            else:
                time.sleep(3)
    except KeyboardInterrupt:
        logging.info("Stopping by user...")
    finally:
        cam_thread.running = False
        cam_thread.join()

if __name__ == '__main__':
    main()




