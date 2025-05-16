import cv2
import requests
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import time
import io
import threading

FLASK_SERVER = 'http://localhost:5000'
AUDIO_RATE = 16000
AUDIO_DURATION = 3

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không mở được webcam.")
    exit()

trip_id = None
trip_duration = None

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

FRAME_SKIP = 2
frame_count = 0

def send_image(face_img):
    _, img_encoded = cv2.imencode('.jpg', face_img)
    files = {'file': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}
    data = {'trip_id': trip_id}
    try:
        res = requests.post(f'{FLASK_SERVER}/upload_image', files=files, data=data)
        if res.ok:
            print("[Ảnh] OK:", res.json())
        else:
            print("[Ảnh] Server lỗi:", res.status_code)
    except Exception as e:
        print("[Ảnh] Gửi lỗi:", e)

def send_audio():
    print("  Ghi âm...")
    audio = sd.rec(int(AUDIO_DURATION * AUDIO_RATE), samplerate=AUDIO_RATE, channels=1, dtype='int16')
    sd.wait()
    wav_buffer = io.BytesIO()
    wav.write(wav_buffer, AUDIO_RATE, audio)
    wav_buffer.seek(0)
    files = {'file': ('temp.wav', wav_buffer, 'audio/wav')}
    try:
        res = requests.post(f'{FLASK_SERVER}/upload_audio', files=files)
        if res.ok:
            print("[Âm thanh] OK:", res.json())
        else:
            print("[Âm thanh] Server lỗi:", res.status_code)
    except Exception as e:
        print("[Âm thanh] Gửi lỗi:", e)

def send_results(trip_id, face_emotion, voice_emotion="N/A"):
    data = {
        'trip_id': trip_id,
        'face_emotion': face_emotion,
        'voice_emotion': voice_emotion
    }
    try:
        res = requests.post(f'{FLASK_SERVER}/send_results', json=data)
        if res.ok:
            print("[Kết quả] OK:", res.json())
        else:
            print("[Kết quả] Server lỗi:", res.status_code)
    except Exception as e:
        print("[Kết quả] Gửi lỗi:", e)

try:
    res = requests.get(f'{FLASK_SERVER}/start_signal')
    if res.ok and res.json().get('start'):
        trip_id = res.json().get('trip_id')
        trip_duration = res.json().get('trip_duration')
        print(f"Bắt đầu Trip ID {trip_id} trong {trip_duration} phút.")

        last_audio_time = 0
        start_time = time.time()

        while time.time() - start_time < trip_duration * 60:
            ret, frame = cap.read()
            if not ret:
                continue

            # Skip frames to reduce lag
            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                cv2.imshow("Real-Time Emotion Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in faces:
                # Corrected face cropping logic
                face = gray[y:y+h, x:x+w]
                face = cv2.equalizeHist(face)
                face = cv2.resize(face, (48, 48))
                threading.Thread(target=send_image, args=(face,)).start()

                # Simulate emotion detection for demonstration
                face_emotion = "Happy"  # Replace with actual emotion detection logic
                threading.Thread(target=send_results, args=(trip_id, face_emotion)).start()

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label = f"{face_emotion}"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow("Real-Time Emotion Detection", frame)

            # Process audio every 30 seconds
            if time.time() - last_audio_time > 30:
                threading.Thread(target=send_audio).start()
                last_audio_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
except KeyboardInterrupt:
    print("Dừng bởi người dùng.")
finally:
    cap.release()
    cv2.destroyAllWindows()