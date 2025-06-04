from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
from keras.models import load_model
import librosa
import csv
import pandas as pd
from datetime import datetime
import logging
import time
from collections import Counter
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
@app.route('/test')
def test():
    return jsonify({"message": "Test successful!"})
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

face_model = load_model('C:/Project/PBL5_HeThongNhanDienCamXuc/emotion_web1/models/emotion_cnn_best11.keras')
voice_model = load_model('C:/Project/PBL5_HeThongNhanDienCamXuc/emotion_web1/models/audio2_best.keras')
print("abc")
print(voice_model.input_shape)
voice_model.summary()

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

latest_emotions = {'face': None, 'voice': None}

collection_details = {'start': False, 'customer_id': None, 'trip_id': None, 'trip_duration': None}

logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def index():
    trip_files = os.listdir('C:/Project/PBL5_HeThongNhanDienCamXuc/emotion_web1/data_emotion')
    trip_ids = [os.path.splitext(file)[0] for file in trip_files if file.endswith('.csv')]

    return render_template('index.html', emotions=latest_emotions, trip_ids=trip_ids)

@app.route('/upload_image', methods=['POST'])
def upload_image():
    try:
        file = request.files['file']
        trip_id = request.form['trip_id']
        path = os.path.join(UPLOAD_FOLDER, 'temp.jpg')
        file.save(path)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        img = cv2.equalizeHist(img)

        img = cv2.resize(img, (48, 48))

        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=-1) 
        img = np.expand_dims(img, axis=0)   

        pred = face_model.predict(img)
        confidence = float(np.max(pred)) 
        emotion = EMOTIONS[np.argmax(pred)]

        logging.debug(f"Preprocessed Image Shape: {img.shape}")
        logging.debug(f"Model Predictions: {pred}")
        logging.debug(f"Confidence: {confidence}, Emotion: {emotion}")

        if confidence < 0.6:  
            emotion = "Uncertain"

        latest_emotions['face'] = emotion

        write_to_csv(trip_id, emotion, 'N/A')

        return jsonify({"emotion": emotion, "confidence": confidence})
    except Exception as e:
        logging.error(f"Error in /upload_image route: {e}")
        return "Internal Server Error", 500
# @app.route('/upload_audio', methods=['POST'])   
# def upload_audio():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     # Tạo tên file theo thời gian
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     filename = f'audio_{timestamp}.wav'
#     filepath = os.path.join(UPLOAD_FOLDER, filename)

#     # Lưu file vào thư mục
#     file.save(filepath)
#     print(f"Đã lưu file âm thanh vào: {filepath}")

#     return jsonify({'message': 'File received', 'filename': filename}), 200

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    file = request.files['file']
    path = os.path.join(UPLOAD_FOLDER, 'temp.wav')
    file.save(path)

    y, sr = librosa.load(path, sr=16000)
    # mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    # mfcc = np.mean(mfcc.T, axis=0).reshape(1, -1)
    # pred = voice_model.predict(mfcc)

    # Dự đoán cảm xúc từ giọng nói
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = librosa.util.fix_length(mfcc, size=128, axis=1)  # Đảm bảo có 128 frame (trục thời gian)

    # Resize về 128x128 (kéo giãn nếu cần)
    mfcc_resized = cv2.resize(mfcc, (128, 128), interpolation=cv2.INTER_LINEAR)

    # Chuẩn hoá và định hình lại
    mfcc_resized = mfcc_resized.astype("float32") / 255.0
    mfcc_resized = np.expand_dims(mfcc_resized, axis=-1)  # (128, 128, 1)
    mfcc_resized = np.expand_dims(mfcc_resized, axis=0)   # (1, 128, 128, 1)
    pred = voice_model.predict(mfcc_resized)
    emotion = EMOTIONS[np.argmax(pred)]
    latest_emotions['voice'] = emotion
    
    # Ghi vào file CSV
    write_to_csv(latest_emotions['voice'] if latest_emotions['voice'] else 'Neutral', emotion, 'N/A')
    
    return jsonify({"emotion": emotion})

@app.route('/summary', methods=['GET', 'POST'])
def summary():
    try:
        trip_files = os.listdir('C:/Project/PBL5_HeThongNhanDienCamXuc/emotion_web1/data_emotion')
        trip_ids = [os.path.splitext(file)[0] for file in trip_files if file.endswith('.csv')]

        trip_id = None
        if request.method == 'POST':
            trip_id = request.form.get('trip_id')
        elif request.method == 'GET':
            trip_id = request.args.get('trip_id')

        if trip_id:
            file_path = f'C:/Project/PBL5_HeThongNhanDienCamXuc/emotion_web1/data_emotion/{trip_id}.csv'
            
            if not os.path.exists(file_path):
                return render_template('summary.html', trip_ids=trip_ids, emotion_percentages={}, satisfaction="No data available")

            df = pd.read_csv(file_path, names=['timestamp', 'face_emotion', 'voice_emotion'])

            filtered_df = df[df['face_emotion'] != 'Uncertain']
            face_emotions = filtered_df['face_emotion'].value_counts(normalize=True) * 100
            emotion_percentages = face_emotions.to_dict()

            satisfaction = None
            if 'happy' in emotion_percentages and emotion_percentages['happy'] > 50:
                satisfaction = "High"
            elif 'angry' in emotion_percentages and emotion_percentages['angry'] > 50:
                satisfaction = "Low"
            else:
                satisfaction = "Moderate"

            logging.debug(f"Emotion Percentages: {emotion_percentages}")
            logging.debug(f"Satisfaction Level: {satisfaction}")

            return render_template('summary.html', trip_ids=trip_ids, emotion_percentages=emotion_percentages, satisfaction=satisfaction)

        return render_template('summary.html', trip_ids=trip_ids, emotion_percentages={}, satisfaction=None)
    except Exception as e:
        logging.error(f"Error in /summary route: {e}")
        return "Internal Server Error", 500

@app.route('/get_latest_emotion')
def get_latest_emotion():
    return jsonify(latest_emotions)

@app.route('/start_collection', methods=['POST'])
def start_collection():
    try:
        collection_details['trip_id'] = request.form['trip_id']
        collection_details['trip_duration'] = int(request.form['trip_duration'])
        collection_details['start'] = True

        logging.info(f"Started data collection for Trip ID {collection_details['trip_id']} for {collection_details['trip_duration']} minutes.")
        return jsonify({"message": "Data collection started successfully.", "status": "started"}), 200
    except Exception as e:
        logging.error(f"Error in /start_collection route: {e}")
        return jsonify({"message": "Failed to start data collection.", "status": "error"}), 500

@app.route('/start_signal', methods=['GET'])
def start_signal():
    return jsonify(collection_details)

@app.route('/reset_signal', methods=['POST'])
def reset_signal():
    collection_details['start'] = False
    return "Signal reset successfully.", 200

@app.route('/upload_results', methods=['POST'])
def upload_results():
    try:
        data = request.json
        trip_id = data['trip_id']
        face_emotion = data['face_emotion']
        voice_emotion = data.get('voice_emotion', 'N/A')

        write_to_csv(trip_id, face_emotion, voice_emotion)

        return jsonify({"message": "Results logged successfully."})
    except Exception as e:
        logging.error(f"Error in /upload_results route: {e}")
        return "Internal Server Error", 500

def write_to_csv(trip_id, face_emotion, voice_emotion):
    file_path = f'C:/Project/PBL5_HeThongNhanDienCamXuc/emotion_web1/data_emotion/{trip_id}.csv'
    
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([timestamp, face_emotion, voice_emotion])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
