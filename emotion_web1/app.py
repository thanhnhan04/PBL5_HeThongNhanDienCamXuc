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
from flask_cors import CORS
import threading
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List  # Add this import at the top of the file

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models
face_model = load_model('E:/Python/PBL5/PBL5/emotion_web1/models/emotion_cnn_best16.keras')
voice_model = load_model('E:/Python/PBL5/PBL5/emotion_web1/models/audio_emotion_audio5.keras')

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

latest_emotions = {'face': None, 'voice': None}
collection_details = {'start': False, 'trip_id': None, 'trip_duration': None}

logging.basicConfig(level=logging.DEBUG)

@app.route('/load_csv', methods=['GET'])
def load_csv():
    trip_id = request.args.get('trip_id')
    csv_file_path = f'E:/Python/PBL5/PBL5/emotion_web1/data_emotion/{trip_id}.csv'
    try:
        if not os.path.exists(csv_file_path):
            return jsonify([])  # Return an empty array if the file doesn't exist

        data = []
        with open(csv_file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                data.append({
                    'timestamp': row[0],
                    'face_emotion': row[1],
                    'voice_emotion': row[2]
                })
        return jsonify(data)
    except Exception as e:
        logging.error(f"Error loading CSV file {csv_file_path}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    trip_files = os.listdir('E:/Python/PBL5/PBL5/emotion_web1/data_emotion')
    trip_ids = [os.path.splitext(file)[0] for file in trip_files if file.endswith('.csv')]
    return render_template('index.html', emotions=latest_emotions, trip_ids=trip_ids)

@app.route('/upload_image', methods=['POST'])
def upload_image():
    try:
        # Check if the trip is still active
        if not collection_details['start']:
            logging.warning("Trip has ended. Rejecting image upload.")
            return "Trip has ended. Data collection is no longer active.", 403

        file = request.files['file']
        trip_id = request.form['trip_id']
        path = os.path.join(UPLOAD_FOLDER, 'temp.jpg')
        file.save(path)

        # Optimized image processing for faster real-time analysis
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return "Invalid image", 400
            
        img = cv2.equalizeHist(img)
        img = cv2.resize(img, (48, 48))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=-1) 
        img = np.expand_dims(img, axis=0)   

        # Faster prediction with reduced batch size
        pred = face_model.predict(img, verbose=0)  # Disable verbose for faster processing
        confidence = float(np.max(pred)) 
        emotion = EMOTIONS[np.argmax(pred)]

        if confidence < 0.6:  
            emotion = "Uncertain"

        latest_emotions['face'] = emotion
        write_to_csv(trip_id, emotion, 'N/A')

        # Clean up temporary file immediately
        try:
            os.remove(path)
        except:
            pass

        return '', 204  # No Content

    except Exception as e:
        logging.error(f"Error in /upload_image route: {e}")
        return "Internal Server Error", 500

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    try:
        # Check if the trip is still active
        if not collection_details['start']:
            logging.warning("Trip has ended. Rejecting audio upload.")
            return "Trip has ended. Data collection is no longer active.", 403

        file = request.files['file']
        path = os.path.join(UPLOAD_FOLDER, 'temp.wav')
        file.save(path)

        y, sr = librosa.load(path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        target_frames = 94
        if mfcc.shape[1] < target_frames:
            pad_width = target_frames - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :target_frames]

        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-9)
        mfcc = np.expand_dims(mfcc, axis=-1)
        mfcc = np.expand_dims(mfcc, axis=0)

        # Faster prediction with reduced batch size
        pred = voice_model.predict(mfcc, verbose=0)  # Disable verbose for faster processing
        confidence = float(np.max(pred))
        emotion = EMOTIONS[np.argmax(pred)]
        
        # Log the audio emotion detection result
        logging.info(f"[Audio] Detected emotion: {emotion} with confidence: {confidence:.3f}")
        
        latest_emotions['voice'] = emotion

        # Use the current trip_id from collection_details
        current_trip_id = collection_details['trip_id'] if collection_details['trip_id'] else 'unknown'
        write_to_csv(current_trip_id, 'N/A', emotion)

        # Clean up temporary file immediately
        try:
            os.remove(path)
        except:
            pass

        return '', 204

    except Exception as e:
        logging.error(f"Error in /upload_audio route: {e}")
        return "Internal Server Error", 500

@app.route('/summary', methods=['GET', 'POST'])
def summary():
    try:
        # Lấy danh sách file CSV từ thư mục chứa dữ liệu
        trip_files = os.listdir('E:/Python/PBL5/PBL5/emotion_web1/data_emotion')
        trip_ids = [os.path.splitext(file)[0] for file in trip_files if file.endswith('.csv')]

        trip_id = None
        if request.method == 'POST':
            trip_id = request.form.get('trip_id')
        elif request.method == 'GET':
            trip_id = request.args.get('trip_id')

        if trip_id:
            file_path = f'E:/Python/PBL5/PBL5/emotion_web1/data_emotion/{trip_id}.csv'
            if not os.path.exists(file_path):
                return render_template('summary.html', trip_ids=trip_ids, emotion_percentages={}, satisfaction="No data available")

            # Đọc dữ liệu CSV
            df = pd.read_csv(file_path, names=['timestamp', 'face_emotion', 'voice_emotion'])

            # Lọc bỏ dòng có cảm xúc không xác định
            filtered_df = df[df['face_emotion'] != 'Uncertain']

            # Tính phần trăm các cảm xúc khuôn mặt
            face_emotions = filtered_df['face_emotion'].value_counts(normalize=True) * 100
            emotion_percentages = face_emotions.to_dict()

            # Gán trọng số cho từng loại cảm xúc
            weights = {
                'happy': 1.0,
                'neutral': 0.5,
                'surprise': 0.0,
                'sad': -0.5,
                'fear': -0.7,
                'disgust': -0.8,
                'angry': -1.0
            }

            # Tính điểm hài lòng
            satisfaction_score = 0
            for emotion, percent in emotion_percentages.items():
                weight = weights.get(emotion.lower(), 0)
                satisfaction_score += percent * weight

            # Xác định mức độ hài lòng
            if satisfaction_score >= 30:
                satisfaction = "High"
            elif satisfaction_score <= -30:
                satisfaction = "Low"
            else:
                satisfaction = "Moderate"

            logging.debug(f"Emotion Percentages: {emotion_percentages}")
            logging.debug(f"Satisfaction Score: {satisfaction_score}")
            logging.debug(f"Satisfaction Level: {satisfaction}")

            return render_template('summary.html', trip_ids=trip_ids, emotion_percentages=emotion_percentages, satisfaction=satisfaction)

        return render_template('summary.html', trip_ids=trip_ids, emotion_percentages={}, satisfaction=None)

    except Exception as e:
        logging.error(f"Error in /summary route: {e}")
        return "Internal Server Error", 500

    
@app.route('/send_audio_signal', methods=['GET'])
def send_audio_signal():
    # Ở đây bạn có thể kiểm tra điều kiện nếu cần
    return jsonify({'send_audio': True})  # hoặc False nếu bạn muốn chặn gửi audio

@app.route('/get_latest_emotion')
def get_latest_emotion():
    # Add debug information
    logging.debug(f"Latest emotions: {latest_emotions}")
    return jsonify(latest_emotions)

@app.route('/audio_debug', methods=['GET'])
def audio_debug():
    """Debug endpoint to check audio processing status"""
    try:
        # Check if voice model is loaded
        model_loaded = voice_model is not None
        
        # Get recent audio processing info
        debug_info = {
            'voice_model_loaded': model_loaded,
            'latest_voice_emotion': latest_emotions.get('voice'),
            'collection_active': collection_details['start'],
            'current_trip_id': collection_details['trip_id'],
            'emotions_list': EMOTIONS
        }
        
        return jsonify(debug_info)
    except Exception as e:
        logging.error(f"Error in audio_debug: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/start_collection', methods=['POST'])
def start_collection():
    try:
        collection_details['trip_id'] = request.form['trip_id']
        collection_details['trip_duration'] = int(request.form['trip_duration'])
        collection_details['start'] = True

        # Schedule trip end after the specified duration
        duration_seconds = collection_details['trip_duration'] * 60
        threading.Timer(duration_seconds, end_trip).start()

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

@app.route('/monthly_summary')
def monthly_summary():
    try:
        directory_path = 'E:/Python/PBL5/PBL5/emotion_web1/data_emotion'
        monthly_stats = read_emotion_data_by_month(directory_path)
        return render_template('monthly_summary.html', monthly_stats=monthly_stats)
    except Exception as e:
        logging.error(f"Error in /monthly_summary route: {e}")
        return "Internal Server Error", 500

import csv
import os
from datetime import datetime

def write_to_csv(trip_id, face_emotion, voice_emotion):
    try:
        # Tạo thư mục nếu chưa có
        output_dir = "E:/Python/PBL5/PBL5/emotion_web1/data_emotion"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Tạo tên file theo trip_id
        filename = os.path.join(output_dir, f'{trip_id}.csv')

        # Tạo file mới nếu chưa có, thêm header
        if not os.path.isfile(filename):
            with open(filename, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Face Emotion', 'Voice Emotion'])

        # Ghi dòng mới
        with open(filename, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            writer.writerow([timestamp, face_emotion, voice_emotion])

        # Improved logging with more details
        if voice_emotion != 'N/A':
            logging.info(f"[CSV] Voice emotion detected: {voice_emotion} for trip {trip_id}")
        if face_emotion != 'N/A':
            logging.info(f"[CSV] Face emotion detected: {face_emotion} for trip {trip_id}")
        
        print(f"Logged: {face_emotion}, {voice_emotion} into {filename}")

    except Exception as e:
        logging.error(f"[ERROR] Failed to write to CSV: {e}")
        print(f"[ERROR] Failed to write to CSV: {e}")

def end_trip():
    collection_details['start'] = False
    logging.info("Trip duration has ended. Data collection is now inactive.")

def read_emotion_data_by_month(directory_path: str) -> Dict[str, List[float]]:
    """
    Calculate monthly satisfaction scores based on emotion data in CSV files.

    Args:
        directory_path (str): Path to the directory containing emotion CSV files.

    Returns:
        Dict[str, List[float]]: A dictionary where keys are months (e.g., '2025-06') and values are lists of satisfaction scores.
    """
    weights = {
        'happy': 1.0,
        'neutral': 0.5,
        'surprise': 0.0,
        'sad': -0.5,
        'fear': -0.7,
        'disgust': -0.8,
        'angry': -1.0
    }

    monthly_scores = defaultdict(list)

    for file_name in os.listdir(directory_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(directory_path, file_name)
            try:
                df = pd.read_csv(file_path, names=['timestamp', 'face_emotion', 'voice_emotion'])
                filtered_df = df[df['face_emotion'] != 'Uncertain']
                face_emotions = filtered_df['face_emotion']

                satisfaction_score = 0
                for emotion in face_emotions:
                    weight = weights.get(emotion.lower(), 0)
                    satisfaction_score += weight

                if not face_emotions.empty:
                    average_score = satisfaction_score / len(face_emotions)
                    month = file_name[:7]  # Extract 'YYYY-MM' from file name
                    monthly_scores[month].append(average_score)
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")

    return monthly_scores

def plot_monthly_satisfaction_chart(monthly_stats: Dict[str, List[float]]):
    """
    Plot monthly satisfaction scores using matplotlib.

    Args:
        monthly_stats (Dict[str, List[float]]): Monthly satisfaction scores.
    """
    months = []
    average_scores = []

    for month, scores in sorted(monthly_stats.items()):
        months.append(month)
        average_scores.append(sum(scores) / len(scores))

    plt.figure(figsize=(10, 6))
    plt.plot(months, average_scores, marker='o', linestyle='-', color='b')
    plt.title('Monthly Customer Satisfaction Scores')
    plt.xlabel('Month')
    plt.ylabel('Average Satisfaction Score')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
