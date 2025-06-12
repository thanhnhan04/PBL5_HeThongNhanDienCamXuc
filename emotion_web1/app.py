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
from typing import Dict, List 
import noisereduce as nr
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
            return jsonify([]) 

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
     
        if not collection_details['start']:
            logging.warning("Trip has ended. Rejecting image upload.")
            return "Trip has ended. Data collection is no longer active.", 403

        file = request.files['file']
        trip_id = request.form['trip_id']
        path = os.path.join(UPLOAD_FOLDER, 'temp.jpg')
        file.save(path)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return "Invalid image", 400
            
        img = cv2.equalizeHist(img)
        img = cv2.resize(img, (48, 48))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=-1) 
        img = np.expand_dims(img, axis=0)   
        pred = face_model.predict(img, verbose=0)  
        confidence = float(np.max(pred)) 
        emotion = EMOTIONS[np.argmax(pred)]

        if confidence < 0.6:  
            emotion = "Uncertain"

        latest_emotions['face'] = emotion
        write_to_csv(trip_id, emotion, 'N/A')

    
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
        if not collection_details['start']:
            logging.warning("Trip has ended. Rejecting audio upload.")
            return "Trip has ended. Data collection is no longer active.", 403

        file = request.files['file']
        path = os.path.join(UPLOAD_FOLDER, 'temp.wav')
        file.save(path)

        y, sr = librosa.load(path, sr=16000)

    
        noise_sample = y[:int(0.5 * sr)]
        y_denoised = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample, prop_decrease=1.0)
        # ===================================

        mfcc = librosa.feature.mfcc(y=y_denoised, sr=sr, n_mfcc=40)
        target_frames = 94
        if mfcc.shape[1] < target_frames:
            pad_width = target_frames - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :target_frames]

     
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-9)
        mfcc = np.expand_dims(mfcc, axis=-1)
        mfcc = np.expand_dims(mfcc, axis=0)

     
        pred = voice_model.predict(mfcc, verbose=0)
        confidence = float(np.max(pred))
        emotion = EMOTIONS[np.argmax(pred)]
        
       
        if confidence < 0.6: 
            emotion = "Uncertain"
            logging.info(f"[Audio] Low confidence ({confidence:.3f}), marking as Uncertain")
        else:
            logging.info(f"[Audio] Detected emotion: {emotion} with confidence: {confidence:.3f}")
        
        latest_emotions['voice'] = emotion

        current_trip_id = collection_details['trip_id'] if collection_details['trip_id'] else 'unknown'
        write_to_csv(current_trip_id, 'N/A', emotion)

        # Clean up
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
                return render_template('summary.html', 
                                     emotion_percentages={}, 
                                     satisfaction="No data available")

            try:
              
                df = pd.read_csv(file_path, names=['timestamp', 'face_emotion', 'voice_emotion'], skiprows=1)
                
              
                df['face_emotion'] = df['face_emotion'].astype(str)

               
                if df.empty:
                    return render_template('summary.html', 
                                         emotion_percentages={}, 
                                         satisfaction="File empty")
           
                total_records = len(df)  
                valid_face_records = len(df[df['face_emotion'] != 'Uncertain'])
                valid_voice_records = len(df[df['voice_emotion'] != 'N/A'])
                
              
                trip_duration = 0
                if total_records > 1:
                    try:
                        start_time = pd.to_datetime(df['timestamp'].iloc[0])
                        end_time = pd.to_datetime(df['timestamp'].iloc[-1])
                        trip_duration = (end_time - start_time).total_seconds() / 60 
                    except Exception as e:
                        logging.warning(f"Could not parse trip duration: {e}")
                        trip_duration = 0

           
                filtered_df = df[df['face_emotion'].notna() & df['face_emotion'].isin(EMOTIONS)]

            
                if not filtered_df.empty:
                    face_emotions = filtered_df['face_emotion'].value_counts(normalize=True) * 100
                    emotion_percentages = face_emotions.to_dict()
                else:
                    emotion_percentages = {}

           
                weights = {
                    'happy': 1.0,
                    'neutral': 0.5,
                    'surprise': 0.2,
                    'sad': -0.5,
                    'fear': -0.7,
                    'disgust': -0.8,
                    'angry': -1.0
                }

         
                satisfaction_score = 0
                total_percent = 0
                
                for emotion, percent in emotion_percentages.items():
                    weight = weights.get(emotion.lower(), 0)
                    satisfaction_score += percent * weight
                    total_percent += percent

           
                if total_percent > 0:
                    normalized_score = satisfaction_score / total_percent
                else:
                    normalized_score = 0

           
                if normalized_score >= 0.4:
                    satisfaction = "Very Satisfied"
                elif normalized_score >= 0.1:
                    satisfaction = "Satisfied"
                elif normalized_score >= -0.1:
                    satisfaction = "Neutral"
                elif normalized_score >= -0.4:
                    satisfaction = "Dissatisfied"
                else:
                    satisfaction = "Very Dissatisfied"

                logging.debug(f"Total records (excluding header): {total_records}")
                logging.debug(f"Emotion Percentages: {emotion_percentages}")
                logging.debug(f"Satisfaction Score: {normalized_score}")
                logging.debug(f"Satisfaction Level: {satisfaction}")

                return render_template('summary.html', 
                                     emotion_percentages=emotion_percentages, 
                                     satisfaction=satisfaction)
                                     
            except Exception as e:
                logging.error(f"Error processing CSV file {file_path}: {e}")
                return render_template('summary.html', 
                                     emotion_percentages={}, 
                                     satisfaction=f"Error processing data: {str(e)}")

        return render_template('summary.html', 
                             emotion_percentages={}, 
                             satisfaction=None)

    except Exception as e:
        logging.error(f"Error in /summary route: {e}")
        return "Internal Server Error", 500

    
@app.route('/send_audio_signal', methods=['GET'])
def send_audio_signal():
 
    return jsonify({'send_audio': True})  

@app.route('/get_latest_emotion')
def get_latest_emotion():
  
    logging.debug(f"Latest emotions: {latest_emotions}")
    return jsonify(latest_emotions)

@app.route('/audio_debug', methods=['GET'])
def audio_debug():
    """Debug endpoint to check audio processing status"""
    try:
 
        model_loaded = voice_model is not None
        
     
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
        all_trips_data = get_all_trip_summaries(directory_path)


        satisfaction_counts = defaultdict(int)
        for trip in all_trips_data:
            satisfaction_counts[trip['satisfaction']] += 1

        monthly_trends = defaultdict(lambda: defaultdict(list))
        for trip in all_trips_data:
            try:

                file_path = os.path.join(directory_path, f"{trip['trip_id']}.csv")
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path, names=['timestamp', 'face_emotion', 'voice_emotion'], skiprows=1)
                    if not df.empty:
                        first_timestamp_str = df['timestamp'].iloc[0]
                        dt_object = pd.to_datetime(first_timestamp_str)
                        month_key = dt_object.strftime('%Y-%m')
                        monthly_trends[month_key][trip['satisfaction']].append(trip['normalized_score'])
            except Exception as e:
                logging.warning(f"Could not determine month for trip {trip['trip_id']}: {e}")

        # Prepare data for template
        monthly_summary_data = {}
        for month, sat_levels in sorted(monthly_trends.items()):
            total_trips_in_month = sum(len(scores) for scores in sat_levels.values())
            monthly_summary_data[month] = {
                'total_trips': total_trips_in_month,
                'satisfaction_breakdown': {level: len(scores) for level, scores in sat_levels.items()}
            }
        
        # Pass all_trips_data for detailed list display
        return render_template('monthly_summary.html', 
                               all_trips_data=all_trips_data, 
                               satisfaction_counts=satisfaction_counts,
                               monthly_summary_data=monthly_summary_data)

    except Exception as e:
        logging.error(f"Error in /monthly_summary route: {e}")
        return "Internal Server Error", 500

@app.route('/test_audio_processing', methods=['GET'])
def test_audio_processing():
   
    try:
        test_info = {
            'audio_processing_logic': {
                'confidence_threshold': 0.6,
                'uncertain_condition': 'confidence < 0.6',
                'na_condition': 'no audio data received',
                'examples': {
                    'high_confidence': 'happy (confidence: 0.85) -> happy',
                    'low_confidence': 'angry (confidence: 0.45) -> Uncertain',
                    'no_audio': 'no audio file -> N/A'
                }
            },
            'current_voice_model': 'audio_emotion_audio5.keras',
            'emotions_list': EMOTIONS,
            'latest_voice_emotion': latest_emotions.get('voice'),
            'collection_active': collection_details['start']
        }
        return jsonify(test_info)
    except Exception as e:
        logging.error(f"Error in test_audio_processing: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/test_satisfaction_calculation', methods=['GET'])
def test_satisfaction_calculation():
    """Test endpoint to demonstrate satisfaction calculation logic"""
    try:
        # Example emotion percentages
        example_emotions = {
            'happy': 60.0,
            'neutral': 30.0,
            'sad': 10.0
        }
        
        # Weights for each emotion
        weights = {
            'happy': 1.0,
            'neutral': 0.5,
            'surprise': 0.2,
            'sad': -0.5,
            'fear': -0.7,
            'disgust': -0.8,
            'angry': -1.0
        }
        
        # Calculate satisfaction score
        satisfaction_score = 0
        total_percent = 0
        
        for emotion, percent in example_emotions.items():
            weight = weights.get(emotion.lower(), 0)
            satisfaction_score += percent * weight
            total_percent += percent
        
        # Normalize score
        if total_percent > 0:
            normalized_score = satisfaction_score / total_percent
        else:
            normalized_score = 0
        
        # Determine satisfaction level
        if normalized_score >= 0.4:
            satisfaction = "Very Satisfied"
        elif normalized_score >= 0.1:
            satisfaction = "Satisfied"
        elif normalized_score >= -0.1:
            satisfaction = "Neutral"
        elif normalized_score >= -0.4:
            satisfaction = "Dissatisfied"
        else:
            satisfaction = "Very Dissatisfied"
        
        test_info = {
            'example_emotions': example_emotions,
            'weights': weights,
            'calculation': {
                'satisfaction_score': satisfaction_score,
                'total_percent': total_percent,
                'normalized_score': round(normalized_score, 3),
                'satisfaction_level': satisfaction
            },
            'formula': {
                'step1': 'satisfaction_score = sum(percent * weight) for each emotion',
                'step2': 'normalized_score = satisfaction_score / total_percent',
                'step3': 'determine satisfaction level based on normalized_score'
            }
        }
        
        return jsonify(test_info)
    except Exception as e:
        logging.error(f"Error in test_satisfaction_calculation: {e}")
        return jsonify({'error': str(e)}), 500

import csv
import os
from datetime import datetime

def write_to_csv(trip_id, face_emotion, voice_emotion):
    try:
  
        output_dir = "E:/Python/PBL5/PBL5/emotion_web1/data_emotion"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    
        filename = os.path.join(output_dir, f'{trip_id}.csv')

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

def calculate_trip_satisfaction(file_path: str) -> Dict[str, any]:
   
    trip_id = os.path.splitext(os.path.basename(file_path))[0]
    weights = {
        'happy': 1.0,
        'neutral': 0.5,
        'surprise': 0.2,
        'sad': -0.5,
        'fear': -0.7,
        'disgust': -0.8,
        'angry': -1.0
    }
    
    try:
        df = pd.read_csv(file_path, names=['timestamp', 'face_emotion', 'voice_emotion'], skiprows=1)
        df['face_emotion'] = df['face_emotion'].astype(str)

        if df.empty:
            return {'trip_id': trip_id, 'satisfaction': "No data", 'normalized_score': None}

        total_records = len(df)
        valid_face_records = len(df[df['face_emotion'] != 'Uncertain'])
        valid_voice_records = len(df[df['voice_emotion'] != 'N/A'])

        trip_duration = 0
        if total_records > 1:
            try:
                start_time = pd.to_datetime(df['timestamp'].iloc[0])
                end_time = pd.to_datetime(df['timestamp'].iloc[-1])
                trip_duration = (end_time - start_time).total_seconds() / 60
            except Exception as e:
                logging.warning(f"Could not parse trip duration for {trip_id}: {e}")
                trip_duration = 0

        filtered_df = df[df['face_emotion'].notna() & df['face_emotion'].isin(EMOTIONS)]
        emotion_percentages = {}
        if not filtered_df.empty:
            face_emotions = filtered_df['face_emotion'].value_counts(normalize=True) * 100
            emotion_percentages = face_emotions.to_dict()

        satisfaction_score = 0
        total_percent = 0
        for emotion, percent in emotion_percentages.items():
            weight = weights.get(emotion.lower(), 0)
            satisfaction_score += percent * weight
            total_percent += percent

        if total_percent > 0:
            normalized_score = satisfaction_score / total_percent
        else:
            normalized_score = 0

        if normalized_score >= 0.4:
            satisfaction = "Very Satisfied"
        elif normalized_score >= 0.1:
            satisfaction = "Satisfied"
        elif normalized_score >= -0.1:
            satisfaction = "Neutral"
        elif normalized_score >= -0.4:
            satisfaction = "Dissatisfied"
        else:
            satisfaction = "Very Dissatisfied"

        return {
            'trip_id': trip_id,
            'total_records': total_records,
            'valid_face_records': valid_face_records,
            'valid_voice_records': valid_voice_records,
            'trip_duration': round(trip_duration, 2),
            'emotion_percentages': emotion_percentages,
            'satisfaction': satisfaction,
            'normalized_score': round(normalized_score, 3)
        }

    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return {'trip_id': trip_id, 'satisfaction': "Error", 'normalized_score': None}

def get_all_trip_summaries(directory_path: str) -> List[Dict[str, any]]:
  
    all_trip_summaries = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(directory_path, file_name)
            summary = calculate_trip_satisfaction(file_path)
            all_trip_summaries.append(summary)
    return all_trip_summaries

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
