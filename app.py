# app.py - Final Render-ready Flask app with image and webcam support, logging, and database storage

import os
import base64
import logging
import sqlite3
from io import BytesIO
from flask import Flask, render_template, request, jsonify, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Configure paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
MODEL_PATH = os.path.join(BASE_DIR, 'emotion_model_vortex.h5')
DB_PATH = os.path.join(BASE_DIR, 'emotions.db')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and database
try:
    logger.info(f"Loading model from {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    logger.info("Model loaded successfully.")
except Exception as e:
    model = None
    logger.error(f"Failed to load model: {e}")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT,
                        filename TEXT,
                        emotion TEXT,
                        confidence REAL
                    )''')
    conn.commit()
    conn.close()

init_db()

# Emotion labels (update if your model uses a different order)
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        name = request.form.get('name')
        file = request.files['image']

        if not file:
            return jsonify({'error': 'No image uploaded'}), 400

        # Save image
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Preprocess
        img = Image.open(filepath).convert('L').resize((48, 48))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        preds = model.predict(img_array)
        emotion_idx = np.argmax(preds[0])
        emotion = EMOTIONS[emotion_idx]
        confidence = float(np.max(preds[0]) * 100)

        # Save to DB
        conn = sqlite3.connect(DB_PATH)
        conn.execute('INSERT INTO users (name, filename, emotion, confidence) VALUES (?, ?, ?, ?)',
                     (name, filename, emotion, confidence))
        conn.commit()
        conn.close()

        image_url = url_for('static', filename=f'uploads/{filename}')

        return jsonify({'emotion': emotion, 'confidence': confidence, 'image_url': image_url})
    except Exception as e:
        logger.error(f"Upload analysis failed: {e}")
        return jsonify({'error': 'Failed to analyze image'}), 500

@app.route('/webcam', methods=['POST'])
def analyze_webcam():
    try:
        data = request.get_json()
        name = data.get('name')
        image_data = data.get('image')

        if not image_data:
            return jsonify({'error': 'No webcam data received'}), 400

        image_data = image_data.split(',')[1]
        img_bytes = base64.b64decode(image_data)
        img = Image.open(BytesIO(img_bytes)).convert('L').resize((48, 48))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        preds = model.predict(img_array)
        emotion_idx = np.argmax(preds[0])
        emotion = EMOTIONS[emotion_idx]
        confidence = float(np.max(preds[0]) * 100)

        filename = f'webcam_{name}.jpg'
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        img.save(filepath)

        conn = sqlite3.connect(DB_PATH)
        conn.execute('INSERT INTO users (name, filename, emotion, confidence) VALUES (?, ?, ?, ?)',
                     (name, filename, emotion, confidence))
        conn.commit()
        conn.close()

        image_url = url_for('static', filename=f'uploads/{filename}')

        return jsonify({'emotion': emotion, 'confidence': confidence, 'image_url': image_url})
    except Exception as e:
        logger.error(f"Webcam analysis failed: {e}")
        return jsonify({'error': 'Failed to analyze emotion'}), 500

@app.route('/health')
def health():
    status = 'healthy' if model else 'unhealthy'
    return jsonify({'status': status})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port)
