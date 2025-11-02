# app.py — Optimized Local Version
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
import sqlite3
from datetime import datetime
from tensorflow.keras.models import load_model
import base64
import threading

# -------------------------------
# CONFIGURATION
# -------------------------------
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = 'emotion_model_vortex.h5'
DB_PATH = 'emotions.db'
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# -------------------------------
# LOAD MODEL (Thread-safe)
# -------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")

print("✅ Loading model...")
MODEL = load_model(MODEL_PATH)
MODEL_LOCK = threading.Lock()
print("✅ Model loaded successfully!")

# -------------------------------
# DATABASE FUNCTIONS
# -------------------------------
def init_db():
    """Initialize the SQLite database."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                image_path TEXT,
                emotion TEXT,
                confidence REAL,
                timestamp TEXT
            )
        ''')

def save_to_db(name, image_path, emotion, confidence):
    """Save prediction results into database."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            INSERT INTO detections (name, image_path, emotion, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (name, image_path, emotion, confidence, timestamp))

# Initialize DB at startup
init_db()

# -------------------------------
# IMAGE PREPROCESSING
# -------------------------------
def preprocess_image(img):
    """Convert image to grayscale, resize, normalize, and reshape for CNN."""
    if img is None:
        raise ValueError("Invalid image input for preprocessing.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    resized = cv2.resize(gray, (48, 48))
    norm = resized.astype('float32') / 255.0
    return np.expand_dims(norm, axis=(0, -1))

# -------------------------------
# PREDICTION FUNCTION
# -------------------------------
def predict_emotion(img):
    """Run model prediction thread-safely."""
    processed = preprocess_image(img)
    with MODEL_LOCK:
        preds = MODEL.predict(processed, verbose=0)[0]
    idx = np.argmax(preds)
    return EMOTIONS[idx], float(preds[idx])

# -------------------------------
# ROUTES
# -------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        name = request.form.get('name', '').strip()
        file = request.files.get('image')

        if not name or not file:
            return jsonify({'error': 'Missing name or image'}), 400

        # Save uploaded image
        filename = f"{int(datetime.now().timestamp())}_{name}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Predict
        img = cv2.imread(filepath)
        emotion, confidence = predict_emotion(img)

        # Save to DB
        rel_path = f"static/uploads/{filename}"
        save_to_db(name, rel_path, emotion, confidence)

        return jsonify({
            'emotion': emotion,
            'confidence': round(confidence, 3),
            'image_url': rel_path
        })

    except Exception as e:
        print("❌ Upload error:", e)
        return jsonify({'error': str(e)}), 500


@app.route('/webcam', methods=['POST'])
def webcam():
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        image_data = data.get('image', '')

        if not name or not image_data:
            return jsonify({'error': 'Missing name or image data'}), 400

        # Decode base64 webcam image
        nparr = np.frombuffer(base64.b64decode(image_data.split(',')[1]), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Save frame
        filename = f"webcam_{int(datetime.now().timestamp())}_{name}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(filepath, img)

        # Predict
        emotion, confidence = predict_emotion(img)

        # Save to DB
        rel_path = f"static/uploads/{filename}"
        save_to_db(name, rel_path, emotion, confidence)

        return jsonify({
            'emotion': emotion,
            'confidence': round(confidence, 3),
            'image_url': rel_path
        })

    except Exception as e:
        print("❌ Webcam error:", e)
        return jsonify({'error': str(e)}), 500


# -------------------------------
# RUN APP
# -------------------------------
if __name__ == '__main__':
    print("🚀 Emotion Vortex running at: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop")
    app.run(host='127.0.0.1', port=5000, debug=True)
