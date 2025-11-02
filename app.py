
import os
import sqlite3
import base64
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io
import threading

app = Flask(__name__)

# Upload folder (inside static/uploads)
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Database path
DB_PATH = "emotions.db"

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            image_path TEXT,
            emotion TEXT,
            confidence REAL,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Load emotion detection model
MODEL_PATH = "emotion_model_vortex.h5"
MODEL_LOCK = threading.Lock()

try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Predict emotion function
def predict_emotion(img):
    try:
        img = img.convert("L")
        img = img.resize((48, 48))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with MODEL_LOCK:
            preds = model.predict(img_array, verbose=0)[0]

        emotion_idx = np.argmax(preds)
        emotion = emotion_labels[emotion_idx]
        confidence = float(preds[emotion_idx])
        return emotion, confidence
    except Exception as e:
        print("Prediction error:", e)
        return None, None


@app.route("/")
def home():
    return render_template("index.html")


# Upload image route
@app.route("/upload", methods=["POST"])
def upload():
    try:
        name = request.form.get("name")
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded."})

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No image selected."})

        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        img = Image.open(file_path)
        emotion, confidence = predict_emotion(img)
        if emotion is None:
            return jsonify({"error": "Failed to analyze emotion."})

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO detections (name, image_path, emotion, confidence, timestamp) VALUES (?, ?, ?, ?, ?)",
                  (name, file_path, emotion, confidence, datetime.now().isoformat()))
        conn.commit()
        conn.close()

        return jsonify({
            "emotion": emotion,
            "confidence": round(confidence * 100, 2),
            "image_url": url_for("static", filename=f"uploads/{filename}")
        })
    except Exception as e:
        print("❌ Upload route error:", e)
        return jsonify({"error": str(e)})


# Webcam route (fixed to handle JSON correctly)
@app.route("/webcam", methods=["POST"])
def webcam():
    try:
        data = request.get_json()
        print("Received data:", data)  # Debugging
        if not data:
            return jsonify({"error": "No JSON data received."})

        name = data.get("name")
        data_url = data.get("image")
        if not data_url:
            return jsonify({"error": "No webcam data received."})

        image_data = base64.b64decode(data_url.split(",")[1])
        img = Image.open(io.BytesIO(image_data))

        emotion, confidence = predict_emotion(img)
        if emotion is None:
            return jsonify({"error": "Failed to analyze webcam image."})

        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_webcam.jpg"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        img.save(file_path)

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO detections (name, image_path, emotion, confidence, timestamp) VALUES (?, ?, ?, ?, ?)",
                  (name, file_path, emotion, confidence, datetime.now().isoformat()))
        conn.commit()
        conn.close()

        return jsonify({
            "emotion": emotion,
            "confidence": round(confidence * 100, 2),
            "image_url": url_for("static", filename=f"uploads/{filename}")
        })
    except Exception as e:
        print("❌ Webcam route error:", e)
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
