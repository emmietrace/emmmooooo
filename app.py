# -------------------------------------------------
# NEW IMPORTS (add at the top with the other imports)
# -------------------------------------------------
from PIL import Image, UnidentifiedImageError
import io
import matplotlib.pyplot as plt          # optional – for debugging / visualisation
from mtcnn import MTCNN
import tensorflow as tf                 # already imported but keep for tflite later

# -------------------------------------------------
# GLOBAL FACE DETECTOR (shared across requests)
# -------------------------------------------------
face_detector = MTCNN()   # creates a TensorFlow-based detector (GPU-friendly)

# -------------------------------------------------
# Helper: load image from bytes (used for both upload & webcam)
# -------------------------------------------------
def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode bytes → OpenCV BGR image (handles Pillow errors gracefully)."""
    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except UnidentifiedImageError:
        raise ValueError("Uploaded file is not a valid image")

# -------------------------------------------------
# Updated preprocess_image – now works on a **face crop** if detected
# -------------------------------------------------
def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    1. Detect face(s) with MTCNN
    2. If a face is found → crop + resize to 48×48 grayscale
    3. Else → fall back to whole-image (keeps original behaviour)
    """
    # ---- 1. face detection -------------------------------------------------
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # MTCNN expects RGB
    detections = face_detector.detect_faces(rgb)

    if detections:
        # pick the largest face (most confident)
        box = max(detections, key=lambda d: d['confidence'])['box']
        x, y, w, h = box
        # add a small margin (10% of the box size)
        margin = int(max(w, h) * 0.1)
        x = max(x - margin, 0)
        y = max(y - margin, 0)
        face = img[y:y + h + 2*margin, x:x + w + 2*margin]
    else:
        face = img                     # no face → use whole image

    # ---- 2. standard 48×48 grayscale preprocessing -------------------------
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if len(face.shape) == 3 else face
    resized = cv2.resize(gray, (48, 48))
    norm = resized.astype('float32') / 255.0
    return np.expand_dims(norm, axis=[-1, 0])   # (1, 48, 48, 1)

# -------------------------------------------------
# /upload route – now uses Pillow to read the file
# -------------------------------------------------
@app.route('/upload', methods=['POST'])
def upload():
    name = request.form.get('name')
    file = request.files.get('image')

    if not (file and name):
        return jsonify({'error': 'Missing name or image'}), 400

    # ---- read image with Pillow (safer than cv2.imdecode for some formats) ----
    img_bytes = file.read()
    try:
        img = load_image_from_bytes(img_bytes)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    # ---- save original (full size) for UI -----------------------------------
    filename = f"{int(datetime.now().timestamp())}_{name}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    cv2.imwrite(filepath, img)

    # ---- prediction ---------------------------------------------------------
    pred = MODEL.predict(preprocess_image(img), verbose=0)[0]
    idx = np.argmax(pred)
    emotion = EMOTIONS[idx]
    conf = float(pred[idx])

    save_to_db(name, f"static/uploads/{filename}", emotion, conf)

    return jsonify({
        'emotion': emotion,
        'confidence': round(conf, 3),
        'image_url': f"static/uploads/{filename}"
    })

# -------------------------------------------------
# /webcam route – unchanged except using the same preprocess_image()
# -------------------------------------------------
@app.route('/webcam', methods=['POST'])
def webcam():
    payload = request.get_json()
    name = payload.get('name')
    data_url = payload.get('image', '')

    if not (name and data_url):
        return jsonify({'error': 'Missing name or image'}), 400

    # strip data-url header
    img_bytes = base64.b64decode(data_url.split(',')[1])
    img = load_image_from_bytes(img_bytes)

    filename = f"webcam_{int(datetime.now().timestamp())}_{name}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    cv2.imwrite(filepath, img)

    pred = MODEL.predict(preprocess_image(img), verbose=0)[0]
    idx = np.argmax(pred)
    emotion = EMOTIONS[idx]
    conf = float(pred[idx])

    save_to_db(name, f"static/uploads/{filename}", emotion, conf)

    return jsonify({
        'emotion': emotion,
        'confidence': round(conf, 3),
        'image_url': f"static/uploads/{filename}"
    })
