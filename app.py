from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask_cors import CORS
from io import BytesIO
import numpy as np

app = Flask(__name__)
CORS(app)

# ----------------- Load federated model ------------------
MODEL_PATH = "potato_federated_model_3classes.h5"
print("🔄 Loading model...")
model = load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully")

# Class index mapping
class_indices = {
    0: "Early Blight",
    1: "Late Blight",
    2: "Healthy"
}

# Thresholds
NOT_LEAF_CONFIDENCE_THRESHOLD = 0.60   # model confidence
LEAF_COLOR_THRESHOLD = 5.0             # how "green/leaf-like" the image must be


# ----------------- Helper: leaf detection ------------------
def is_leaf_like(pil_img):
    """
    Very simple heuristic:
    - convert image to numpy
    - compute green-ness score: G - (R + B)/2
    - if average score is too low, it's probably not a leaf
    """
    arr = np.array(pil_img).astype("float32")  # (H, W, 3)

    if arr.ndim != 3 or arr.shape[2] != 3:
        return False  # not a normal RGB image

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    green_score = np.mean(g - (r + b) / 2.0)

    # You can print this to tune the threshold:
    # print("green_score:", green_score)

    return green_score > LEAF_COLOR_THRESHOLD


# ----------------- Routes ------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # 1) Check if file is present
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        # 2) Read bytes and load image
        img_bytes = file.read()
        pil_img = image.load_img(BytesIO(img_bytes), target_size=(128, 128))

        # 3) Leaf check (before running the model)
        leaf_flag = is_leaf_like(pil_img)

        # 4) Preprocess for model
        img_array = image.img_to_array(pil_img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # 5) Model prediction
        preds = model.predict(img_array)
        max_prob = float(np.max(preds))
        pred_class = int(np.argmax(preds))

        # 6) Combine both checks
        if (not leaf_flag) or (max_prob < NOT_LEAF_CONFIDENCE_THRESHOLD):
            # either looks non-leaf OR model not confident
            return jsonify({
                "label": "Not a leaf",
                "probabilities": preds.tolist(),
                "confidence": max_prob,
                "leaf_like": bool(leaf_flag)
            })

        # otherwise, treat as true leaf and use disease class
        label = class_indices.get(pred_class, "Unknown")

        return jsonify({
            "label": label,
            "probabilities": preds.tolist(),
            "confidence": max_prob,
            "leaf_like": bool(leaf_flag)
        })

    except Exception as e:
        print("❌ Error during prediction:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
