from flask import Flask, render_template, Response, request, jsonify, send_file
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort # DeepSort might have issues with ONNX/TensorRT if not careful
import cv2
import threading
import time
import numpy as np
from PIL import Image
import io
import os # <--- Ensure this is imported

# Try to import torch for torch.compile
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not found. torch.compile optimization will be skipped.")


app = Flask(__name__)

# Base directory for models
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Define available models and their preferred optimized formats
# Order of preference for loading: .engine -> .onnx -> .pt (with torch.compile)
MODELS = {
    "yolov8n": {
        "pt": os.path.join(MODELS_DIR, "yolov8n.pt"),
        "onnx": os.path.join(MODELS_DIR, "yolov8n.onnx"),
        "engine": os.path.join(MODELS_DIR, "yolov8n.engine")
    },
    "yolov8m": {
        "pt": os.path.join(MODELS_DIR, "yolov8m.pt"),
        "onnx": os.path.join(MODELS_DIR, "yolov8m.onnx"),
        "engine": os.path.join(MODELS_DIR, "yolov8m.engine")
    },
    "yolov8l": {
        "pt": os.path.join(MODELS_DIR, "yolov8l.pt"),
        "onnx": os.path.join(MODELS_DIR, "yolov8l.onnx"),
        "engine": os.path.join(MODELS_DIR, "yolov8l.engine")
    },
    "yolov8x": {
        "pt": os.path.join(MODELS_DIR, "yolov8x.pt"),
        "onnx": os.path.join(MODELS_DIR, "yolov8x.onnx"),
        "engine": os.path.join(MODELS_DIR, "yolov8x.engine")
    },
    # If "combined" is a custom .pt model, handle it similarly
    "combined": {
        "pt": os.path.join(MODELS_DIR, "yolov8_combined.pt"),
        "onnx": os.path.join(MODELS_DIR, "yolov8_combined.onnx"),
        "engine": os.path.join(MODELS_DIR, "yolov8_combined.engine")
    }
}

current_model_name = "yolov8n"
current_model = None # Will be initialized by load_model function

is_detecting = False
log_data = []

# Define a color palette for different classes
COLORS = {
    "person": (0, 200, 255),  # Orange-ish for persons
    "bicycle": (255, 0, 0),
    "car": (0, 0, 255),
    "motorcycle": (255, 255, 0),
    "airplane": (0, 255, 255),
    "bus": (255, 0, 255),
    "train": (128, 0, 0),
    "truck": (0, 128, 0),
    "boat": (0, 0, 128),
    # Add more classes and their desired colors here (use BGR format for OpenCV)
}

# Function to get a color for a given class. If not in COLORS, use a default.
def get_color_for_class(class_name):
    return COLORS.get(class_name, (0, 255, 0)) # Default to green (BGR)

def load_model(name):
    """Loads the YOLO model, preferring optimized formats."""
    global current_model_name, current_model
    if name not in MODELS:
        print(f"Model '{name}' not found in available models.")
        return False

    model_paths = MODELS[name]
    loaded_successfully = False

    # 1. Try to load TensorRT .engine model first (fastest on NVIDIA GPUs)
    if os.path.exists(model_paths["engine"]):
        try:
            current_model = YOLO(model_paths["engine"])
            current_model_name = name
            print(f"Loaded {name} model from TensorRT engine: {model_paths['engine']}")
            loaded_successfully = True
        except Exception as e:
            print(f"Failed to load TensorRT engine {model_paths['engine']}: {e}")
            print("Attempting to load ONNX or PyTorch model instead.")

    # 2. If TensorRT failed or not available, try ONNX .onnx model
    if not loaded_successfully and os.path.exists(model_paths["onnx"]):
        try:
            current_model = YOLO(model_paths["onnx"])
            current_model_name = name
            print(f"Loaded {name} model from ONNX: {model_paths['onnx']}")
            loaded_successfully = True
        except Exception as e:
            print(f"Failed to load ONNX model {model_paths['onnx']}: {e}")
            print("Attempting to load PyTorch model instead.")

    # 3. If both failed, try PyTorch .pt model
    if not loaded_successfully and os.path.exists(model_paths["pt"]):
        try:
            current_model = YOLO(model_paths["pt"])
            current_model_name = name
            print(f"Loaded {name} model from PyTorch: {model_paths['pt']}")

            # Apply torch.compile if PyTorch is available and model is a PyTorch model
            if TORCH_AVAILABLE and isinstance(current_model.model, torch.nn.Module):
                try:
                    # Compile the model for faster inference (PyTorch 2.0+)
                    # Use fullgraph=True for maximum optimization, but can cause graph breaks.
                    # mode='reduce-overhead' is a good balance.
                    current_model.model = torch.compile(current_model.model, mode='reduce-overhead')
                    print(f"Applied torch.compile to {name} PyTorch model.")
                except Exception as e:
                    print(f"Failed to apply torch.compile to {name} model: {e}")
            loaded_successfully = True
        except Exception as e:
            print(f"Failed to load PyTorch model {model_paths['pt']}: {e}")
            print(f"Could not load model '{name}' from any available format.")

    if not loaded_successfully:
        print(f"Error: No valid model file found for '{name}' in {MODELS_DIR}. Please check your model paths.")
        # Fallback to a default if nothing loaded, or raise error
        # For robustness, we might want to ensure a model is always loaded or handle this gracefully.
        return False
    return True

# Initialize a model on startup
# This line is intentionally kept here as a global initialization,
# but also needs to be called explicitly in __main__ for the first run.
load_model(current_model_name)

# DeepSort Tracker initialization (Keep outside generate_frames for persistent tracking)
# Note: DeepSort usually works with numerical detections (xyxy, conf, class_id).
# Ultralytics model.track() internally handles DeepSort compatible output.
# If you load ONNX/TensorRT, model.track might not be available or behave differently.
# For simplicity, if using .engine or .onnx, you might revert to model.predict + manual DeepSort update.
# For now, we'll assume model.track still works with the loaded model types, as Ultralytics often wraps them.
tracker = DeepSort(max_age=30) # Re-initialize if model changes? Not strictly necessary unless tracking logic depends on model type.

def generate_frames():
    global is_detecting, log_data, current_model

    cap = cv2.VideoCapture(0) # Default webcam

    # Set camera resolution (optional, but good for consistent input size)
    # Adjust these values based on your camera and desired performance/accuracy trade-off
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame from camera. Exiting video feed.")
            break

        if current_model is None:
            # If no model is loaded, just show the raw frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.1) # Avoid busy-waiting
            continue

        annotated = frame.copy() # Always make a copy for annotation

        if is_detecting:
            # Use model.track for live video with ByteTrack, it handles the output for deepsort
            # conf and iou thresholds for better detection of multiple objects.
            # imgsz for consistent input size, matching what might be optimized for.
            results = current_model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=0.25, # Adjusted slightly for potentially more detections
                iou=0.45,  # Adjusted slightly for more detections in crowds
                imgsz=640 # Ensure consistent input size for optimized models
            )[0]

            if results.boxes is not None:
                detections = results.boxes
                for box in detections:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = current_model.names[cls]
                    track_id = int(box.id[0]) if box.id is not None else -1 # Get track_id

                    # Get color for the current label
                    color = get_color_for_class(label)

                    cv2.rectangle(annotated, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                    text = f"{label} ID:{track_id} {conf:.2f}" if track_id != -1 else f"{label} {conf:.2f}"
                    cv2.putText(annotated, text, (xyxy[0], xyxy[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                log_data.append(f"Detected {len(detections)} objects.")
            else:
                log_data.append("No detections.")

            ret, buffer = cv2.imencode('.jpg', annotated)
        else:
            ret, buffer = cv2.imencode('.jpg', frame) # Show raw frame if not detecting

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()


@app.route('/')
def index():
    return render_template('index_realtime.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    global is_detecting
    action = request.form.get('action')
    is_detecting = (action == 'start')
    print(f"Detection toggled to: {is_detecting}")
    return '', 204


@app.route('/set_model', methods=['POST'])
def set_model():
    global current_model_name
    new_model_name = request.form.get('model')
    print(f"Attempting to set model to: {new_model_name}")
    if new_model_name != current_model_name: # Only reload if different
        if load_model(new_model_name):
            print(f"Model successfully switched to: {current_model_name}")
            return jsonify(status="success", message=f"Model set to {current_model_name}")
        else:
            print(status="error", message=f"Failed to load model {new_model_name}")
            return jsonify(status="error", message=f"Failed to load model {new_model_name}"), 400
    else:
        print(f"Model already set to: {current_model_name}. No change needed.")
        return jsonify(status="info", message=f"Model already set to {current_model_name}")


@app.route('/detection_log')
def detection_log():
    return jsonify(logs=log_data[-20:])


@app.route("/detect_image", methods=["POST"])
def detect_image():
    global current_model
    if 'image' not in request.files:
        return "No image uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "Empty filename", 400

    if current_model is None:
        return "No model loaded for image detection. Please select a model.", 503

    try:
        file.stream.seek(0)
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return f"Failed to read image: {str(e)}", 400

    image_np = np.array(image)
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Use model.predict for single image inference
    results = current_model.predict(
        source=image_cv2,
        conf=0.25, # Adjusted for more detections
        iou=0.45,  # Adjusted for more detections in crowds
        imgsz=640  # Consistent input size
    )[0]
    annotated = image_cv2.copy()

    if results.boxes is not None:
        detections = results.boxes
        for box in detections:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = current_model.names[cls]

            # Get color for the current label
            color = get_color_for_class(label)

            cv2.rectangle(annotated, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
            cv2.putText(annotated, f"{label} {conf:.2f}", (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    _, buffer = cv2.imencode('.jpg', annotated)
    return send_file(io.BytesIO(buffer.tobytes()), mimetype='image/jpeg')


if __name__ == '__main__':
    # Initial model load before starting the Flask app
    # This line ensures current_model is initialized when the app starts
    print(f"Initializing with model: {current_model_name}")
    load_model(current_model_name) # <--- THIS LINE WAS MISSING AND HAS BEEN RE-ADDED

    # Define the port, reading from environment variable 'PORT' or defaulting to 5000
    port = int(os.getenv('PORT', 5000)) # Reads PORT env var, defaults to 5000

    app.run(debug=True, host='0.0.0.0', port=port) # Listen on all interfaces