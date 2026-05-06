"""
ASL Recognition Web Server (Image-Based CNN Version)
MobileNetV3 vs DenseNet121
"""

import os
import cv2
import time
import numpy as np
import tensorflow as tf
from flask import Flask, Response, render_template, jsonify, request
from flask_cors import CORS
from flask import send_from_directory
import threading

from asl_modules import (
    SpeechEngine, TextBuffer, TranslationMode,
    CLASS_NAMES, DEFAULT_IMG_SIZE, crop_hand
)
from tensorflow.keras.layers import GlobalAveragePooling2D

app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
CORS(app)

class ImageBasedASLClassifier:
    """Trained image-based ASL classifier."""

    def __init__(self,
                 model_mobilenet_path: str = "saved_models/mobilenetv3_model.keras",
                 model_densenet_path: str = "saved_models/densenet121_model.keras",
                 classes_path: str = "saved_models/landmark_classifier_classes.npy"):

        self.model_mobilenet = None
        self.model_densenet = None
        self.class_names = CLASS_NAMES
        self.active_model = 'mobilenetv3'

        if os.path.exists(model_mobilenet_path):
            self.model_mobilenet = tf.keras.models.load_model(model_mobilenet_path, custom_objects={'GlobalAveragePooling2D': GlobalAveragePooling2D})
            print("[Classifier] MobileNetV3 loaded & ready!")

        if os.path.exists(model_densenet_path):
            self.model_densenet = tf.keras.models.load_model(model_densenet_path, custom_objects={'GlobalAveragePooling2D': GlobalAveragePooling2D})
            print("[Classifier] DenseNet121 loaded & ready!")

        if os.path.exists(classes_path):
            self.class_names = list(np.load(classes_path, allow_pickle=True))

    def predict(self, image: np.ndarray) -> tuple:
        """Predict the sign from an image."""
        if self.active_model == 'mobilenetv3':
            model = self.model_mobilenet
        else:
            model = self.model_densenet
            
        if model is None:
            return "?", 0.0

        pred = model.predict(image, verbose=0)[0]
        idx = np.argmax(pred)
        return self.class_names[idx], float(pred[idx])
        
    def set_model(self, model_name: str):
        if model_name in ['mobilenetv3', 'densenet121']:
            self.active_model = model_name

class ASLWebProcessor:
    MIN_CONFIDENCE = 0.85
    REQUIRED_STREAK = 6

    def __init__(self):
        self.classifier = None
        self.text_buffer = None
        self.speech_engine = None
        
        self.img_size = DEFAULT_IMG_SIZE
        self.temporal_streak = []
        
        self.current_letter = ""
        self.current_confidence = 0.0
        self.current_word = ""
        self.sentence = ""
        self.pending_info = ("", 0, 0)
        self.validation_status = "waiting"
        
        self.is_running = False
        self.lock = threading.Lock()
        
        self._initialize()
    
    def _initialize(self):
        print("[Server] Initializing Image-Based ASL processor...")
        self.text_buffer = TextBuffer(mode=TranslationMode.BALANCED)
        self.speech_engine = SpeechEngine()
        self.classifier = ImageBasedASLClassifier()
        self.is_running = True
    
    def check_temporal_consistency(self, letter: str, confidence: float) -> tuple:
        if not self.temporal_streak or self.temporal_streak[-1][0] != letter:
            self.temporal_streak = [(letter, confidence)]
        else:
            self.temporal_streak.append((letter, confidence))
        
        if len(self.temporal_streak) > 15:
            self.temporal_streak = self.temporal_streak[-15:]
            
        streak_count = len(self.temporal_streak)
        if streak_count >= self.REQUIRED_STREAK:
            avg_conf = np.mean([c for _, c in self.temporal_streak])
            return True, streak_count, avg_conf
        return False, streak_count, confidence
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        with self.lock:
            # Preprocess the frame
            img = cv2.resize(frame, self.img_size)
            img = crop_hand(img)
            img = np.expand_dims(img, axis=0)
            img = img / 255.0

            # Predict
            letter, confidence = self.classifier.predict(img)

            is_consistent, streak, avg_conf = self.check_temporal_consistency(letter, confidence)

            if is_consistent and streak >= self.REQUIRED_STREAK and avg_conf >= self.MIN_CONFIDENCE:
                display_letter = letter
                display_conf = avg_conf
                engine_used = f"{self.classifier.active_model} (Verified)"
                status = "accepted"
                color = (0, 255, 0) # Green
                self.text_buffer.add_letter(letter, avg_conf)
            else:
                display_letter = letter
                display_conf = avg_conf
                engine_used = f"{self.classifier.active_model} (Tracking...)"
                status = f"streak {streak}/{self.REQUIRED_STREAK}"
                color = (0, 255, 255) # Yellow
                self.text_buffer._pending_count = 0

            # Update state for web UI
            self.current_letter = display_letter
            self.current_confidence = display_conf
            self.validation_status = status
            self.current_word = self.text_buffer.get_current_word()
            self.sentence = self.text_buffer.get_sentence()
            self.pending_info = self.text_buffer.get_pending_info()
            
            # Draw on frame
            cv2.rectangle(frame, (10, 10), (200, 120), color, -1)
            cv2.putText(frame, display_letter, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5)
            cv2.putText(frame, f"{display_conf:.0%}", (210, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, engine_used, (210, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, status, (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            
            return frame
    
    def get_state(self) -> dict:
        with self.lock:
            return {
                "letter": self.current_letter,
                "confidence": self.current_confidence,
                "word": self.current_word,
                "sentence": self.sentence,
                "pending": {"letter": self.pending_info[0], "count": self.pending_info[1], "required": self.pending_info[2]},
                "validation": {"status": self.validation_status, "streak": len(self.temporal_streak), "required_streak": self.REQUIRED_STREAK},
                "mode": self.text_buffer.mode.value if self.text_buffer else "unknown",
                "model": self.classifier.active_model if self.classifier else "unknown"
            }
    
    def set_mode(self, mode: str):
        mode_map = {"instant": TranslationMode.INSTANT, "balanced": TranslationMode.BALANCED, "accurate": TranslationMode.ACCURATE, "strict": TranslationMode.STRICT}
        if mode in mode_map:
            if self.text_buffer: self.text_buffer.set_mode(mode_map[mode])
            if mode == "instant": self.REQUIRED_STREAK = 3; self.MIN_CONFIDENCE = 0.70
            elif mode == "balanced": self.REQUIRED_STREAK = 6; self.MIN_CONFIDENCE = 0.85
            elif mode == "accurate": self.REQUIRED_STREAK = 10; self.MIN_CONFIDENCE = 0.90
            elif mode == "strict": self.REQUIRED_STREAK = 15; self.MIN_CONFIDENCE = 0.95
            self.temporal_streak = []
            
    def set_model(self, model_name: str):
        if self.classifier:
            self.classifier.set_model(model_name)
    
    def clear_text(self):
        if self.text_buffer:
            self.text_buffer.clear_all()
            self.current_word = ""
            self.sentence = ""
    
    def cleanup(self):
        self.is_running = False
        if self.speech_engine: self.speech_engine.shutdown()

processor = None
def get_processor():
    global processor
    if processor is None: processor = ASLWebProcessor()
    return processor

def generate_frames():
    proc = get_processor()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    while proc.is_running:
        success, frame = cap.read()
        if not success: break
        frame = cv2.flip(frame, 1)
        frame = proc.process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret: continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/state')
def get_state(): return jsonify(get_processor().get_state())

@app.route('/api/mode', methods=['POST'])
def set_mode():
    mode = request.get_json().get('mode', 'accurate')
    get_processor().set_mode(mode)
    return jsonify({"status": "ok", "mode": mode})

@app.route('/api/set_model', methods=['POST'])
def set_model():
    model_name = request.get_json().get('model', 'mobilenetv3')
    get_processor().set_model(model_name)
    return jsonify({"status": "ok", "model": model_name})

@app.route('/api/clear', methods=['POST'])
def clear_text():
    get_processor().clear_text()
    return jsonify({"status": "ok"})

@app.route('/api/speak', methods=['POST'])
def speak_text():
    proc = get_processor()
    if proc.speech_engine and proc.sentence:
        threading.Thread(target=proc.speech_engine.speak, args=(proc.sentence,), daemon=True).start()
    return jsonify({"status": "ok"})

@app.route('/get_generated_image/<filename>')
def get_generated_image(filename):
    """Mengambil gambar dari folder saved_generate."""
    return send_from_directory('saved_generate', filename)

if __name__ == '__main__':
    print(f"\n{'='*60}\n  ASL Web Server (Image-Based CNN)\n{'='*60}\n  Buka browser: http://localhost:5000\n{'='*60}\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
