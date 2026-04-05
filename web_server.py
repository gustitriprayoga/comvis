"""
ASL Recognition Web Server
Flask application with video streaming and real-time ASL recognition.
Modern web interface with dark theme and sleek design.
"""

import os
import cv2
import time
import json
import numpy as np
from flask import Flask, Response, render_template, jsonify, request
from flask_cors import CORS
import threading

import tensorflow as tf

from asl_modules import (
    HandDetector, SpeechEngine, TextBuffer, TranslationMode,
    LandmarkASLClassifier, CLASS_NAMES, DEFAULT_IMG_SIZE
)

app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
CORS(app)


class ASLWebProcessor:
    """ASL Recognition processor for web streaming with accuracy improvements."""
    
    # Validation thresholds (BALANCED - not too fast, not too slow)
    MIN_CONFIDENCE = 0.85      # Minimum confidence to accept
    MIN_GAP = 0.15             # Minimum gap between top-1 and top-2
    REQUIRED_STREAK = 6        # Required consecutive same predictions (slowed down)
    STABILITY_THRESHOLD = 0.03 # Hand movement threshold (stricter)
    
    def __init__(self, model_path: str = "saved_models/asl_model_best.keras"):
        self.model = None
        self.hand_detector = None
        self.text_buffer = None
        self.speech_engine = None
        
        self.img_size = DEFAULT_IMG_SIZE
        self.prediction_history = []
        self.landmarks_history = []
        self.temporal_streak = []
        
        self.current_letter = ""
        self.current_confidence = 0.0
        self.current_word = ""
        self.sentence = ""
        self.pending_info = ("", 0, 0)
        self.validation_status = "waiting"
        
        self.is_running = False
        self.lock = threading.Lock()
        
        self._initialize(model_path)
    
    def _initialize(self, model_path: str):
        """Initialize all components."""
        print("[Server] Initializing ASL processor with accuracy improvements...")
        
        # Load model
        if os.path.exists(model_path):
            print(f"[Server] Loading model: {model_path}")
            self.model = tf.keras.models.load_model(model_path)
            print("[Server] Model loaded successfully")
        else:
            print(f"[Server] Warning: Model not found at {model_path}")
        
        # Initialize hand detector
        self.hand_detector = HandDetector(
            max_num_hands=2, 
            min_detection_confidence=0.7
        )
        
        # Initialize text buffer - BALANCED mode for better accuracy
        self.text_buffer = TextBuffer(mode=TranslationMode.BALANCED)
        
        # Initialize speech engine
        self.speech_engine = SpeechEngine()
        
        # Create CLAHE for preprocessing
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Initialize LANDMARK-BASED classifier (PRIMARY - more accurate!)
        self.landmark_classifier = LandmarkASLClassifier()
        print("[Server] Landmark-based ASL classifier initialized")
        
        self.is_running = True
        print("[Server] ASL processor ready with LANDMARK-BASED classification")
    
    def preprocess_hand(self, hand_image: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing with CLAHE lighting normalization."""
        # Resize
        resized = cv2.resize(hand_image, self.img_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Apply CLAHE to L channel for lighting normalization
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        normalized_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Convert to RGB and normalize
        rgb = cv2.cvtColor(normalized_bgr, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        
        return np.expand_dims(normalized, axis=0)
    
    def smart_classify(self, predictions: np.ndarray) -> tuple:
        """
        Smart classification with multi-stage validation.
        Returns (letter, confidence, is_valid, validation_status)
        """
        sorted_idx = np.argsort(predictions)[::-1]
        top1_idx = sorted_idx[0]
        top2_idx = sorted_idx[1]
        
        top1_conf = float(predictions[top1_idx])
        top2_conf = float(predictions[top2_idx])
        gap = top1_conf - top2_conf
        
        letter = CLASS_NAMES[top1_idx] if top1_idx < len(CLASS_NAMES) else "?"
        
        # Stage 1: Confidence check
        if top1_conf < self.MIN_CONFIDENCE:
            return letter, top1_conf, False, f"low_conf ({top1_conf:.0%})"
        
        # Stage 2: Gap check
        if gap < self.MIN_GAP:
            return letter, top1_conf, False, f"ambiguous (gap {gap:.0%})"
        
        return letter, top1_conf, True, "passed"
    
    def check_temporal_consistency(self, letter: str, confidence: float) -> tuple:
        """
        Check if same letter detected for N consecutive frames.
        Returns (is_consistent, streak_count)
        """
        if not self.temporal_streak or self.temporal_streak[-1][0] != letter:
            self.temporal_streak = [(letter, confidence)]
        else:
            self.temporal_streak.append((letter, confidence))
        
        # Keep only recent entries
        if len(self.temporal_streak) > 10:
            self.temporal_streak = self.temporal_streak[-10:]
        
        streak_count = len(self.temporal_streak)
        
        if streak_count >= self.REQUIRED_STREAK:
            avg_conf = np.mean([c for _, c in self.temporal_streak])
            return True, streak_count, avg_conf
        
        return False, streak_count, confidence
    
    def check_gesture_stability(self, landmarks: np.ndarray) -> bool:
        """Check if hand is stable (not moving)."""
        self.landmarks_history.append(landmarks.copy())
        
        if len(self.landmarks_history) > 10:
            self.landmarks_history = self.landmarks_history[-10:]
        
        if len(self.landmarks_history) < 3:
            return False
        
        # Calculate movement variance
        movements = []
        for i in range(1, len(self.landmarks_history)):
            diff = np.abs(self.landmarks_history[i] - self.landmarks_history[i-1]).mean()
            movements.append(diff)
        
        avg_movement = np.mean(movements)
        return avg_movement < self.STABILITY_THRESHOLD
    
    def classify_asl(self, hand_image: np.ndarray) -> tuple:
        """Original classify for compatibility."""
        if self.model is None:
            return "?", 0.0, []
        
        preprocessed = self.preprocess_hand(hand_image)
        predictions = self.model.predict(preprocessed, verbose=0)[0]
        
        idx = np.argmax(predictions)
        confidence = predictions[idx]
        letter = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else "?"
        
        # Get top 3
        top3_idx = np.argsort(predictions)[-3:][::-1]
        top3 = [(CLASS_NAMES[i], float(predictions[i])) for i in top3_idx]
        
        return letter, float(confidence), top3, predictions
    
    def get_validated_prediction(self, hand_image: np.ndarray, landmarks: np.ndarray) -> tuple:
        """
        Full validation pipeline using LANDMARK-BASED classification.
        Uses finger position analysis instead of image-based CNN for accuracy.
        Returns (letter, confidence, is_accepted, status_message)
        """
        # ===============================================
        # PRIMARY: Use LANDMARK-BASED classification
        # This is much more accurate as it analyzes actual finger positions
        # ===============================================
        
        # Get handedness for proper analysis
        handedness = "Right"  # Default, could be improved with actual detection
        
        # Classify using TRAINED landmark model (96% accuracy!)
        letter, confidence = self.landmark_classifier.classify(landmarks)
        
        # If landmark classifier is uncertain, letter will be "?"
        if letter == "?" or confidence < 0.6:
            # FALLBACK: Try CNN model (but this may have bias issues)
            if self.model is not None:
                preprocessed = self.preprocess_hand(hand_image)
                predictions = self.model.predict(preprocessed, verbose=0)[0]
                cnn_letter, cnn_conf, passed, _ = self.smart_classify(predictions)
                
                # Only use CNN if it's confident AND landmark was uncertain
                if passed and cnn_conf > 0.85:
                    letter = cnn_letter
                    confidence = cnn_conf * 0.8  # Reduce confidence since CNN has bias
        
        # Stage 2: Gesture stability check
        is_stable = self.check_gesture_stability(landmarks)
        if not is_stable:
            return letter, confidence, False, "keep_still"
        
        # Stage 3: Temporal consistency (require same letter for N frames)
        is_consistent, streak, avg_conf = self.check_temporal_consistency(letter, confidence)
        
        if not is_consistent:
            return letter, confidence, False, f"streak {streak}/{self.REQUIRED_STREAK}"
        
        # All checks passed!
        return letter, avg_conf, True, "accepted"
    
    def get_smoothed_prediction(self, letter: str, confidence: float) -> tuple:
        """Legacy method for compatibility."""
        self.prediction_history.append((letter, confidence))
        if len(self.prediction_history) > 5:
            self.prediction_history.pop(0)
        
        from collections import Counter
        letters = [p[0] for p in self.prediction_history]
        most_common = Counter(letters).most_common(1)[0][0]
        avg_conf = np.mean([c for l, c in self.prediction_history if l == most_common])
        return most_common, avg_conf

    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame with multi-stage validation."""
        with self.lock:
            # Detect hands
            hands = self.hand_detector.detect(frame, draw=True)
            
            if hands:
                # 1. Gabungin data 2 tangan pakai fungsi yang baru kita bikin di ASL Classifier
                combined_features = self.landmark_classifier.process_two_hands(hands)
                
                # 2. Tebak hurufnya berdasarkan gabungan 2 tangan
                letter, confidence = self.landmark_classifier.classify(combined_features)
                
                # (Sisa logic lainnya kayak gesture stability & temporal streak tetep sama)
                is_stable = True # Ganti logic lu kalau butuh cek stabilitas 2 tangan
                is_consistent, streak, avg_conf = self.check_temporal_consistency(letter, confidence)
                
                is_accepted = is_consistent and is_stable
                status = "accepted" if is_accepted else f"streak {streak}/{self.REQUIRED_STREAK}"
                
                self.current_letter = letter
                self.current_confidence = avg_conf
                self.validation_status = status
                
                if is_accepted:
                    finalized = self.text_buffer.add_letter(letter, avg_conf)
                    if finalized and self.speech_engine:
                        threading.Thread(target=self.speech_engine.speak, args=(finalized,), daemon=True).start()
                
                self.current_word = self.text_buffer.get_current_word()
                self.sentence = self.text_buffer.get_sentence()
                self.pending_info = self.text_buffer.get_pending_info()
                
                # 3. Gambar kotak untuk SETIAP tangan yang kedetect
                for i, hand in enumerate(hands): 
                    x, y, w, h = hand.bbox
                    color = (0, 255, 0) if is_accepted else (0, 255, 255) if "streak" in status else (0, 100, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                    
                    # Tulis huruf dan status cuma di tangan pertama aja biar layar ga penuh
                    if i == 0:  # <--- Ganti jadi ngecek urutan index (0 = tangan pertama)
                        cv2.rectangle(frame, (x, y - 50), (x + 60, y), color, -1)
                        cv2.putText(frame, letter, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
                        cv2.putText(frame, f"{avg_conf:.0%}", (x + 70, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv2.putText(frame, status, (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            else:
                self.current_letter = ""
                self.current_confidence = 0.0
                self.validation_status = "no_hand"
            
            return frame
    
    def get_state(self) -> dict:
        """Get current recognition state with validation info."""
        with self.lock:
            return {
                "letter": self.current_letter,
                "confidence": self.current_confidence,
                "word": self.current_word,
                "sentence": self.sentence,
                "pending": {
                    "letter": self.pending_info[0],
                    "count": self.pending_info[1],
                    "required": self.pending_info[2]
                },
                "validation": {
                    "status": self.validation_status,
                    "streak": len(self.temporal_streak),
                    "required_streak": self.REQUIRED_STREAK
                },
                "mode": self.text_buffer.mode.value if self.text_buffer else "unknown"
            }
    
    def set_mode(self, mode: str):
        """Set translation mode and update camera strictness."""
        mode_map = {
            "instant": TranslationMode.INSTANT,
            "balanced": TranslationMode.BALANCED,
            "accurate": TranslationMode.ACCURATE,
            "strict": TranslationMode.STRICT,
            "manual": TranslationMode.MANUAL
        }
        
        if mode in mode_map:
            # 1. Update Text Buffer (Logika ngetik huruf)
            if self.text_buffer:
                self.text_buffer.set_mode(mode_map[mode])
                
            # 2. UPDATE SENSITIVITAS KAMERA (Ini yang bikin kerasa bedanya di layar!)
            if mode == "instant":
                self.REQUIRED_STREAK = 3       # Super ngebut, nahan bentar langsung ke-detect
                self.MIN_CONFIDENCE = 0.70     # Toleransi AI dinaikin
                self.STABILITY_THRESHOLD = 0.08 # Bebas gerak dikit nggak masalah
            elif mode == "balanced":
                self.REQUIRED_STREAK = 6       # Normal (Seimbang)
                self.MIN_CONFIDENCE = 0.85
                self.STABILITY_THRESHOLD = 0.03
            elif mode == "accurate":
                self.REQUIRED_STREAK = 8      # Teliti (Butuh nahan lebih lama)
                self.MIN_CONFIDENCE = 0.90
                self.STABILITY_THRESHOLD = 0.02
            elif mode == "strict":
                self.REQUIRED_STREAK = 10      # Sangat Teliti (Harus super stabil & mirip banget)
                self.MIN_CONFIDENCE = 0.95
                self.STABILITY_THRESHOLD = 0.01
                
            # Bersihin history pas ganti mode biar transisinya mulus
            self.temporal_streak = []
            self.landmarks_history = []
    
    def clear_text(self):
        """Clear all text."""
        if self.text_buffer:
            self.text_buffer.clear_all()
            self.current_word = ""
            self.sentence = ""
    
    def speak_text(self):
        """Speak current sentence."""
        if self.speech_engine and self.sentence:
            threading.Thread(target=self.speech_engine.speak, args=(self.sentence,), daemon=True).start()
    
    def cleanup(self):
        """Cleanup resources."""
        self.is_running = False
        if self.hand_detector:
            self.hand_detector.release()
        if self.speech_engine:
            self.speech_engine.shutdown()


# Global processor instance
processor = None


def get_processor():
    global processor
    if processor is None:
        processor = ASLWebProcessor()
    return processor


def generate_frames():
    """Generator for MJPEG streaming - OPTIMIZED FOR HIGH FPS."""
    proc = get_processor()
    cap = cv2.VideoCapture(0)
    
    # Optimized camera settings for higher FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Reduced from 1280 for faster processing
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Reduced from 720
    cap.set(cv2.CAP_PROP_FPS, 60)            # Request 60 FPS
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # Minimal buffer for lower latency
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Use MJPG codec
    
    while proc.is_running:
        success, frame = cap.read()
        if not success:
            break
        
        # Flip and process frame
        frame = cv2.flip(frame, 1)
        frame = proc.process_frame(frame)
        
        # Encode frame with lower quality for faster streaming
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()


@app.route('/')
def index():
    """Serve main page."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/state')
def get_state():
    """Get current recognition state."""
    proc = get_processor()
    return jsonify(proc.get_state())


@app.route('/api/mode', methods=['POST'])
def set_mode():
    """Set translation mode."""
    data = request.get_json()
    mode = data.get('mode', 'accurate')
    proc = get_processor()
    proc.set_mode(mode)
    return jsonify({"status": "ok", "mode": mode})


@app.route('/api/clear', methods=['POST'])
def clear_text():
    """Clear all text."""
    proc = get_processor()
    proc.clear_text()
    return jsonify({"status": "ok"})


@app.route('/api/speak', methods=['POST'])
def speak_text():
    """Speak current sentence."""
    proc = get_processor()
    proc.speak_text()
    return jsonify({"status": "ok"})


@app.route('/api/speak_word', methods=['POST'])
def speak_word():
    """Speak a single word (for auto-speak feature)."""
    proc = get_processor()
    data = request.get_json() or {}
    word = data.get('word', '')
    
    if word and proc.speech_engine:
        threading.Thread(
            target=proc.speech_engine.speak, 
            args=(word,), 
            daemon=True
        ).start()
    
    return jsonify({"status": "ok", "word": word})


@app.route('/api/speak_as_words', methods=['POST'])
def speak_as_words():
    """
    Speak text as WORDS, not letters.
    Combines single letters with spaces into words.
    Example: "A K U" -> "AKU", "H A L O" -> "HALO"
    """
    proc = get_processor()
    data = request.get_json() or {}
    text = data.get('text', '')
    
    if text and proc.speech_engine:
        # Combine letters: "A K U" -> "AKU"
        # Split by spaces and check if each part is single character
        parts = text.split()
        
        combined_words = []
        current_word = ""
        
        for part in parts:
            if len(part) == 1:  # Single letter
                current_word += part
            else:  # Already a word
                if current_word:
                    combined_words.append(current_word)
                    current_word = ""
                combined_words.append(part)
        
        # Don't forget last accumulated word
        if current_word:
            combined_words.append(current_word)
        
        # Join words with space
        final_text = ' '.join(combined_words)
        
        print(f"[TTS] Original: '{text}' -> Combined: '{final_text}'")
        
        threading.Thread(
            target=proc.speech_engine.speak, 
            args=(final_text,), 
            daemon=True
        ).start()
        
        return jsonify({"status": "ok", "original": text, "spoken": final_text})
    
    return jsonify({"status": "ok", "spoken": ""})


def run_server(host='0.0.0.0', port=5000, debug=False):
    """Run the Flask server."""
    print(f"\n{'='*60}")
    print("  ASL Recognition Web Server")
    print(f"{'='*60}")
    print(f"\n  Open browser: http://localhost:{port}")
    print(f"\n  Press Ctrl+C to stop")
    print(f"{'='*60}\n")
    
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ASL Recognition Web Server')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)
