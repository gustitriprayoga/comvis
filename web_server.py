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
    CLASS_NAMES, DEFAULT_IMG_SIZE, crop_hand,
    HandDetector
)
try:
    from tensorflow.keras.layers import GlobalAveragePooling2D
except (ImportError, ModuleNotFoundError):
    from keras.layers import GlobalAveragePooling2D

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
            self.model_mobilenet = self._load_model_safe(model_mobilenet_path)
            if self.model_mobilenet:
                print("[Classifier] MobileNetV3 loaded & ready!")

        if os.path.exists(model_densenet_path):
            self.model_densenet = self._load_model_safe(model_densenet_path)
            if self.model_densenet:
                print("[Classifier] DenseNet121 loaded & ready!")

        if os.path.exists(classes_path):
            self.class_names = list(np.load(classes_path, allow_pickle=True))
    
    @staticmethod
    def _load_model_safe(path):
        """Load model dengan kompatibilitas Keras 2 ↔ Keras 3."""
        try:
            # Coba load biasa
            return tf.keras.models.load_model(path, compile=False)
        except TypeError as e:
            if 'renorm' in str(e) or 'Unrecognized keyword' in str(e):
                # Model Keras 2 punya args BatchNormalization yang tidak dikenali Keras 3
                print(f"[Classifier] Keras 2→3 compat fix untuk {path}")
                try:
                    import keras
                    # Buat BatchNormalization wrapper yang abaikan args lama
                    _OrigBN = keras.layers.BatchNormalization
                    class CompatBatchNormalization(_OrigBN):
                        def __init__(self, **kwargs):
                            # Buang args Keras 2 yang tidak dikenal Keras 3
                            for k in ['renorm', 'renorm_clipping', 'renorm_momentum']:
                                kwargs.pop(k, None)
                            super().__init__(**kwargs)
                    
                    return tf.keras.models.load_model(
                        path, compile=False,
                        custom_objects={'BatchNormalization': CompatBatchNormalization}
                    )
                except Exception as e2:
                    print(f"[Classifier] ERROR loading {path}: {e2}")
                    return None
            else:
                print(f"[Classifier] ERROR loading {path}: {e}")
                return None
        except Exception as e:
            print(f"[Classifier] ERROR loading {path}: {e}")
            return None

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
        self.hand_detector = None
        
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
        
        # Frame skipping: prediksi hanya setiap N frame (mengurangi lag)
        self.frame_count = 0
        self.PREDICT_EVERY_N = 3  # Prediksi setiap 3 frame
        self.last_crop_preview = None  # Preview crop untuk debugging
        
        self._initialize()
    
    def _initialize(self):
        print("[Server] Initializing BISINDO processor (2 tangan)...")
        self.text_buffer = TextBuffer(mode=TranslationMode.BALANCED)
        self.speech_engine = SpeechEngine()
        self.classifier = ImageBasedASLClassifier()
        # BISINDO menggunakan 2 tangan!
        self.hand_detector = HandDetector(max_num_hands=2, min_detection_confidence=0.5, static_mode=False)
        print("[Server] HandDetector (MediaPipe, 2 tangan) loaded & ready!")
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
            # 1. Deteksi tangan menggunakan MediaPipe (max 2 tangan untuk BISINDO)
            hands = self.hand_detector.detect(frame, draw=True)

            if not hands:
                # Tidak ada tangan terdeteksi
                self.current_letter = ""
                self.current_confidence = 0.0
                self.validation_status = "waiting"
                self.temporal_streak = []
                
                cv2.putText(frame, "Tunjukkan KEDUA tangan ke kamera", (30, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
                cv2.putText(frame, "(BISINDO = 2 tangan)", (100, 270),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)
                return frame

            # 2. Crop area kedua tangan menggunakan LANDMARK positions
            #    Training data BISINDO = close-up tangan saja, jadi crop harus ketat
            fh, fw = frame.shape[:2]
            
            # Kumpulkan SEMUA titik landmark dari semua tangan (posisi piksel)
            all_px, all_py = [], []
            for hand in hands:
                for lm in hand.landmarks:
                    all_px.append(int(lm[0] * fw))
                    all_py.append(int(lm[1] * fh))
            
            if not all_px:
                return frame
            
            # Hitung bounding box dari landmark (bukan dari MediaPipe bbox)
            # Ini lebih ketat karena hanya berdasarkan posisi jari/tangan
            lm_x1, lm_y1 = min(all_px), min(all_py)
            lm_x2, lm_y2 = max(all_px), max(all_py)
            
            # Padding proporsional (15% dari ukuran area tangan)
            pad_w = int((lm_x2 - lm_x1) * 0.15)
            pad_h = int((lm_y2 - lm_y1) * 0.15)
            padding = max(pad_w, pad_h, 15)  # minimal 15px
            
            x1 = max(0, lm_x1 - padding)
            y1 = max(0, lm_y1 - padding)
            x2 = min(fw, lm_x2 + padding)
            y2 = min(fh, lm_y2 + padding)
            
            # Buat crop menjadi SQUARE (seperti training data 640x640)
            crop_w = x2 - x1
            crop_h = y2 - y1
            crop_size = max(crop_w, crop_h)
            
            # Center the square crop
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            half = crop_size // 2
            
            sq_x1 = max(0, cx - half)
            sq_y1 = max(0, cy - half)
            sq_x2 = min(fw, cx + half)
            sq_y2 = min(fh, cy + half)
            
            if (sq_x2 - sq_x1) < 30 or (sq_y2 - sq_y1) < 30:
                return frame
            
            # Crop area tangan saja (ketat, square)
            hand_crop = frame[sq_y1:sq_y2, sq_x1:sq_x2]

            # 3. Resize & convert BGR→RGB (ImageDataGenerator training pakai PIL = RGB)
            # Menggunakan FULL FRAME asli (tidak di-crop) sesuai permintaan
            img_resized = cv2.resize(frame, self.img_size)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  # PENTING: BGR→RGB!
            img = np.expand_dims(img_rgb / 255.0, axis=0).astype(np.float32)
            
            # Simpan preview crop untuk ditampilkan di frame
            self.last_crop_preview = img_resized.copy()

            # 4. Predict (frame skipping untuk mengurangi lag)
            self.frame_count += 1
            if self.frame_count % self.PREDICT_EVERY_N == 0 or not self.current_letter:
                letter, confidence = self.classifier.predict(img)
            else:
                letter = self.current_letter if self.current_letter else "?"
                confidence = self.current_confidence

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
            
            # Draw crop area yang dikirim ke model (square, ketat di tangan)
            cv2.rectangle(frame, (sq_x1, sq_y1), (sq_x2, sq_y2), color, 3)
            
            # Info tangan terdeteksi
            hand_count = len(hands)
            hand_label = f"{hand_count} tangan" if hand_count > 1 else "1 tangan"
            hand_color = (0, 255, 0) if hand_count >= 2 else (0, 165, 255)  # Hijau jika 2, oranye jika 1
            
            # Overlay prediksi
            cv2.rectangle(frame, (10, 10), (200, 120), color, -1)
            cv2.putText(frame, display_letter, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5)
            cv2.putText(frame, f"{display_conf:.0%}", (210, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, engine_used, (210, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"{hand_label}", (210, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)
            cv2.putText(frame, status, (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            
            # Preview crop di pojok kanan bawah (agar user bisa lihat apa yang model terima)
            if self.last_crop_preview is not None:
                preview = cv2.resize(self.last_crop_preview, (120, 120))
                ph, pw = preview.shape[:2]
                frame[fh-ph-10:fh-10, fw-pw-10:fw-10] = preview
                cv2.rectangle(frame, (fw-pw-12, fh-ph-12), (fw-8, fh-8), (255, 255, 255), 1)
                cv2.putText(frame, "Model Input", (fw-pw-10, fh-ph-18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
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
        if self.hand_detector: self.hand_detector.release()

processor = None
def get_processor():
    global processor
    if processor is None: processor = ASLWebProcessor()
    return processor

def generate_frames():
    proc = get_processor()
    
    # Coba berbagai kombinasi camera index dan backend
    cap = None
    for backend_name, backend_id in [("DSHOW", cv2.CAP_DSHOW), ("default", cv2.CAP_ANY)]:
        for cam_idx in [0, 1, 2]:
            test_cap = cv2.VideoCapture(cam_idx, backend_id)
            if test_cap.isOpened():
                ret, frame = test_cap.read()
                if ret:
                    cap = test_cap
                    print(f"[Server] Kamera ditemukan: index={cam_idx}, backend={backend_name}")
                    break
                else:
                    test_cap.release()
        if cap is not None:
            break
    
    if cap is None:
        print("[Server] ERROR: Tidak ada kamera yang tersedia!")
        # Kirim frame hitam dengan pesan error
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "Kamera tidak ditemukan!", (100, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', blank)
        if ret:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    while proc.is_running:
        success, frame = cap.read()
        if not success: continue  # Skip frame jika gagal, jangan break
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
    # Coba ambil teks dari request body dulu, fallback ke proc.sentence
    data = request.get_json(silent=True)
    text = None
    if data and data.get('text'):
        text = data['text']
    elif proc.sentence:
        text = proc.sentence
    
    if proc.speech_engine and text:
        threading.Thread(target=proc.speech_engine.speak, args=(text,), daemon=True).start()
    return jsonify({"status": "ok"})

@app.route('/get_generated_image/<filename>')
def get_generated_image(filename):
    """Mengambil gambar dari folder saved_generate."""
    return send_from_directory('saved_generate', filename)

if __name__ == '__main__':
    print(f"\n{'='*60}\n  ASL Web Server (Image-Based CNN)\n{'='*60}\n  Buka browser: http://localhost:5000\n{'='*60}\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
