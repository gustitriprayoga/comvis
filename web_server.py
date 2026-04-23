"""
ASL Recognition Web Server (DUAL-ENGINE VERSION)
MobileNetV2 for Real-time Tracking, EfficientNetB0 for Accuracy Validation.
"""

import os
import cv2
import time
import numpy as np
from flask import Flask, Response, render_template, jsonify, request
from flask_cors import CORS
from flask import send_from_directory
import threading

from asl_modules import (
    HandDetector, SpeechEngine, TextBuffer, TranslationMode,
    DualEngineASLClassifier, CLASS_NAMES, DEFAULT_IMG_SIZE
)

app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
CORS(app)

class ASLWebProcessor:
    MIN_CONFIDENCE = 0.85      
    MIN_GAP = 0.15             
    REQUIRED_STREAK = 6        
    STABILITY_THRESHOLD = 0.03 
    
    def __init__(self):
        self.hand_detector = None
        self.text_buffer = None
        self.speech_engine = None
        self.dual_classifier = None
        
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
        print("[Server] Initializing Strict Dual-Engine ASL processor...")
        self.hand_detector = HandDetector(max_num_hands=2, min_detection_confidence=0.7)
        self.text_buffer = TextBuffer(mode=TranslationMode.BALANCED)
        self.speech_engine = SpeechEngine()
        self.dual_classifier = DualEngineASLClassifier()
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
            hands = self.hand_detector.detect(frame, draw=True)
            
            if hands:
                combined_features = self.dual_classifier.process_two_hands(hands)
                
                # ============================================================
                # 1. MOBILE-NET V2 SELALU JALAN DI DEPAN (REAL-TIME TRACKER)
                # ============================================================
                letter_fast, conf_fast = self.dual_classifier.predict_fast(combined_features)
                
                # Bangun streak pelacakan pakai MobileNetV2
                is_consistent, streak, avg_conf = self.check_temporal_consistency(letter_fast, conf_fast)
                
                # ============================================================
                # 2. JIKA TANGAN STABIL -> EFFICIENT-NET B0 MENGAMBIL ALIH
                # ============================================================
                if is_consistent and streak >= self.REQUIRED_STREAK and avg_conf >= self.MIN_CONFIDENCE:
                    letter_acc, conf_acc = self.dual_classifier.predict_accurate(combined_features)
                    
                    display_letter = letter_acc
                    display_conf = conf_acc
                    engine_used = "EfficientNet-B0 (Verified)"
                    status = "accepted"
                    color = (0, 255, 0) # Warna Hijau (OK)
                    
                    # CUMA EFFICIENT-NET B0 YANG BOLEH NGETIK KE BUFFER!
                    self.text_buffer.add_letter(letter_acc, conf_acc)
                    
                else:
                    # Tampilan saat tangan masih bergerak (Real-time Tracker)
                    display_letter = letter_fast
                    display_conf = avg_conf
                    engine_used = "MobileNetV2 (Tracking...)"
                    status = f"streak {streak}/{self.REQUIRED_STREAK}"
                    color = (0, 255, 255) # Warna Kuning (Loading)
                    
                    # Mencegah tulisan bocor saat tangan masih gerak
                    self.text_buffer._pending_count = 0
                
                # Update status yang bakal ditarik ke Web HTML
                self.current_letter = display_letter
                self.current_confidence = display_conf
                self.validation_status = status
                self.current_word = self.text_buffer.get_current_word()
                self.sentence = self.text_buffer.get_sentence()
                self.pending_info = self.text_buffer.get_pending_info()
                
                # Menggambar kotak dan teks di atas video
                for i, hand in enumerate(hands): 
                    x, y, w, h = hand.bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                    
                    if i == 0: 
                        cv2.rectangle(frame, (x, y - 50), (x + 60, y), color, -1)
                        cv2.putText(frame, display_letter, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
                        cv2.putText(frame, f"{display_conf:.0%}", (x + 70, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv2.putText(frame, status, (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                        cv2.putText(frame, engine_used, (x, y + h + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            else:
                self.current_letter = ""
                self.current_confidence = 0.0
                self.validation_status = "no_hand"
                self.temporal_streak = []
                self.text_buffer._pending_count = 0 # Reset biar gak ngetik sendiri
            
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
                "mode": self.text_buffer.mode.value if self.text_buffer else "unknown"
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
    
    def clear_text(self):
        if self.text_buffer:
            self.text_buffer.clear_all()
            self.current_word = ""
            self.sentence = ""
    
    def cleanup(self):
        self.is_running = False
        if self.hand_detector: self.hand_detector.release()
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

@app.route('/api/speak_as_words', methods=['POST'])
def speak_as_words():
    proc = get_processor()
    text = (request.get_json() or {}).get('text', '')
    if text and proc.speech_engine:
        parts = text.split()
        combined_words = []
        current_word = ""
        for part in parts:
            if len(part) == 1: current_word += part
            else:
                if current_word:
                    combined_words.append(current_word)
                    current_word = ""
                combined_words.append(part)
        if current_word: combined_words.append(current_word)
        final_text = ' '.join(combined_words)
        threading.Thread(target=proc.speech_engine.speak, args=(final_text,), daemon=True).start()
        return jsonify({"status": "ok", "spoken": final_text})
    return jsonify({"status": "ok", "spoken": ""})

@app.route('/get_generated_image/<filename>')
def get_generated_image(filename):
    """Mengambil gambar dari folder saved_generate."""
    return send_from_directory('saved_generate', filename)

if __name__ == '__main__':
    print(f"\n{'='*60}\n  ASL Web Server (STRICT DUAL-ENGINE)\n{'='*60}\n  Buka browser: http://localhost:5000\n{'='*60}\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
