# ASL Recognition Complete Module
# Versi Final: DUAL-ENGINE (MobileNetV2 + EfficientNetB0) & 126 Landmark 2 Tangan

import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from queue import Queue
import threading
import time

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x): return x

# ============================================================================
# SECTION 1: CONSTANTS & CONFIGURATION
# ============================================================================

CLASS_NAMES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]
NUM_CLASSES = len(CLASS_NAMES)
DEFAULT_IMG_SIZE = (200, 200)

# ============================================================================
# SECTION 2: DATA LOADING MODULE
# ============================================================================

def load_asl_dataset(data_dir: str, img_size: tuple = DEFAULT_IMG_SIZE, 
                     normalize: bool = True, verbose: bool = True) -> tuple:
    """Load ASL alphabet dataset from directory."""
    images = []
    labels = []
    available_classes = [c for c in CLASS_NAMES if os.path.exists(os.path.join(data_dir, c))]
    
    if verbose:
        print(f"Loading dataset from: {data_dir}")
        print(f"Found {len(available_classes)} classes")
    
    iterator = tqdm(available_classes) if verbose and HAS_TQDM else available_classes
    
    for class_idx, class_name in enumerate(iterator):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(img_size, Image.Resampling.LANCZOS)
                images.append(np.array(img))
                labels.append(class_idx)
            except Exception as e:
                continue
    
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    if normalize:
        images = images / 255.0
    if verbose:
        print(f"Loaded {len(images)} images")
    return images, labels, available_classes

def create_data_generators(train_dir: str, img_size: tuple = DEFAULT_IMG_SIZE,
                           batch_size: int = 32, validation_split: float = 0.2,
                           augment: bool = True) -> tuple:
    """Create data generators with augmentation for training."""
    if augment:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            validation_split=validation_split
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1./255, validation_split=validation_split)
    
    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size,
        class_mode='categorical', subset='training', shuffle=True, seed=42
    )
    val_gen = train_datagen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size,
        class_mode='categorical', subset='validation', shuffle=False, seed=42
    )
    return train_gen, val_gen, train_gen.class_indices

def get_class_weights(labels: np.ndarray, num_classes: int = NUM_CLASSES) -> dict:
    """Calculate class weights for handling imbalanced dataset."""
    unique_classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=unique_classes, y=labels)
    class_weights = {cls: weights[idx] for idx, cls in enumerate(unique_classes)}
    for i in range(num_classes):
        if i not in class_weights:
            class_weights[i] = 1.0
    return class_weights

# ============================================================================
# SECTION 3: MODEL ARCHITECTURE (DUAL-ENGINE)
# ============================================================================

def create_mobilenet_engine(num_classes: int = NUM_CLASSES, input_dim: int = 126) -> Model:
    """Engine 1: Ringan & Super Cepat untuk Real-time (MobileNetV2-style)"""
    inputs = Input(shape=(input_dim,))
    x = Dense(256, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x) 
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name="Engine_Fast_MobileNet")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_efficientnet_engine(num_classes: int = NUM_CLASSES, input_dim: int = 126) -> Model:
    """Engine 2: Lebih Dalam & Teliti untuk Verifikasi (EfficientNetB0-style)"""
    inputs = Input(shape=(input_dim,))
    x = Dense(512, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name="Engine_Accurate_EfficientNet")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def get_callbacks(model_save_path: str = 'saved_models/asl_model_best.keras',
                  patience: int = 10) -> list:
    """Create training callbacks."""
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    return [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
    ]

# ============================================================================
# SECTION 4: HAND DETECTION (MediaPipe)
# ============================================================================

@dataclass
class HandResult:
    """Container for hand detection results."""
    landmarks: np.ndarray
    bbox: Tuple[int, int, int, int]
    handedness: str
    confidence: float

class HandDetector:
    """Hand detection using MediaPipe."""
    
    def __init__(self, max_num_hands: int = 2, min_detection_confidence: float = 0.7, static_mode: bool = False):
        import mediapipe as mp
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence, 
            min_tracking_confidence=0.5
        )
        self._results = None
    
    def detect(self, frame: np.ndarray, draw: bool = False) -> List[HandResult]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._results = self.hands.process(rgb_frame)
        detected_hands = []
        h, w = frame.shape[:2]
        
        if self._results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(self._results.multi_hand_landmarks):
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
                x_coords = (landmarks[:, 0] * w).astype(int)
                y_coords = (landmarks[:, 1] * h).astype(int)
                padding = 40
                x, y = max(0, x_coords.min() - padding), max(0, y_coords.min() - padding)
                x2, y2 = min(w, x_coords.max() + padding), min(h, y_coords.max() + padding)
                
                handedness = 'Right'
                confidence = 0.0
                if self._results.multi_handedness:
                    handedness = self._results.multi_handedness[idx].classification[0].label
                    confidence = self._results.multi_handedness[idx].classification[0].score
                
                detected_hands.append(HandResult(landmarks, (x, y, x2-x, y2-y), handedness, confidence))
                
                if draw:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())
        return detected_hands
    
    def release(self):
        if hasattr(self, 'hands'):
            self.hands.close()

# ============================================================================
# SECTION 5: LANDMARK CLASSIFIER (DUAL-ENGINE)
# ============================================================================

# ============================================================================
# SECTION 5: LANDMARK CLASSIFIER (DUAL-ENGINE)
# ============================================================================

class DualEngineASLClassifier:
    """Trained dual-engine landmark-based ASL classifier."""
    
    def __init__(self, 
                 model_fast_path: str = "saved_models/mobilenet_landmark.keras",
                 model_acc_path: str = "saved_models/efficientnet_landmark.keras",
                 classes_path: str = "saved_models/landmark_classifier_classes.npy"):
        
        self.model_fast = None
        self.model_acc = None
        self.class_names = CLASS_NAMES
        
        if os.path.exists(model_fast_path) and os.path.exists(model_acc_path):
            self.model_fast = tf.keras.models.load_model(model_fast_path)
            self.model_acc = tf.keras.models.load_model(model_acc_path)
            print("[DualEngine] 🔥 MobileNetV2 & EfficientNetB0 loaded & ready!")
            
        if os.path.exists(classes_path):
            self.class_names = list(np.load(classes_path, allow_pickle=True))
    
    def process_two_hands(self, detected_hands) -> np.ndarray:
        if not detected_hands: return np.zeros(126)
        hands = sorted(detected_hands, key=lambda h: h.landmarks[0][0])
        combined = np.zeros(126)
        base_wrist = hands[0].landmarks[0].copy()

        all_norms = []
        for hand in hands[:2]:
            all_norms.append(hand.landmarks - base_wrist)

        max_dist = max([np.max(np.linalg.norm(n, axis=1)) for n in all_norms])
        for i, norm in enumerate(all_norms):
            if max_dist > 0: norm = norm / max_dist
            combined[i*63:(i+1)*63] = norm.flatten()
        return combined

    def predict_fast(self, combined_features: np.ndarray) -> tuple:
        """Hanya menggunakan MobileNetV2 (Sangat Cepat buat Tracking)"""
        if self.model_fast is None: return "?", 0.0
        flat = combined_features.reshape(1, -1)
        pred = self.model_fast.predict(flat, verbose=0)[0]
        idx = np.argmax(pred)
        return self.class_names[idx], float(pred[idx])

    def predict_accurate(self, combined_features: np.ndarray) -> tuple:
        """Hanya menggunakan EfficientNet-B0 (Sangat Akurat buat Verifikasi)"""
        if self.model_acc is None: return "?", 0.0
        flat = combined_features.reshape(1, -1)
        pred = self.model_acc.predict(flat, verbose=0)[0]
        idx = np.argmax(pred)
        return self.class_names[idx], float(pred[idx])

    def classify(self, combined_features: np.ndarray) -> tuple:
        # Legacy support
        letter, conf = self.predict_fast(combined_features)
        if conf < 0.90:
            return self.predict_accurate(combined_features) + ("EfficientNet-B0",)
        return letter, conf, "MobileNetV2"

# ============================================================================
# SECTION 6: TEXT-TO-SPEECH ENGINE & BUFFER
# ============================================================================

class TranslationMode(Enum):
    INSTANT = "instant"
    BALANCED = "balanced"
    ACCURATE = "accurate"
    STRICT = "strict"
    MANUAL = "manual"

class SpeechEngine:
    """Text-to-Speech Engine dengan Bahasa Indonesia."""
    def __init__(self, language: str = 'id'):
        self.language = language
        self._speech_queue = Queue()
        self._is_speaking = False
        self._worker_thread = None
        self._stop_flag = False
    
    def speak(self, text: str, async_mode: bool = True):
        if not text or not text.strip():
            return
        if async_mode:
            self._speech_queue.put(text)
            if self._worker_thread is None or not self._worker_thread.is_alive():
                self._stop_flag = False
                self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
                self._worker_thread.start()
        else:
            self._speak_sync(text)
    
    def _speak_sync(self, text: str):
        try:
            from gtts import gTTS
            import tempfile
            import platform
            self._is_speaking = True
            
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            temp_path = temp_file.name
            temp_file.close() 
            
            tts = gTTS(text=text, lang=self.language, slow=False)
            tts.save(temp_path)
            
            if platform.system() == 'Windows':
                try:
                    import pygame
                    pygame.mixer.init()
                    pygame.mixer.music.load(temp_path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                    pygame.mixer.music.stop()
                    pygame.mixer.quit()
                except ImportError:
                    print("[Server] Pygame belum keinstall bro. Ketik: pip install pygame")
            elif platform.system() == 'Darwin':
                os.system(f'afplay "{temp_path}" 2>/dev/null')
            else:
                os.system(f'mpg123 "{temp_path}" 2>/dev/null')
                
            try: os.remove(temp_path)
            except: pass
            self._is_speaking = False
            
        except Exception as e:
            print(f"[TTS Error]: {e}")
            self._is_speaking = False
    
    def _worker_loop(self):
        while not self._stop_flag:
            try:
                text = self._speech_queue.get(timeout=0.5)
                self._speak_sync(text)
                self._speech_queue.task_done()
            except: continue
    
    def shutdown(self):
        self._stop_flag = True


class TextBuffer:
    """Text buffer untuk mengumpulkan prediksi huruf."""
    MODE_SETTINGS = {
        TranslationMode.INSTANT: {'repeats': 2, 'hold_time': 0.1, 'conf': 0.50},
        TranslationMode.BALANCED: {'repeats': 3, 'hold_time': 0.2, 'conf': 0.60},
        TranslationMode.ACCURATE: {'repeats': 4, 'hold_time': 0.3, 'conf': 0.70},
        TranslationMode.STRICT: {'repeats': 5, 'hold_time': 0.5, 'conf': 0.80},
        TranslationMode.MANUAL: {'repeats': 1, 'hold_time': 0.0, 'conf': 0.40},
    }
    
    def __init__(self, mode: TranslationMode = TranslationMode.BALANCED):
        self.mode = mode
        self._current_word = ""
        self._sentence = ""
        self._pending_letter = ""
        self._pending_count = 0
        self._pending_start_time = 0.0
        self._pending_confidence_sum = 0.0
        self._last_added_letter = ""
        self._last_add_time = 0.0
    
    def add_letter(self, letter: str, confidence: float = 1.0) -> Optional[str]:
        current_time = time.time()
        settings = self.MODE_SETTINGS.get(self.mode, self.MODE_SETTINGS[TranslationMode.BALANCED])
        
        if letter in ['space', 'del', 'nothing']:
            if letter == 'space' and self._current_word:
                word = self._current_word
                self._sentence += word + " "
                self._current_word = ""
                return word
            elif letter == 'del' and self._current_word:
                self._current_word = self._current_word[:-1]
            return None
        
        if confidence < settings['conf']:
            return None
        
        if letter == self._pending_letter:
            self._pending_count += 1
            self._pending_confidence_sum += confidence
            time_held = current_time - self._pending_start_time
            
            if self._pending_count >= settings['repeats'] and time_held >= settings['hold_time']:
                if letter != self._last_added_letter or current_time - self._last_add_time >= 10.0:
                    self._current_word += letter
                    self._last_added_letter = letter
                    self._last_add_time = current_time
                self._pending_letter = ""
                self._pending_count = 0
        else:
            self._pending_letter = letter
            self._pending_count = 1
            self._pending_start_time = current_time
            self._pending_confidence_sum = confidence
        return None
    
    def get_current_word(self) -> str:
        return self._current_word
    
    def get_sentence(self) -> str:
        return self._sentence + self._current_word
    
    def get_pending_info(self) -> Tuple[str, int, int]:
        settings = self.MODE_SETTINGS.get(self.mode)
        return self._pending_letter, self._pending_count, settings['repeats'] if settings else 3
    
    def set_mode(self, mode: TranslationMode):
        self.mode = mode
    
    def clear_all(self):
        self._current_word = ""
        self._sentence = ""
        self._pending_letter = ""
        self._pending_count = 0

# ============================================================================
# SECTION 7: DATASET EXTRACTION
# ============================================================================

def extract_landmarks_from_dataset(data_dir: str, save_path: str = 'saved_models/landmarks_train.npz'):
    """Extract hand landmarks from dataset images."""
    print("Extracting landmarks from dataset...")
    detector = HandDetector(max_num_hands=2, static_mode=True)
    all_landmarks = []
    all_labels = []
    available_classes = [c for c in CLASS_NAMES if os.path.exists(os.path.join(data_dir, c))]
    
    for class_idx, class_name in enumerate(tqdm(available_classes)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        for img_name in os.listdir(class_dir): 
            img_path = os.path.join(class_dir, img_name)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                hands = detector.detect(img, draw=False)
                if hands:
                    hands = sorted(hands, key=lambda h: h.landmarks[0][0])
                    combined = np.zeros(126)
                    base_wrist = hands[0].landmarks[0].copy()
                    
                    all_norms = []
                    for hand in hands[:2]:
                        all_norms.append(hand.landmarks - base_wrist)
                        
                    max_dist = max([np.max(np.linalg.norm(n, axis=1)) for n in all_norms])
                    for i, norm in enumerate(all_norms):
                        if max_dist > 0: norm = norm / max_dist
                        combined[i*63:(i+1)*63] = norm.flatten()
                        
                    all_landmarks.append(combined)
                    all_labels.append(class_idx)
            except Exception:
                continue
    
    detector.release()
    X = np.array(all_landmarks)
    y = np.array(all_labels)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, X=X, y=y)
    print(f"Saved {len(X)} landmark samples to {save_path}")
    return X, y

# ============================================================================
# SECTION 8: EVALUATION FUNCTIONS
# ============================================================================

def evaluate_model(model, X_test, y_test):
    """Evaluate model and save Classification Report + Confusion Matrix as PNG into 'saved_generate'."""
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd 
    import os

    # 1. BIKIN FOLDER OUTPUT BARU
    OUTPUT_DIR = 'saved_generate'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Prediksi data
    predictions = model.predict(X_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    
    # Ambil nama kelas yang aktif
    unique_classes = np.unique(y_test)
    target_names = [CLASS_NAMES[i] for i in unique_classes]
    
    # ============================================================
    # GENERATE CLASSIFICATION REPORT (TABLE PNG)
    # ============================================================
    report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).iloc[:-1, :].T # Buang baris support total biar rapi
    
    plt.figure(figsize=(10, 12))
    sns.heatmap(report_df, annot=True, cmap="YlGnBu", cbar=False, fmt=".2f")
    plt.title(f'Classification Report - {model.name}')
    
    # Simpan ke folder saved_generate
    report_file = os.path.join(OUTPUT_DIR, f'report_{model.name.lower()}.png')
    plt.savefig(report_file, dpi=300, bbox_inches='tight')
    print(f"Laporan Klasifikasi disimpan: {report_file}")
    plt.close()

    # ============================================================
    # GENERATE CONFUSION MATRIX (HEATMAP PNG)
    # ============================================================
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix - {model.name}')
    plt.ylabel('Label Asli')
    plt.xlabel('Tebakan AI')
    plt.tight_layout()
    
    # Simpan ke folder saved_generate
    matrix_file = os.path.join(OUTPUT_DIR, f'confusion_matrix_{model.name.lower()}.png')
    plt.savefig(matrix_file, dpi=300)
    print(f"Confusion Matrix disimpan: {matrix_file}")
    plt.show()

def plot_training_history(history):
    """Plot training history."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()