# ASL Recognition Complete Module
# Semua kode untuk training, data loading, hand detection, dan inference
# Dibuat untuk konsolidasi dari folder src/

"""
ASL Recognition Complete Module
==============================
File ini berisi semua kode yang diperlukan untuk:
1. Data Loading & Preprocessing
2. Model Architecture (CNN MobileNetV2)
3. Training Pipeline
4. Hand Detection (MediaPipe)
5. Landmark-based Classification
6. Text-to-Speech Engine

Penggunaan:
-----------
1. Untuk training: Jalankan section training di notebook
2. Untuk inference: Import modul yang diperlukan
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Input
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.applications import MobileNetV2
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass
from collections import deque
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
# SECTION 3: MODEL ARCHITECTURE
# ============================================================================

def create_mobilenet_model(num_classes: int = NUM_CLASSES,
                           input_shape: tuple = (*DEFAULT_IMG_SIZE, 3),
                           trainable_base: bool = False,
                           dropout_rate: float = 0.5) -> Model:
    """Create ASL classifier using MobileNetV2 as base model."""
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = trainable_base
    
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate/2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def create_landmark_model(num_classes: int = NUM_CLASSES, input_dim: int = 63) -> Model:
    """Create landmark-based classifier model."""
    inputs = Input(shape=(input_dim,))
    x = Dense(256, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
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
    
    def __init__(self, max_num_hands: int = 2, min_detection_confidence: float = 0.7):
        import mediapipe as mp
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence, min_tracking_confidence=0.5
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
    
    def crop_hand(self, frame: np.ndarray, hand_result: HandResult,
                  target_size: Tuple[int, int] = (200, 200)) -> np.ndarray:
        x, y, w, h = hand_result.bbox
        size = max(w, h)
        cx, cy = x + w // 2, y + h // 2
        x1, y1 = max(0, cx - size // 2), max(0, cy - size // 2)
        x2, y2 = min(frame.shape[1], x1 + size), min(frame.shape[0], y1 + size)
        cropped = frame[y1:y2, x1:x2]
        if cropped.size > 0:
            return cv2.resize(cropped, target_size, interpolation=cv2.INTER_LANCZOS4)
        return np.zeros((*target_size, 3), dtype=np.uint8)
    
    def release(self):
        if hasattr(self, 'hands'):
            self.hands.close()

# ============================================================================
# SECTION 5: LANDMARK CLASSIFIER
# ============================================================================

class LandmarkASLClassifier:
    """Trained landmark-based ASL classifier."""
    
    def __init__(self, model_path: str = "saved_models/landmark_classifier.keras",
                 classes_path: str = "saved_models/landmark_classifier_classes.npy"):
        self.model = None
        self.class_names = CLASS_NAMES
        
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print(f"[LandmarkClassifier] Model loaded: {model_path}")
        if os.path.exists(classes_path):
            self.class_names = list(np.load(classes_path, allow_pickle=True))
    
    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        if landmarks is None or len(landmarks) == 0:
            return None
        normalized = landmarks.copy()
        wrist = normalized[0].copy()
        normalized = normalized - wrist
        max_dist = np.max(np.linalg.norm(normalized, axis=1))
        if max_dist > 0:
            normalized = normalized / max_dist
        return normalized
    
    def classify(self, landmarks: np.ndarray) -> Tuple[str, float]:
        if self.model is None or landmarks is None or len(landmarks) != 21:
            return "?", 0.0
        normalized = self.normalize_landmarks(landmarks)
        if normalized is None:
            return "?", 0.0
        flat = normalized.flatten().reshape(1, -1)
        predictions = self.model.predict(flat, verbose=0)[0]
        idx = np.argmax(predictions)
        return self.class_names[idx] if idx < len(self.class_names) else "?", float(predictions[idx])

# ============================================================================
# SECTION 6: TEXT-TO-SPEECH ENGINE
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
            self._is_speaking = True
            tts = gTTS(text=text, lang=self.language, slow=False)
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                temp_path = f.name
                tts.save(temp_path)
            os.system(f'afplay {temp_path} 2>/dev/null')
            try: os.unlink(temp_path)
            except: pass
            self._is_speaking = False
        except Exception:
            print(f"[TTS]: {text}")
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
                if letter != self._last_added_letter or current_time - self._last_add_time >= 1.0:
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
# SECTION 7: TRAINING FUNCTIONS
# ============================================================================

def train_cnn_model(train_dir: str, epochs: int = 30, batch_size: int = 32,
                    save_path: str = 'saved_models/asl_model_best.keras'):
    """Train CNN model on ASL dataset."""
    print("=" * 60)
    print("TRAINING ASL CNN MODEL")
    print("=" * 60)
    
    train_gen, val_gen, class_indices = create_data_generators(train_dir, batch_size=batch_size)
    model = create_mobilenet_model()
    callbacks = get_callbacks(save_path)
    
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)
    print(f"\nModel saved to: {save_path}")
    return model, history


def extract_landmarks_from_dataset(data_dir: str, save_path: str = 'saved_models/landmarks_train.npz'):
    """Extract hand landmarks from dataset images."""
    print("Extracting landmarks from dataset...")
    detector = HandDetector(max_num_hands=1)
    
    all_landmarks = []
    all_labels = []
    
    available_classes = [c for c in CLASS_NAMES if os.path.exists(os.path.join(data_dir, c))]
    
    for class_idx, class_name in enumerate(tqdm(available_classes)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        for img_name in os.listdir(class_dir)[:100]:  # Limit per class
            img_path = os.path.join(class_dir, img_name)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                hands = detector.detect(img, draw=False)
                if hands:
                    landmarks = hands[0].landmarks
                    # Normalize
                    wrist = landmarks[0].copy()
                    normalized = landmarks - wrist
                    max_dist = np.max(np.linalg.norm(normalized, axis=1))
                    if max_dist > 0:
                        normalized = normalized / max_dist
                    all_landmarks.append(normalized.flatten())
                    all_labels.append(class_idx)
            except Exception:
                continue
    
    detector.release()
    
    X = np.array(all_landmarks)
    y = np.array(all_labels)
    np.savez(save_path, X=X, y=y)
    print(f"Saved {len(X)} landmark samples to {save_path}")
    return X, y


def train_landmark_model(X: np.ndarray, y: np.ndarray, epochs: int = 50,
                         save_path: str = 'saved_models/landmark_classifier.keras'):
    """Train landmark-based classifier."""
    print("=" * 60)
    print("TRAINING LANDMARK CLASSIFIER")
    print("=" * 60)
    
    from tensorflow.keras.utils import to_categorical
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_val_cat = to_categorical(y_val, NUM_CLASSES)
    
    model = create_landmark_model()
    callbacks = get_callbacks(save_path)
    
    history = model.fit(X_train, y_train_cat, validation_data=(X_val, y_val_cat),
                        epochs=epochs, batch_size=32, callbacks=callbacks)
    
    # Save class names
    np.save(save_path.replace('.keras', '_classes.npy'), CLASS_NAMES)
    print(f"Model saved to: {save_path}")
    return model, history

# ============================================================================
# SECTION 8: EVALUATION FUNCTIONS
# ============================================================================

def evaluate_model(model, X_test, y_test):
    """Evaluate model and print metrics."""
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)
    
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES[:len(np.unique(y_test))]))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES[:len(np.unique(y_test))],
                yticklabels=CLASS_NAMES[:len(np.unique(y_test))])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    return accuracy


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


if __name__ == "__main__":
    print("ASL Recognition Complete Module")
    print("=" * 60)
    print("Available functions:")
    print("  - load_asl_dataset(data_dir)")
    print("  - create_data_generators(train_dir)")
    print("  - create_mobilenet_model()")
    print("  - create_landmark_model()")
    print("  - train_cnn_model(train_dir)")
    print("  - train_landmark_model(X, y)")
    print("  - HandDetector()")
    print("  - LandmarkASLClassifier()")
    print("  - SpeechEngine()")
    print("  - TextBuffer()")
