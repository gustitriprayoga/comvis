# ASL Recognition Complete Module
# Versi Final: DUAL-ENGINE (MobileNetV3 vs DenseNet121) & Image-Based CNN

import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
# Keras compatibility shim (TF < 2.16 vs TF >= 2.16 / Keras 3)
try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, GlobalAveragePooling2D
    from tensorflow.keras.applications import MobileNetV3Small, DenseNet121
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
except (ImportError, ModuleNotFoundError):
    # Keras 3 standalone (TF >= 2.16)
    try:
        from keras.src.legacy.preprocessing.image import ImageDataGenerator
    except (ImportError, ModuleNotFoundError):
        from keras.preprocessing.image import ImageDataGenerator
    from keras import Model
    from keras.layers import Dense, Dropout, BatchNormalization, Input, GlobalAveragePooling2D
    from keras.applications import MobileNetV3Small, DenseNet121
    from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
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
DEFAULT_IMG_SIZE = (224, 224)

# ============================================================================
# SECTION 2: DATA LOADING MODULE
# ============================================================================


def create_data_generators(train_dir: str, img_size: tuple = DEFAULT_IMG_SIZE,
                           batch_size: int = 32, validation_split: float = 0.2,
                           test_split: float = 0.1, augment: bool = True) -> tuple:
    """Create data generators with augmentation for training."""
    
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    # Get all file paths and their labels
    filepaths = []
    labels = []
    
    # Discover classes from directory names and sort them
    discovered_classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    if not discovered_classes:
        raise ValueError(f"No class directories found in {train_dir}")

    for cls in discovered_classes:
        cls_path = os.path.join(train_dir, cls)
        for file in os.listdir(cls_path):
            filepaths.append(os.path.join(cls_path, file))
            labels.append(cls)

    # Create a DataFrame
    df = pd.DataFrame({'filepath': filepaths, 'label': labels})

    # Split into train/validation and test sets
    train_val_df, test_df = train_test_split(df, test_size=test_split, random_state=42, stratify=df['label'])

    # Split train/validation into train and validation sets
    train_df, val_df = train_test_split(train_val_df, test_size=validation_split, random_state=42, stratify=train_val_df['label'])

    if augment:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2],
            horizontal_flip=True,
            fill_mode='nearest'
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_dataframe(
        train_df,
        x_col='filepath',
        y_col='label',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=discovered_classes,
        shuffle=True
    )

    val_gen = test_datagen.flow_from_dataframe(
        val_df,
        x_col='filepath',
        y_col='label',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=discovered_classes,
        shuffle=False
    )
    
    test_gen = test_datagen.flow_from_dataframe(
        test_df,
        x_col='filepath',
        y_col='label',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=discovered_classes,
        shuffle=False
    )

    return train_gen, val_gen, test_gen, train_gen.class_indices
    


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
# SECTION 3: MODEL ARCHITECTURE (MobileNetV3 vs DenseNet121)
# ============================================================================
def create_mobilenetv3_model(num_classes: int = NUM_CLASSES, img_size: tuple = DEFAULT_IMG_SIZE) -> Model:
    """Engine 1: MobileNetV3Small — Transfer Learning (base frozen by default)."""
    base_model = MobileNetV3Small(input_shape=img_size + (3,), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze: hanya train head dulu (Fase 1)

    inputs = Input(shape=img_size + (3,))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name="MobileNetV3")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_densenet121_model(num_classes: int = NUM_CLASSES, img_size: tuple = DEFAULT_IMG_SIZE) -> Model:
    """Engine 2: DenseNet121 — Transfer Learning (base frozen by default)."""
    base_model = DenseNet121(input_shape=img_size + (3,), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze: hanya train head dulu (Fase 1)

    inputs = Input(shape=img_size + (3,))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name="DenseNet121")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def unfreeze_model(model, num_layers_to_unfreeze: int = 20, learning_rate: float = 1e-5):
    """Fase 2: Unfreeze beberapa layer terakhir base model untuk fine-tuning.
    
    Args:
        model: Model Keras yang sudah di-train Fase 1.
        num_layers_to_unfreeze: Jumlah layer terakhir base model yang di-unfreeze.
        learning_rate: Learning rate rendah untuk fine-tuning agar tidak merusak weights.
    """
    base_model = model.layers[1]  # Layer kedua adalah base model
    base_model.trainable = True
    
    # Freeze semua layer kecuali N layer terakhir
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
    
    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    total_count = len(base_model.layers)
    print(f"[Fine-tune] {trainable_count}/{total_count} layers base model di-unfreeze")
    
    # Re-compile dengan learning rate lebih rendah
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def get_callbacks(model_save_path: str = 'saved_models/asl_model_best.keras',
                  patience: int = 7) -> list:
    """Create training callbacks."""
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    return [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
    ]

# ============================================================================
# SECTION 4: HAND DETECTION (MediaPipe — kompatibel versi lama & baru)
# ============================================================================

@dataclass
class HandResult:
    """Container for hand detection results."""
    landmarks: np.ndarray
    bbox: Tuple[int, int, int, int]
    handedness: str
    confidence: float

# Koneksi antar landmark tangan untuk drawing manual
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),       # Thumb
    (0,5),(5,6),(6,7),(7,8),       # Index
    (0,9),(9,10),(10,11),(11,12),  # Middle
    (0,13),(13,14),(14,15),(15,16),# Ring
    (0,17),(17,18),(18,19),(19,20),# Pinky
    (5,9),(9,13),(13,17)           # Palm
]

class HandDetector:
    """Hand detection using MediaPipe. 
    Kompatibel dengan:
    - mediapipe <= 0.10.14 (mp.solutions API)
    - mediapipe >= 0.10.21 (mp.tasks API)
    """

    def __init__(self, max_num_hands: int = 2, min_detection_confidence: float = 0.7, 
                 static_mode: bool = False, model_path: str = 'hand_landmarker.task'):
        import mediapipe as mp
        self._use_tasks_api = False
        self._mp = mp

        # Coba gunakan mp.solutions (versi lama) terlebih dahulu
        try:
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            self.hands = self.mp_hands.Hands(
                static_image_mode=static_mode,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=0.5
            )
            print("[HandDetector] Menggunakan mp.solutions API (legacy)")
        except AttributeError:
            # mp.solutions tidak tersedia → gunakan mp.tasks API (versi baru)
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision as mp_vision

            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"[HandDetector] File model '{model_path}' tidak ditemukan!\n"
                    f"Download dari: https://storage.googleapis.com/mediapipe-models/"
                    f"hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
                )
            
            base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=max_num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_tracking_confidence=0.5,
                running_mode=mp_vision.RunningMode.IMAGE if static_mode else mp_vision.RunningMode.IMAGE
            )
            self.hands = mp_vision.HandLandmarker.create_from_options(options)
            self._use_tasks_api = True
            print("[HandDetector] Menggunakan mp.tasks API (baru)")

        self._results = None

    def detect(self, frame: np.ndarray, draw: bool = False) -> List[HandResult]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected_hands = []
        h, w = frame.shape[:2]

        if self._use_tasks_api:
            # --- MediaPipe Tasks API (baru) ---
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
            result = self.hands.detect(mp_image)

            if result.hand_landmarks:
                for idx, hand_lms in enumerate(result.hand_landmarks):
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float32)
                    x_coords = (landmarks[:, 0] * w).astype(int)
                    y_coords = (landmarks[:, 1] * h).astype(int)
                    padding = 40
                    x = max(0, int(x_coords.min()) - padding)
                    y_min = max(0, int(y_coords.min()) - padding)
                    x2 = min(w, int(x_coords.max()) + padding)
                    y2 = min(h, int(y_coords.max()) + padding)

                    handedness = 'Right'
                    confidence = 0.0
                    if result.handedness and idx < len(result.handedness):
                        handedness = result.handedness[idx][0].category_name
                        confidence = result.handedness[idx][0].score

                    detected_hands.append(HandResult(landmarks, (x, y_min, x2-x, y2-y_min), handedness, confidence))

                    if draw:
                        self._draw_landmarks_manual(frame, landmarks, w, h)
        else:
            # --- MediaPipe Solutions API (lama) ---
            self._results = self.hands.process(rgb_frame)

            if self._results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(self._results.multi_hand_landmarks):
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
                    x_coords = (landmarks[:, 0] * w).astype(int)
                    y_coords = (landmarks[:, 1] * h).astype(int)
                    padding = 40
                    x = max(0, x_coords.min() - padding)
                    y_min = max(0, y_coords.min() - padding)
                    x2 = min(w, x_coords.max() + padding)
                    y2 = min(h, y_coords.max() + padding)

                    handedness = 'Right'
                    confidence = 0.0
                    if self._results.multi_handedness:
                        handedness = self._results.multi_handedness[idx].classification[0].label
                        confidence = self._results.multi_handedness[idx].classification[0].score

                    detected_hands.append(HandResult(landmarks, (x, y_min, x2-x, y2-y_min), handedness, confidence))

                    if draw:
                        self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style())
        return detected_hands

    def _draw_landmarks_manual(self, frame: np.ndarray, landmarks: np.ndarray, w: int, h: int):
        """Draw hand landmarks dan connections secara manual (untuk Tasks API)."""
        pts = [(int(lm[0] * w), int(lm[1] * h)) for lm in landmarks]
        
        # Draw connections (garis antar landmark)
        for start, end in HAND_CONNECTIONS:
            if start < len(pts) and end < len(pts):
                cv2.line(frame, pts[start], pts[end], (0, 255, 0), 2)
        
        # Draw landmark points
        for pt in pts:
            cv2.circle(frame, pt, 4, (0, 0, 255), -1)

    def release(self):
        if hasattr(self, 'hands'):
            self.hands.close()

# (DualEngineASLClassifier removed — tidak digunakan di sistem image-based)

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

# (extract_landmarks_from_dataset removed — tidak digunakan di sistem image-based)

# ============================================================================
# SECTION 8: EVALUATION FUNCTIONS
# ============================================================================

def evaluate_model(model, test_generator):
    """Evaluate model: Classification Report, Confusion Matrix, Accuracy, Precision, Recall, F1-Score, Inference Time.
    Semua hasil disimpan ke folder saved_generate/ sebagai PNG."""
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend agar tidak blocking
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd 
    import os
    import time

    OUTPUT_DIR = 'saved_generate'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Prediksi data + hitung inference time
    print(f"\n[Evaluasi] Mengevaluasi {model.name}...")
    start_time = time.time()
    predictions = model.predict(test_generator, verbose=1)
    total_inference_time = time.time() - start_time
    num_images = len(test_generator.filenames)
    inference_time = total_inference_time / num_images

    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    target_names = list(test_generator.class_indices.keys())

    # ============================================================
    # CLASSIFICATION REPORT (Heatmap PNG)
    # ============================================================
    report_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).iloc[:-1, :].T

    plt.figure(figsize=(10, 12))
    sns.heatmap(report_df, annot=True, cmap="YlGnBu", cbar=False, fmt=".2f")
    plt.title(f'Classification Report - {model.name}')
    report_file = os.path.join(OUTPUT_DIR, f'report_{model.name.lower()}.png')
    plt.savefig(report_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Classification Report disimpan: {report_file}")

    # ============================================================
    # CONFUSION MATRIX (Heatmap PNG)
    # ============================================================
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix - {model.name}')
    plt.ylabel('Label Asli')
    plt.xlabel('Prediksi AI')
    plt.tight_layout()
    matrix_file = os.path.join(OUTPUT_DIR, f'confusion_matrix_{model.name.lower()}.png')
    plt.savefig(matrix_file, dpi=300)
    plt.close()
    print(f"  Confusion Matrix disimpan: {matrix_file}")

    # ============================================================
    # PRINT METRICS
    # ============================================================
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"\n  === Hasil Evaluasi {model.name} ===")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  Inference : {inference_time:.6f} detik/gambar ({1/inference_time:.1f} FPS)")
    print(f"  Total waktu inferensi: {total_inference_time:.2f} detik untuk {num_images} gambar")

    return {
        'accuracy': accuracy, 'precision': precision,
        'recall': recall, 'f1_score': f1,
        'inference_time': inference_time
    }


def plot_training_history(history, model_name: str = 'model'):
    """Plot training history dan simpan ke saved_generate/."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs('saved_generate', exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title(f'{model_name} - Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title(f'{model_name} - Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    save_path = f'saved_generate/training_history_{model_name.lower()}.png'
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  Training history disimpan: {save_path}")