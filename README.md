# ASL Recognition Web Application

Sistem **Computer Vision** untuk pengenalan bahasa isyarat **ASL (American Sign Language)** secara real-time menggunakan **Deep Learning** dan **MediaPipe Hand Detection**. Aplikasi ini memungkinkan pengguna untuk berkomunikasi menggunakan bahasa isyarat yang kemudian diterjemahkan menjadi teks dan suara.

---

## 📋 Daftar Isi

1. [Tentang Proyek](#-tentang-proyek)
2. [Instalasi](#-instalasi)
3. [Menjalankan Aplikasi](#-menjalankan-aplikasi)
4. [Cara Penggunaan](#-cara-penggunaan)
5. [Teknologi yang Digunakan](#-teknologi-yang-digunakan)
6. [Arsitektur Sistem](#-arsitektur-sistem)
7. [Struktur Proyek](#-struktur-proyek)
8. [API Reference](#-api-reference)
9. [Training Model (Opsional)](#-training-model-opsional)
10. [Troubleshooting](#-troubleshooting)

---

## 🎯 Tentang Proyek

### Latar Belakang
Proyek ini dikembangkan sebagai solusi untuk membantu komunikasi antara penyandang tunarungu/tunawicara dengan masyarakat umum. Dengan menggunakan kamera webcam, sistem dapat mengenali gerakan tangan bahasa isyarat ASL dan menerjemahkannya menjadi huruf/kata dalam waktu nyata.

### Fitur Utama

| Fitur | Deskripsi |
|-------|-----------|
| **Real-time Recognition** | Pengenalan huruf/simbol ASL secara langsung melalui webcam |
| **Dual Classification** | Kombinasi CNN (MobileNetV2) dan Landmark-based classifier |
| **Multi-stage Validation** | Validasi berlapis untuk mengurangi kesalahan deteksi |
| **Text-to-Speech** | Konversi hasil terjemahan ke suara dalam Bahasa Indonesia |
| **4 Mode Kecepatan** | Cepat, Normal, Teliti, Sangat Teliti sesuai kebutuhan |
| **Web Interface** | Tampilan modern dengan dark theme dan responsive design |

### Kelas yang Dikenali
Sistem dapat mengenali **29 kelas**:
- **Huruf A-Z** (26 huruf alfabet)
- **del** - Hapus karakter terakhir
- **space** - Tambah spasi (finalisasi kata)
- **nothing** - Tidak ada gesture terdeteksi

---

## 📦 Instalasi

### Persyaratan Sistem

> ⚠️ **Penting:** Gunakan Python 3.11 untuk kompatibilitas terbaik. Python 3.12+ mungkin memiliki masalah.

### Langkah Instalasi

#### 1. Buka Terminal/Command Prompt

Masuk ke folder proyek:
```bash
cd path/ke/Comvis-Project
```

#### 2. Install Dependencies

**Cara Cepat (Rekomendasi):**
```bash
pip install -r requirements.txt
```

**Jika ada error, install manual dengan versi spesifik:**
```bash
pip install tensorflow==2.16.2
pip install mediapipe==0.10.14
pip install protobuf==4.25.8
pip install opencv-python flask flask-cors numpy scikit-learn pillow gtts tqdm matplotlib seaborn
```

> ⚠️ **PENTING:** Jangan gunakan MediaPipe versi 0.10.21 ke atas karena tidak kompatibel!

#### 3. Verifikasi Instalasi

```bash
python -c "import tensorflow; import mediapipe; import cv2; import flask; print('✅ Instalasi berhasil!')"
```

#### 4. Pastikan Model Tersedia

Cek folder `saved_models/` berisi:
```
saved_models/
├── asl_model_best.keras        # Model CNN (~28 MB)
├── landmark_classifier.keras   # Model Landmark (~782 KB)
└── landmark_classifier_classes.npy
```

> Jika model tidak ada, lihat bagian [Training Model](#-training-model-opsional).

---

## 🚀 Menjalankan Aplikasi

### 1. Jalankan Server

```bash
python web_server.py --port 8000
```

### 2. Buka Browser

Akses alamat:
```
http://localhost:8000
```

### 3. Izinkan Akses Kamera

Klik **"Allow"** atau **"Izinkan"** saat browser meminta izin kamera.

### Output Terminal yang Normal

```
============================================================
  ASL Recognition Web Server
============================================================

  Open browser: http://localhost:8000
  Press Ctrl+C to stop
============================================================

[Server] Initializing ASL processor...
[Server] Model loaded successfully
[LandmarkClassifier] Model loaded
[Server] ASL processor ready
 * Running on http://0.0.0.0:8000/
```

### Menghentikan Server

Tekan `Ctrl+C` di terminal.

---

## 🎮 Cara Penggunaan

### Keyboard Shortcuts

| Tombol | Fungsi |
|--------|--------|
| **Spasi** | Finalisasi kata (tambah spasi) |
| **C** | Hapus semua teks |
| **S** | Bicara (Text-to-Speech) |
| **1** | Mode Cepat |
| **2** | Mode Normal (default) |
| **3** | Mode Teliti |
| **4** | Mode Sangat Teliti |

### Mode Pengenalan

| Mode | Kecepatan | Akurasi | Untuk Siapa |
|------|-----------|---------|-------------|
| **Cepat** | ⚡⚡⚡ | ⭐⭐ | Pengguna mahir |
| **Normal** | ⚡⚡ | ⭐⭐⭐ | Semua pengguna |
| **Teliti** | ⚡ | ⭐⭐⭐⭐ | Pemula |
| **Sangat Teliti** | 🐢 | ⭐⭐⭐⭐⭐ | Akurasi maksimal |

### Tips Penggunaan

1. **Pencahayaan** - Pastikan ruangan terang
2. **Posisi** - Tangan di tengah layar, jarak 30-60 cm
3. **Stabilitas** - Tahan gesture sampai progress bar hijau penuh
4. **Latar belakang** - Gunakan latar polos jika memungkinkan

### Indikator Warna

| Warna | Arti |
|-------|------|
| 🟢 Hijau | Huruf diterima |
| 🟡 Kuning | Sedang validasi |
| 🟠 Orange | Ditolak (kurang yakin) |

---

## 🛠 Teknologi yang Digunakan

### Machine Learning & Computer Vision

| Library | Versi | Fungsi |
|---------|-------|--------|
| TensorFlow | 2.16.2 | Framework deep learning |
| MediaPipe | 0.10.14 | Hand landmark detection |
| OpenCV | 4.10+ | Image processing |
| Keras | 3.x | Neural network API |
| scikit-learn | 1.3+ | ML utilities |

### Web & Audio

| Library | Fungsi |
|---------|--------|
| Flask | Web server |
| Flask-CORS | Cross-origin requests |
| gTTS | Text-to-Speech Indonesia |

---

## 🏗 Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────┐
│                      WEB BROWSER                             │
│   Video Feed  │  Hasil Terjemahan  │  Tombol Kontrol        │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    FLASK WEB SERVER                          │
│         /video_feed  │  /api/state  │  /api/speak           │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   ASL WEB PROCESSOR                          │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │ MediaPipe   │  │  Landmark    │  │   Text Buffer       │ │
│  │ (Hand Det.) │─▶│  Classifier  │─▶│   (Word Builder)    │ │
│  └─────────────┘  └──────────────┘  └─────────────────────┘ │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     TRAINED MODELS                           │
│   asl_model_best.keras  │  landmark_classifier.keras        │
└─────────────────────────────────────────────────────────────┘
```

### Alur Kerja

1. **Capture** - Webcam menangkap frame
2. **Detect** - MediaPipe mendeteksi tangan dan 21 titik landmark
3. **Classify** - Model mengklasifikasi gesture menjadi huruf
4. **Validate** - Multi-stage validation (confidence, stability, temporal)
5. **Output** - Huruf ditampilkan dan bisa dikonversi ke suara

---

## 📁 Struktur Proyek

```
Comvis-Project/
├── README.md                         # Dokumentasi (file ini)
├── requirements.txt                  # Daftar dependencies
│
├── web_server.py                     # Flask web server
├── asl_modules.py                    # Core modules ASL
├── asl_recognition_complete.ipynb    # Notebook training
│
├── saved_models/                     # Model terlatih
│   ├── asl_model_best.keras
│   ├── landmark_classifier.keras
│   └── landmark_classifier_classes.npy
│
├── web/                              # Frontend
│   ├── templates/index.html
│   └── static/
│       ├── css/style.css
│       └── js/app.js
│
└── dataset/                          # (Opsional) Data training
    └── asl_alphabet_train/
```

---

## 🔌 API Reference

| Endpoint | Method | Fungsi |
|----------|--------|--------|
| `/` | GET | Halaman utama |
| `/video_feed` | GET | Stream video MJPEG |
| `/api/state` | GET | Status pengenalan saat ini |
| `/api/mode` | POST | Ubah mode (`{"mode": "balanced"}`) |
| `/api/clear` | POST | Hapus semua teks |
| `/api/speak` | POST | Text-to-Speech |

### Contoh Response `/api/state`

```json
{
  "letter": "A",
  "confidence": 0.95,
  "word": "HALO",
  "sentence": "HALO ",
  "mode": "balanced",
  "validation": {
    "status": "accepted",
    "streak": 4
  }
}
```

---

## 🎓 Training Model (Opsional)

Jika model tidak tersedia atau ingin melatih ulang:

### 1. Siapkan Dataset

Download dari [Kaggle ASL Alphabet](https://www.kaggle.com/grassknoted/asl-alphabet) dan extract ke:
```
dataset/asl_alphabet_train/
├── A/
├── B/
└── ... (sampai Z, del, nothing, space)
```

### 2. Jalankan Notebook

```bash
jupyter notebook asl_recognition_complete.ipynb
```

Jalankan semua cells dari atas ke bawah.

### 3. Verifikasi

Pastikan file model tersimpan di `saved_models/`.

---

## 🔧 Troubleshooting

### ❌ `module 'mediapipe' has no attribute 'solutions'`

```bash
pip uninstall mediapipe -y
pip install mediapipe==0.10.14
```

### ❌ `ImportError: cannot import name 'runtime_version'`

```bash
pip install tensorflow==2.16.2 protobuf==4.25.8
```

### ❌ `AttributeError: module 'ml_dtypes' has no attribute 'float8_e3m4'`

```bash
pip uninstall jax jaxlib -y
```

### ❌ Kamera Tidak Terdeteksi

1. Izinkan akses kamera di browser
2. Tutup aplikasi lain yang menggunakan kamera
3. Coba port lain: `python web_server.py --port 5001`

### ❌ Port Sudah Digunakan

**macOS/Linux:**
```bash
lsof -i :8000
kill -9 <PID>
```

**Windows:**
```bash
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### ❌ Model Tidak Ditemukan

Pastikan folder `saved_models/` berisi file `.keras`. Jika kosong, lakukan training model.

---

## 📊 Dataset

| Atribut | Nilai |
|---------|-------|
| Sumber | ASL Alphabet Dataset (Kaggle) |
| Jumlah Kelas | 29 |
| Ukuran Gambar | 200x200 RGB |
| Total Gambar | ~87,000 |

---

## 📄 Lisensi

MIT License © 2026

---

**Dibuat untuk membantu komunikasi yang lebih inklusif** ❤️
