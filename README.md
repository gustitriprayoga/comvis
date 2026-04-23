BISINDO AI - Dual-Engine Real-time Translator

Sistem **Computer Vision** tingkat lanjut untuk pengenalan bahasa isyarat **BISINDO** secara real-time menggunakan **MediaPipe Hand Detection (2 Tangan / 126 Titik)** dan arsitektur **Dual-Engine Deep Learning**. Sistem ini tidak sekadar menebak, melainkan melacak secara cepat dan memvalidasi secara akurat.

---

## 📋 Daftar Isi

1. [Tentang Proyek](#-tentang-proyek)
2. [Fitur Utama](#-fitur-utama)
3. [Arsitektur Dual-Engine](#-arsitektur-dual-engine)
4. [Instalasi](#-instalasi)
5. [Cara Penggunaan](#-cara-penggunaan)
6. [Struktur Proyek](#-struktur-proyek)
7. [Troubleshooting](#-troubleshooting)

---

## 🎯 Tentang Proyek

Proyek ini dikembangkan untuk menerjemahkan bahasa isyarat BISINDO menjadi teks dan suara secara real-time melalui kamera webcam. Pada versi V2.0 ini, sistem telah dirombak menggunakan **Dual-Engine Model** yang memisahkan tugas pelacakan gerakan (Tracking) dan tugas verifikasi (Akurasi) untuk meminimalisir kesalahan (_typo_) dan mencegah _double typing_.

## ✨ Fitur Utama

- **Deteksi 2 Tangan (126 Landmarks):** Mengekstrak fitur spasial 3D secara detail (dilengkapi logika _anti-flip_ dan _relative positioning_ dari pergelangan tangan).
- **Dual-Engine Recognition:** Kombinasi cerdas antara kecepatan dan presisi tinggi menggunakan dua otak AI yang saling melengkapi.
- **Auto-Speak (Text-to-Speech):** Membacakan kalimat yang sudah diterjemahkan secara otomatis dengan suara (dioptimalkan pembacaan per kata).
- **Live Evaluation UI:** Menampilkan _Classification Report_ dan _Confusion Matrix_ secara langsung dari dalam website untuk transparansi performa model.
- **Smart Buffer:** Mencegah deteksi berulang/spam saat posisi tangan pengguna masih bergerak atau belum stabil.

## 🧠 Arsitektur Dual-Engine

Sistem menggunakan dua model Deep Learning yang bertugas layaknya _Support_ dan _Carry_:

1. **Engine 1: MobileNetV2-style (Fast Tracker):** Bekerja di barisan depan. Model ini sangat ringan dan responsif, bertugas memantau isyarat tangan secara real-time saat tangan masih bergerak mencari posisi yang pas.
2. **Engine 2: EfficientNetB0-style (High Accuracy Validator):** Bertindak sebagai "Hakim". Hanya akan dipanggil ketika posisi tangan sudah ditahan stabil namun Engine 1 masih ragu (< 90% confidence). Hanya model ini yang diizinkan untuk mengetik huruf ke layar agar terjemahan 100% akurat.

---

## ⚙️ Instalasi

### 1. Clone Repository & Setup Virtual Environment

Sangat disarankan menggunakan virtual environment agar _library_ tidak bentrok.

```bash
git clone <repo-url>
cd comvis
python -m venv venv

# Aktifkan venv (Untuk Windows)
venv\Scripts\activate

# Aktifkan venv (Untuk Mac/Linux)
source venv/bin/activate
2. Install Dependencies
Pastikan menggunakan versi Python yang direkomendasikan (3.9 - 3.11).

Bash
pip install numpy opencv-python tensorflow mediapipe flask flask-cors scikit-learn matplotlib seaborn tqdm gtts pygame pandas
🚀 Cara Penggunaan
Tahap 1: Persiapan Dataset & Training (WAJIB)
Karena menggunakan sistem Dual-Engine terbaru, Anda wajib melatih ulang AI agar file otak .keras dan grafik evaluasi tercipta.

Siapkan folder dataset BISINDO (berisi folder A-Z) di komputer Anda.

Pastikan path dataset di file train_baru.py sudah sesuai dengan lokasi folder Anda (contoh: E:\Project\dataset\bisindo\images\train).

Jalankan proses ekstraksi dan training:

Bash
python train_baru.py
Tunggu hingga terminal menampilkan Classification Report untuk kedua mesin dan otomatis menyimpan grafik ke folder saved_generate/.

Tahap 2: Menjalankan Web Server
Setelah dua file .keras selesai dibuat dari tahap training, jalankan server UI:

Bash
python web_server.py
Buka browser dan akses: http://localhost:5000

📂 Struktur Proyek
Plaintext
├── asl_modules.py           # Core sistem: Dual-Engine, TextBuffer, HandDetector, Metrik Evaluasi
├── train_baru.py            # Script pipeline untuk ekstraksi 126 titik & melatih dua AI sekaligus
├── web_server.py            # Flask backend dengan logika Real-time Switching Engine
├── saved_models/            # [Otomatis] Folder tempat AI menyimpan 'otak'-nya (.keras & .npy)
├── saved_generate/          # [Otomatis] Folder output grafik evaluasi AI (Report & Matrix PNG)
└── web/                     # Frontend UI
    ├── templates/index.html # Tampilan Web (UI/UX modern + Modal Evaluasi)
    └── static/
        ├── css/style.css    # CSS Styling
        └── js/app.js        # Script integrasi API dan fungsi Frontend
🔧 Troubleshooting
1. Kamera menampikan huruf "?", Confidence 0%, dan "Model: None"
Penyebab: AI belum dilatih sehingga file .keras tidak ditemukan.
Solusi: Tutup server, jalankan python train_baru.py hingga selesai 100%, lalu jalankan kembali web_server.py.

2. Modal Grafik Evaluasi di Website Tidak Muncul
Penyebab: File gambar evaluasi belum ter-generate.
Solusi: Sama seperti poin 1, pastikan Anda merunning train_baru.py agar folder saved_generate terisi dengan grafik evaluasi terbaru.

3. Kamera Tidak Terdeteksi / Error OpenCV
Pastikan tidak ada aplikasi lain (Zoom, GMeet, dll) yang sedang menahan akses kamera Anda.

4. Error ModuleNotFoundError: No module named '...'
Pastikan Virtual Environment (venv) sudah aktif sebelum menjalankan pip install.
```
