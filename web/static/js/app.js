/**
 * ASL Recognition Web Application
 * Aplikasi Penerjemah Bahasa Isyarat ASL
 * 
 * AUTO-SPEAK LOGIC (FIXED):
 * =========================
 * - Baca seluruh kalimat setelah 1 detik tidak ada perubahan
 * - Tidak bergantung pada currentWord (karena currentWord tetap tampil)
 * - Hanya cek apakah sentence berubah
 */

class ASLApp {
    constructor() {
        this.pollInterval = null;
        this.pollRate = 100;
        this.isConnected = false;
        
        // Auto-speak settings
        this.autoSpeakEnabled = true;
        this.autoSpeakDelay = 1000; // 1 detik
        
        // Tracking
        this.lastSpokenSentence = '';     // Kalimat terakhir yang sudah dibaca
        this.sentenceChangedTime = 0;     // Waktu terakhir kalimat berubah
        this.isTyping = false;            // Apakah sedang mengetik (currentWord ada isinya)
        this.hasStarted = false;          // Flag untuk welcome message
        
        this.elements = {
            statusDot: document.getElementById('statusDot'),
            statusText: document.getElementById('statusText'),
            detectedLetter: document.getElementById('detectedLetter'),
            confidence: document.getElementById('confidence'),
            currentWord: document.getElementById('currentWord'),
            sentenceDisplay: document.getElementById('sentenceDisplay'),
            progressFill: document.getElementById('progressFill'),
            progressText: document.getElementById('progressText'),
            btnClear: document.getElementById('btnClear'),
            btnSpeak: document.getElementById('btnSpeak'),
            modeButtons: document.querySelectorAll('.mode-btn'),
            autoSpeakTimer: document.getElementById('autoSpeakTimer')
        };
        
        this.init();
    }
    
    init() {
        this.bindEvents();
        this.startPolling();
        this.setConnected(true);
        this.startAutoSpeakCheck();
    }
    
    bindEvents() {
        this.elements.btnClear.addEventListener('click', () => this.clearText());
        this.elements.btnSpeak.addEventListener('click', () => this.speakText());
        
        this.elements.modeButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                const mode = btn.dataset.mode;
                this.setMode(mode);
                this.elements.modeButtons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
            });
        });
        
        document.addEventListener('keydown', (e) => {
            if (e.key === 's' || e.key === 'S') {
                this.speakText();
            } else if (e.key === 'c' || e.key === 'C') {
                this.clearText();
            } else if (e.key >= '1' && e.key <= '4') {
                const modes = ['instant', 'balanced', 'accurate', 'strict'];
                const index = parseInt(e.key) - 1;
                if (modes[index]) {
                    this.setMode(modes[index]);
                    this.elements.modeButtons.forEach(b => {
                        b.classList.toggle('active', b.dataset.mode === modes[index]);
                    });
                }
            }
        });
    }
    
    startPolling() {
        this.pollInterval = setInterval(() => this.fetchState(), this.pollRate);
    }
    
    stopPolling() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
    }
    
    /**
     * AUTO-SPEAK - SIMPLIFIED LOGIC
     * =============================
     * 1. Jika sentence berubah, reset timer
     * 2. Jika 1 detik berlalu tanpa perubahan, baca kalimat
     * 3. Tidak peduli currentWord (karena bisa tetap tampil)
     */
    startAutoSpeakCheck() {
        setInterval(() => {
            if (!this.autoSpeakEnabled) return;
            
            const sentence = this.elements.sentenceDisplay.textContent;
            const isPlaceholder = sentence === 'Mulai bahasa isyarat untuk menerjemahkan...';
            
            // Jika sentence berubah, reset timer
            if (sentence !== this.lastSentence) {
                this.lastSentence = sentence;
                this.sentenceChangedTime = Date.now();
                this.speakScheduled = false;
                console.log('Sentence berubah:', sentence);
            }
            
            // Hitung waktu sejak perubahan terakhir
            const timeSinceChange = Date.now() - this.sentenceChangedTime;
            
            // Kondisi untuk speak:
            // 1. Bukan placeholder
            // 2. Sentence berbeda dari yang terakhir dibaca
            // 3. Sudah 1 detik sejak perubahan
            // 4. Belum di-schedule
            const canSpeak = !isPlaceholder &&
                            sentence.trim() !== '' &&
                            sentence !== this.lastSpokenSentence &&
                            timeSinceChange >= this.autoSpeakDelay &&
                            !this.speakScheduled;
            
            // Update display
            if (isPlaceholder || sentence.trim() === '') {
                this.elements.autoSpeakTimer.textContent = 'Menunggu kalimat...';
            } else if (sentence === this.lastSpokenSentence) {
                this.elements.autoSpeakTimer.textContent = 'Sudah dibacakan';
            } else {
                const remaining = Math.max(0, this.autoSpeakDelay - timeSinceChange);
                if (remaining > 0) {
                    this.elements.autoSpeakTimer.textContent = `Akan dibaca dalam ${(remaining / 1000).toFixed(1)} detik`;
                } else {
                    this.elements.autoSpeakTimer.textContent = 'Sedang membaca...';
                }
            }
            
            // AUTO-SPEAK
            if (canSpeak) {
                this.speakScheduled = true;
                console.log('AUTO-SPEAK:', sentence);
                this.speakSentence(sentence);
                this.lastSpokenSentence = sentence;
                
                setTimeout(() => {
                    this.elements.autoSpeakTimer.textContent = 'Selesai dibacakan';
                }, 100);
            }
            
        }, 100);
    }
    
    async fetchState() {
        try {
            const response = await fetch('/api/state');
            if (!response.ok) throw new Error('API error');
            
            const state = await response.json();
            this.updateUI(state);
            
            if (!this.isConnected) {
                this.setConnected(true);
            }
        } catch (error) {
            console.error('State fetch error:', error);
            this.setConnected(false);
        }
    }
    
    updateUI(state) {
        const letter = state.letter || '-';
        const confidence = state.confidence || 0;
        
        this.elements.detectedLetter.textContent = letter;
        this.elements.confidence.textContent = `${Math.round(confidence * 100)}%`;
        
        const validation = state.validation || {};
        const validationStatus = validation.status || 'waiting';
        
        if (validationStatus === 'accepted') {
            this.elements.detectedLetter.style.color = '#22c55e';
        } else if (validationStatus.includes('streak')) {
            this.elements.detectedLetter.style.color = '#f59e0b';
        } else if (confidence > 0) {
            this.elements.detectedLetter.style.color = '#ef4444';
        } else {
            this.elements.detectedLetter.style.color = '#6b6b7b';
        }
        
        const word = state.word || '-';
        this.elements.currentWord.textContent = word || '-';
        
        const sentence = state.sentence || '';
        
        // Logic welcome message: hanya muncul pertama kali
        if (sentence && sentence.trim() !== '') {
            this.elements.sentenceDisplay.textContent = sentence;
            this.hasStarted = true;
            this.elements.sentenceDisplay.classList.remove('text-muted');
        } else {
            if (!this.hasStarted) {
                this.elements.sentenceDisplay.textContent = 'Tunjukkan isyarat tangan ke kamera untuk mulai...';
                this.elements.sentenceDisplay.classList.add('text-muted');
            } else {
                // Jika sudah pernah mulai tapi sekarang kosong (misal dihapus), biarkan kosong
                this.elements.sentenceDisplay.textContent = ''; 
                this.elements.sentenceDisplay.classList.remove('text-muted');
            }
        }
        
        const streak = validation.streak || 0;
        const requiredStreak = validation.required_streak || 5;
        
        if (validationStatus.includes('streak') && streak > 0) {
            const progress = Math.min((streak / requiredStreak) * 100, 100);
            this.elements.progressFill.style.width = `${progress}%`;
            this.elements.progressText.textContent = `Tahan posisi... ${streak}/${requiredStreak}`;
            this.elements.progressFill.style.background = 'linear-gradient(90deg, #f59e0b, #fbbf24)';
        } else if (validationStatus === 'accepted') {
            this.elements.progressFill.style.width = '100%';
            this.elements.progressText.textContent = 'Berhasil ditangkap!';
            this.elements.progressFill.style.background = 'linear-gradient(90deg, #10b981, #34d399)';
        } else {
            const pending = state.pending || {};
            if (pending.letter && pending.required > 0) {
                const progress = Math.min((pending.count / pending.required) * 100, 100);
                this.elements.progressFill.style.width = `${progress}%`;
                this.elements.progressText.textContent = `${pending.letter}: ${pending.count}/${pending.required}`;
                
                if (progress >= 100) {
                    this.elements.progressFill.style.background = 'linear-gradient(90deg, #22c55e, #4ade80)';
                } else {
                    this.elements.progressFill.style.background = 'linear-gradient(90deg, #6366f1, #818cf8)';
                }
            } else {
                this.elements.progressFill.style.width = '0%';
                let statusText = 'Siap menangkap isyarat...';
                if (validationStatus === 'low_conf') {
                    statusText = 'Kurang jelas, coba tahan lebih lama';
                } else if (validationStatus === 'ambiguous') {
                    statusText = 'Bentuk kurang jelas, coba lebih tegas';
                } else if (validationStatus === 'hand_moving' || validationStatus === 'keep_still') {
                    statusText = 'Tangan bergerak, coba tahan diam...';
                } else if (validationStatus === 'no_hand') {
                    statusText = 'Arahkan tangan ke kamera';
                }
                this.elements.progressText.textContent = statusText;
            }
        }
    }
    
    setConnected(connected) {
        this.isConnected = connected;
        
        if (connected) {
            this.elements.statusDot.classList.remove('offline');
            this.elements.statusText.textContent = 'Terhubung';
        } else {
            this.elements.statusDot.classList.add('offline');
            this.elements.statusText.textContent = 'Terputus';
        }
    }
    
    async setMode(mode) {
        try {
            await fetch('/api/mode', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode })
            });
        } catch (error) {
            console.error('Error:', error);
        }
    }
    
    async clearText() {
        try {
            await fetch('/api/clear', { method: 'POST' });
            this.elements.currentWord.textContent = '-';
            this.elements.sentenceDisplay.textContent = 'Mulai bahasa isyarat untuk menerjemahkan...';
            this.lastSentence = '';
            this.lastSpokenSentence = '';
            this.sentenceChangedTime = 0;
            this.speakScheduled = false;
        } catch (error) {
            console.error('Error:', error);
        }
    }
    
    async speakText() {
        try {
            await fetch('/api/speak', { method: 'POST' });
        } catch (error) {
            console.error('Error:', error);
        }
    }
    
    // Baca kalimat sebagai KATA (bukan per huruf)
    // "A K U" -> dibaca "AKU"
    async speakSentence(sentence) {
        try {
            await fetch('/api/speak_as_words', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: sentence })
            });
        } catch (error) {
            console.error('Error:', error);
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.aslApp = new ASLApp();
});
