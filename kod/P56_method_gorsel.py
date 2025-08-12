# 1. Gerekli kütüphaneleri ve fonksiyonları içe aktar
import numpy as np
import matplotlib.pyplot as plt
from P56_method_1 import active_speech_level
from audio_utils import read_wav_mono

# 2. Ses dosyasını oku ve P.56 metodunu kullanarak analiz et
fs, sig = read_wav_mono("C:\\D----------------\\Staj\\makale\\ses\\21.wav")
level_db, activity, info = active_speech_level(sig, fs)

# 3. Görselleştirme için gerekli verileri 'info' sözlüğünden al (yeniden hesaplama yapma)
q_tilde = info["q_tilde"]      # Hangover uygulanmış sinyal zarfı (envelope)
lo = info["lo"]                # Aktif konuşma için hesaplanan eşik değeri
active_mask = info["active_mask"] # Aktif bölgeleri gösteren boolean maske

# 4. Grafik için zaman eksenini oluştur
time = np.arange(len(sig)) / fs

# 5. Grafiği ve bileşenlerini çizdir
plt.figure(figsize=(12, 6))

plt.plot(time, sig, label="Orijinal Sinyal", color="gray", alpha=0.7)
plt.plot(time, q_tilde, label="Sinyal Zarfı (q_tilde)", color="orange")
plt.axhline(lo, color="red", linestyle="--", label=f"Aktif Konuşma Eşiği (lo = {lo:.4f})")
plt.fill_between(time, -1, 1, where=active_mask, color="green", alpha=0.2, label="Tespit Edilen Aktif Konuşma")

# Eksenleri ve başlığı isimlendir
plt.xlabel("Zaman (s)")
plt.ylabel("Genlik")
plt.title(f"Aktif Konuşma Bölgeleri (Seviye: {level_db:.2f} dBov, Aktivite: {activity*100:.2f}%)")
plt.legend()
plt.tight_layout()

# Grafiği göster
plt.show()
