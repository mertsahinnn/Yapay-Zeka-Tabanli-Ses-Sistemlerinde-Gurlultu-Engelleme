import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from audio_utils import read_wav_mono

"""
Tam istedigim gibi bir kod olmadi daha gelistirilecek

"""

def get_irs_filter_coefficients():
    """
    ITU-T P.48'de belirtilen ve P.862 (PESQ) gibi standartlarda kullanılan
    Değiştirilmiş IRS Gönderme (Modified IRS Send) filtresinin 
    katsayılarını döndürür.
    
    Bu katsayılar, 8000 Hz örnekleme frekansı için tasarlanmıştır.
    Filtre, ikinci dereceden bölümlerin (Second-Order Sections - SOS)
    bir kaskadı olarak uygulanır. Bu, sayısal kararlılığı artırır.

    Returns:
        sos (ndarray): Filtrenin SOS formatındaki katsayıları.
                       Her satır [b0, b1, b2, a0, a1, a2] formatındadır.
    """
    # 8 kHz örnekleme frekansı için Değiştirilmiş IRS Gönderme Filtresi
    # Katsayılar ITU-T G.191 STL'den (Software Tools Library) alınmıştır.
    # Bu, bir yüksek geçiren filtre ve bir alçak geçiren filtrenin
    # birleşiminden oluşur.
    
    # 1. Bölüm: Yüksek geçiren filtre (High-pass)
    b1 = np.array([1.0, -1.0, 0.0])
    a1 = np.array([1.0, -0.9375, 0.0])
    
    # 2. Bölüm: Karmaşık konjuge kutuplu rezonans filtresi (Low-pass-like)
    b2 = np.array([1.0, 0.4375, 0.0])
    a2 = np.array([1.0, -1.4375, 0.53125])

    # SciPy'nin sosfilt formatı için katsayıları birleştirme
    # [b0, b1, b2, a0, a1, a2]
    # Her zaman a0=1.0 varsayılır.
    sos = np.array([
        [b1[0], b1[1], b1[2], a1[0], a1[1], a1[2]],
        [b2[0], b2[1], b2[2], a2[0], a2[1], a2[2]]
    ])
    
    return sos

def apply_irs_filter(audio_signal, sample_rate):
    """
    Bir ses sinyaline ITU-T IRS filtresini uygular.
    
    Filtre katsayıları 8000 Hz için tasarlandığından, giriş sinyali
    eğer farklı bir örnekleme oranına sahipse otomatik olarak 8000 Hz'e
    yeniden örneklenir (resample).

    Args:
        audio_signal (np.ndarray): Filtrelenecek ses sinyali.
        sample_rate (int): Ses sinyalinin örnekleme frekansı (Hz).

    Returns:
        filtered_signal (np.ndarray): IRS filtresi uygulanmış ses sinyali.
                                      Örnekleme frekansı 8000 Hz'dir.
    """
    target_fs = 8000
    
    # Örnekleme frekansını kontrol et ve gerekirse yeniden örnekle
    if sample_rate != target_fs:
        print(f"Uyarı: Sinyal {sample_rate} Hz'den {target_fs} Hz'e yeniden örnekleniyor.")
        num_samples = int(len(audio_signal) * float(target_fs) / sample_rate)
        audio_signal = signal.resample(audio_signal, num_samples)
    
    # Sinyalin float64 formatında olduğundan emin ol
    audio_signal = audio_signal.astype(np.float64)
    
    # Filtre katsayılarını al
    irs_sos = get_irs_filter_coefficients()
    
    # Filtreyi uygula
    filtered_signal = signal.sosfilt(irs_sos, audio_signal)
    
    return filtered_signal



# --- GERÇEK KULLANIM SENARYOSU ---

# 1. Varsayımsal bir 16000 Hz'lik ses dosyası oluşturalım
#    Siz bu adımı atlayıp kendi dosyanızı yükleyeceksiniz.
sample_rate_16k = 16000
duration = 3 # saniye
frequency1 = 400  # Konuşma bandı içi
frequency2 = 6000 # Konuşma bandı dışı
t = np.linspace(0., duration, int(sample_rate_16k * duration), endpoint=False)
# Hem bant içi hem bant dışı frekans içeren bir sinyal
test_signal_16k = 0.5 * np.sin(2 * np.pi * frequency1 * t) + 0.3 * np.sin(2 * np.pi * frequency2 * t)
wavfile.write("ornek_16k.wav", sample_rate_16k, test_signal_16k.astype(np.float32))

# 2. 16000 Hz'lik ses dosyanızı yükleyin
try:
    # Kendi .wav dosyanızın adını buraya yazın
    orijinal_ornekleme_hizi, orijinal_sinyal = read_wav_mono("C:\\D----------------\\Staj\\makale\\ses\\20.wav")

    

    print(f"Ses dosyası başarıyla yüklendi: ornek_16k.wav")
    print(f"Orijinal Örnekleme Hızı: {orijinal_ornekleme_hizi} Hz")

    # 3. IRS filtresini doğrudan uygulayın
    # Fonksiyon, 16000 Hz'i algılayıp 8000 Hz'e indirecektir.
    filtrelenmis_sinyal = apply_irs_filter(orijinal_sinyal, orijinal_ornekleme_hizi)

    print(f"Sinyal başarıyla filtrelendi.")
    print(f"Filtrelenmiş sinyalin yeni örnekleme hızı: 8000 Hz")
    
    # 4. İsteğe Bağlı: Filtrelenmiş sinyali tekrar 16000 Hz'e çıkarmak
    # Eğer sonucun da orijinal örnekleme hızında olmasını istiyorsanız,
    # tekrar yukarı örnekleyebilirsiniz (upsample).
    num_samples_16k_final = int(len(filtrelenmis_sinyal) * float(orijinal_ornekleme_hizi) / 8000)
    filtrelenmis_sinyal_16k = signal.resample(filtrelenmis_sinyal, num_samples_16k_final)
    
    print("Filtrelenmiş sinyal tekrar 16000 Hz'e yükseltildi.")

    # 5. Sonucu yeni bir .wav dosyasına kaydedin
    wavfile.write("filtrelenmis_sonuc_16k.wav", orijinal_ornekleme_hizi, filtrelenmis_sinyal_16k.astype(np.float32))
    print("Filtrelenmiş sonuç 'filtrelenmis_sonuc_16k.wav' olarak kaydedildi.")

except FileNotFoundError:
    print("Hata: 'ornek_16k.wav' dosyası bulunamadı. Lütfen dosya adını kontrol edin.")
except Exception as e:
    print(f"Bir hata oluştu: {e}")



def plot_frequency_response(sos, fs):
    """Filtrenin frekans tepkisini çizer."""
    w, h = signal.sosfreqz(sos, worN=8000, fs=fs)
    db = 20 * np.log10(np.maximum(np.abs(h), 1e-5))
    
    plt.figure(figsize=(12, 6))
    plt.plot(w, db)
    plt.title('ITU-T IRS Filtresi Frekans Tepkisi')
    plt.xlabel('Frekans (Hz)')
    plt.ylabel('Kazanç (dB)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(-40, 5)
    plt.xlim(20, 4000)
    plt.show()

    

# --- Örnek Kullanım ---
if __name__ == '__main__':
    # 1. Filtrenin kendisinin nasıl davrandığını görelim
    fs = 8000
    irs_coefficients = get_irs_filter_coefficients()
    plot_frequency_response(irs_coefficients, fs)

    # 2. Bir test sinyali üzerinde deneyelim (Beyaz Gürültü)
    # Beyaz gürültünün frekans spektrumu düz olduğu için filtrenin şekli net görülür.
    np.random.seed(0)
    original_signal = np.random.randn(fs * 2) # 2 saniyelik beyaz gürültü

    # Filtreyi uygula
    filtered_signal = apply_irs_filter(original_signal, sample_rate=fs)

    # Zaman domeninde sinyalleri çiz
    time = np.linspace(0, len(original_signal)/fs, num=len(original_signal))
    plt.figure(figsize=(12, 6))
    plt.plot(time, original_signal, label='Orijinal Sinyal (Beyaz Gürültü)', alpha=0.7)
    plt.plot(time, filtered_signal, label='IRS Filtreli Sinyal', alpha=0.9)
    plt.title('Zaman Domeninde Sinyaller')
    plt.xlabel('Zaman (s)')
    plt.ylabel('Amplitüd')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 0.1) # Sadece ilk 100ms'yi gösterelim
    plt.show()

    # Frekans domeninde sinyalleri çiz (FFT)
    n_fft = 2048
    freqs = np.fft.rfftfreq(n_fft, 1/fs)
    
    original_fft = np.abs(np.fft.rfft(original_signal, n=n_fft))
    filtered_fft = np.abs(np.fft.rfft(filtered_signal, n=n_fft))

    plt.figure(figsize=(12, 6))
    plt.plot(freqs, 20*np.log10(original_fft), label='Orijinal Sinyal Spektrumu', alpha=0.7)
    plt.plot(freqs, 20*np.log10(filtered_fft), label='IRS Filtreli Sinyal Spektrumu', linewidth=2)
    plt.title('Frekans Domeninde Sinyaller (Spektrum)')
    plt.xlabel('Frekans (Hz)')
    plt.ylabel('Büyüklük (dB)')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, fs/2)
    plt.show()