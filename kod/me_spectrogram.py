import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio

def audio_to_spectrogram(audio_path, params = None):
    """ Ses dosyasini mel-spektrograma donusturme"""
    
    # Params
    sr = 16000
    n_fft = 320
    hop_length = 160
    n_mels = 80
    
    # Ses dosyasini yukle
    
    audio, _ = librosa.load(audio_path, sr=sr)
    
    # Mel-spektrograma hesapla
    mel_spec = librosa.feature.melspectrogram(
        y = audio,
        sr = sr,
        n_fft = n_fft,
        hop_length = hop_length,
        n_mels = n_mels,
        fmin = 0, # min frekans
        fmax = sr // 2 # max frekans
    )
    
    # dB scale'e donustur
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec, mel_spec_db, audio, sr

def plot_mel_spec(mel_spec_db, sr, hop_length, title = "Mel-Spektrogram"):
    """" Mel-spektrogrami gorsellestirme """
    
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(mel_spec_db,
                             sr=sr,
                             hop_length=hop_length,
                             x_axis='time',
                             y_axis='mel',
                             cmap='viridis'
                             )
    
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Zaman (s)')
    plt.ylabel('Mel Frekans')
    plt.tight_layout()
    plt.show()
    

def main():
    audio_path = 'C:\\D----------------\\Staj\\makale\\Veri seti ornekleri\\LKFS_mix\\1.wav'  # Ses dosyasinin yolu
    
    try:
        mel_spec, mel_spec_db, audio, sr = audio_to_spectrogram(audio_path)
        print(f"Ses boyutu: {audio.shape}")
        print(f"Mel-spektrogram boyutu: {mel_spec_db.shape}")
        print(f"Örnekleme frekansı: {sr} Hz")
        print(f"Ses uzunluğu: {len(audio)/sr:.2f} saniye")
        
        plot_mel_spec(mel_spec_db, sr, hop_length=256)  # hop_length parametresini düzelttim
        
    except FileNotFoundError:
        print(f"Ses dosyası bulunamadı: {audio_path}")
        print("Lütfen dosya yolunu kontrol edin.")
    except Exception as e:
        print(f"Hata oluştu: {e}")

# main() fonksiyonunu çağır
if __name__ == "__main__":
    main()