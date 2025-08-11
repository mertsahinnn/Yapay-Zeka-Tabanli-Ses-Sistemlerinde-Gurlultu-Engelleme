import numpy as np
from scipy.io import wavfile
import random
import os
from P56_method_1 import active_speech_level
from audio_utils import read_wav_mono

def mix_at_snr(speech, noise, fs, snr_db, speech_info=None, use_active_speech=True):
    """Bir konuşma ve gürültü sinyalini hedeflenen SNR'da karıştırır."""
    N = len(speech)
    
    # 1. Gürültü sinyalinin uzunluğunu konuşma ile eşleştir
    if len(noise) < N:
        repeats = int(np.ceil(N / len(noise)))
        noise = np.tile(noise, repeats)[:N]
    elif len(noise) > N:
        start = random.randint(0, len(noise) - N)
        noise = noise[start:start+N]

    # 2. Konuşma gücünü (Ps) hesapla
    if use_active_speech and speech_info is not None and "ActLev_sample_units" in speech_info:
        Ps = speech_info["ActLev_sample_units"] ** 2  # RMS^2 = güç
    else:
        Ps = np.mean(speech**2)

    # 3. Gürültü gücünü (Pn) hesapla ve ayarla
    Pn0 = np.mean(noise**2)
    if Pn0 <= 0:
        raise ValueError("Noise signal has zero power")

    # Hedef gürültü gücü
    factor = 10 ** (snr_db / 10.0)
    Pn_desired = Ps / factor
    alpha = np.sqrt(Pn_desired / Pn0)

    # 4. Sinyalleri karıştır
    mixed = speech + (alpha * noise)

    # 5. Kırpılmayı (clipping) önle
    max_abs = np.max(np.abs(mixed))
    if max_abs > 1.0:
        mixed = mixed / max_abs

    return mixed, alpha

def main():
    """Belirtilen klasörlerdeki tüm konuşma ve gürültü dosyalarını,
    belirtilen SNR seviyelerinde toplu olarak karıştırır."""
    # --- Ayarlar: Bu klasör yollarını ve SNR listesini düzenleyin ---
    SPEECH_DIR = "C:\\D----------------\\Staj\\makale\\ses"
    NOISE_DIR = "C:\\D----------------\\Staj\\makale\\gurultu" # Gürültü dosyalarınızın olduğu klasör
    OUTPUT_DIR = "C:\\D----------------\\Staj\\makale\\karisik_sesler"
    TARGET_SNRS = [0, 5, 10, 15]  # dB cinsinden SNR değerleri listesi
    # ----------------------------------------------------------------

    # Giriş klasörlerindeki tüm .wav dosyalarını bul
    try:
        speech_files = [os.path.join(SPEECH_DIR, f) for f in os.listdir(SPEECH_DIR) if f.lower().endswith('.wav')]
        noise_files = [os.path.join(NOISE_DIR, f) for f in os.listdir(NOISE_DIR) if f.lower().endswith('.wav')]
    except FileNotFoundError as e:
        print(f"Hata: Klasör bulunamadı - {e.filename}")
        return

    if not speech_files or not noise_files:
        print("Hata: Konuşma veya gürültü klasörlerinden biri boş veya .wav dosyası içermiyor.")
        return

    total_jobs = len(speech_files) * len(noise_files) * len(TARGET_SNRS)
    print(f"Toplam {total_jobs} karıştırma işlemi gerçekleştirilecek...")
    job_count = 0

    # Her bir konuşma dosyasını, her bir gürültü dosyası ve SNR değeri ile karıştır
    for speech_path in speech_files:
        fs_s, speech = read_wav_mono(speech_path)
        _, _, speech_info = active_speech_level(speech, fs_s)

        for noise_path in noise_files:
            fs_n, noise = read_wav_mono(noise_path)

            if fs_s != fs_n:
                print(f"Uyarı: Örnekleme frekansları uyuşmuyor. Atlanıyor: {os.path.basename(speech_path)} ({fs_s}Hz) & {os.path.basename(noise_path)} ({fs_n}Hz)")
                continue

            for snr in TARGET_SNRS:
                job_count += 1
                print(f"[{job_count}/{total_jobs}] İşleniyor: {os.path.basename(speech_path)} + {os.path.basename(noise_path)} @ {snr}dB")

                mixed, _ = mix_at_snr(speech, noise, fs_s, snr_db=snr, speech_info=speech_info, use_active_speech=True)

                # Dinamik çıktı yolu ve adı oluştur
                speech_basename = os.path.splitext(os.path.basename(speech_path))[0]
                noise_basename = os.path.splitext(os.path.basename(noise_path))[0]
                snr_output_dir = os.path.join(OUTPUT_DIR, f"{snr}dB")
                os.makedirs(snr_output_dir, exist_ok=True)
                output_filename = f"{speech_basename}__{noise_basename}__{snr}dB.wav"
                output_path = os.path.join(snr_output_dir, output_filename)

                wavfile.write(output_path, fs_s, (mixed * 32767).astype(np.int16))

    print(f"\n--- Tüm {total_jobs} işlem tamamlandı. Çıktılar '{OUTPUT_DIR}' klasörüne kaydedildi. ---")

if __name__ == "__main__":
    main()