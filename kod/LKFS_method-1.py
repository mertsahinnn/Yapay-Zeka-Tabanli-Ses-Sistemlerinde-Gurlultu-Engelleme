import librosa
import soundfile as sf
import numpy as np
import pyloudnorm as pln
import random
import os

def add_noise_to_speech(speech_path, noise_path, output_path, target_snr_db_range=[-2.5, 17.5], sr=16000):
    """
    Belirtilen SNR aralığında rastgele konuşma dosyasına gürültü ekler.

    Args:
        speech_path (str): Temiz konuşma dosyasının yolu.
        noise_path (str): Gürültü dosyasının yolu.
        output_path (str): Gürültülü çıktının kaydedileceği yol.
        target_snr_db_range (list): Hedef SNR'nin rastgele seçileceği [min_db, max_db] aralığı.
        sr (int): Örnekleme hızı.
    """
    try:
        # 1. Ses dosyalarını yükle
        speech, sr_speech = librosa.load(speech_path, sr=sr)
        noise, sr_noise = librosa.load(noise_path, sr=sr)

        # Örnekleme hızlarının uyumluluğunu kontrol et
        # Eğer konuşma veya gürültü dosyası hedef örnekleme hızından (sr) farklıysa uyarı ver
        if sr_speech != sr or sr_noise != sr: 
            print(f"Uyarı: Örnekleme hızları farklı olabilir. Hedef: {sr}. Yeniden örnekleniyor.")
            

        # 2. Gürültü dosyasını konuşma dosyasıyla aynı uzunluğa getir
        if len(noise) < len(speech):
            # Gürültü kısa ise döngüye alarak tekrarla
            num_repeats = int(np.ceil(len(speech) / len(noise)))
            noise = np.tile(noise, num_repeats)[:len(speech)]
        elif len(noise) > len(speech):
            # Gürültü uzun ise rastgele bir yerden kes
            start_index = random.randint(0, len(noise) - len(speech))
            noise = noise[start_index : start_index + len(speech)]

        # 3. LKFS metre oluştur (ITU-R BS.1770-4 standardına göre)
        meter = pln.Meter(sr, block_size=0.400)

        # 4. Konuşma ve gürültü dosyalarının LKFS değerlerini hesapla
        # (-inf gibi hataları önlemek için sessiz sinyalleri kontrol et)
        if np.max(np.abs(speech)) < 1e-4 or np.max(np.abs(noise)) < 1e-4:
            print(f"Uyarı: '{os.path.basename(speech_path)}' veya '{os.path.basename(noise_path)}' çok sessiz. Bu çift atlanıyor.")
            return

        speech_loudness = meter.integrated_loudness(speech)
        noise_loudness = meter.integrated_loudness(noise)
        
        # 5. Rastgele bir hedef SNR seç
        target_snr_db = random.uniform(target_snr_db_range[0], target_snr_db_range[1])

        # 6. Gürültü sinyalini hedef SNR'ye göre ölçekle
        noise_gain_db = (speech_loudness - target_snr_db) - noise_loudness
        noise_gain_linear = 10**(noise_gain_db / 20.0)
        scaled_noise = noise * noise_gain_linear

        # 7. Konuşma ve ölçeklenmiş gürültüyü karıştır
        noisy_speech = speech + scaled_noise

        # 8. Normalizasyon (sesin kırpılmasını önlemek için)
        max_amplitude = np.max(np.abs(noisy_speech))
        if max_amplitude > 1.0:
            noisy_speech = noisy_speech / max_amplitude


        # 9. Gürültülü sesi kaydet
        sf.write(output_path, noisy_speech, sr)
        print(f"-> Başarılı: '{os.path.basename(output_path)}' oluşturuldu (Hedef SNR: {target_snr_db:.2f} dB)")

    except Exception as e:
        print(f"HATA: '{os.path.basename(speech_path)}' işlenirken bir sorun oluştu: {e}")


# --- YENİ KULLANIM YÖNTEMİ (KLASÖR BAZLI) ---
if __name__ == "__main__":

    
    # 1. Temiz konuşma dosyalarınızın bulunduğu klasör
    CLEAN_SPEECH_DIR = "C:\\D----------------\\Staj\\makale\\ses"

    # 2. Gürültü dosyalarınızın bulunduğu klasör
    NOISE_DIR = "C:\\D----------------\\Staj\\makale\\gurultu" # Örnek: ch01.wav bu klasörde olmalı

    # 3. Oluşturulacak gürültülü ses dosyalarının kaydedileceği klasör
    OUTPUT_DIR = "C:\\D----------------\\Staj\\makale\\LKFS_mix"
    # --- AYARLAR SONU ---

    # Çıktı klasörünün varlığını kontrol et, yoksa oluştur
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Çıktı klasörü: {OUTPUT_DIR}")

    # Ses ve gürültü dosyalarını listele (sadece .wav uzantılıları al)
    try:
        clean_files = [f for f in os.listdir(CLEAN_SPEECH_DIR) if f.lower().endswith('.wav')]
        noise_files = [f for f in os.listdir(NOISE_DIR) if f.lower().endswith('.wav')]
    except FileNotFoundError as e:
        print(f"HATA: Belirtilen klasörlerden biri bulunamadı. Lütfen yolları kontrol edin. Hata: {e}")
        exit()


    if not clean_files:
        print(f"HATA: Temiz konuşma klasöründe (.wav) dosyası bulunamadı: '{CLEAN_SPEECH_DIR}'")
        exit()
    if not noise_files:
        print(f"HATA: Gürültü klasöründe (.wav) dosyası bulunamadı: '{NOISE_DIR}'")
        exit()

    print(f"Toplam {len(clean_files)} temiz konuşma ve {len(noise_files)} gürültü dosyası bulundu.")
    print("-" * 50)

    # Her bir temiz konuşma dosyası için döngü başlat
    for speech_filename in clean_files:
        # Rastgele bir gürültü dosyası seç
        noise_filename = random.choice(noise_files)

        # Tam dosya yollarını oluştur
        speech_path_full = os.path.join(CLEAN_SPEECH_DIR, speech_filename)
        noise_path_full = os.path.join(NOISE_DIR, noise_filename)

        # Anlaşılır bir çıktı dosya adı oluştur
        speech_basename = os.path.splitext(speech_filename)[0]
        noise_basename = os.path.splitext(noise_filename)[0]
        output_filename = f"{speech_basename}_MIXEDWITH_{noise_basename}.wav"
        output_path_full = os.path.join(OUTPUT_DIR, output_filename)

        print(f"İşleniyor: '{speech_filename}' + '{noise_filename}'")

        # Gürültü ekleme fonksiyonunu çağır
        add_noise_to_speech(
            speech_path=speech_path_full,
            noise_path=noise_path_full,
            output_path=output_path_full,
            target_snr_db_range=[-10, 10],
            sr=16000
        )
        print("-" * 25)

    print("Tüm işlemler tamamlandı.")