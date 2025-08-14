import librosa
import soundfile as sf
import numpy as np
import pyloudnorm as pln
import random
import os
from collections import Counter # Counter sınıfını import ediyoruz

def add_noise_to_speech(speech_path, noise_path, output_path, target_snr_db_range=[-2.5, 17.5], sr=16000):
    """
    Belirtilen SNR aralığında rastgele konuşma dosyasına gürültü ekler.

    Args:
        speech_path (str): Temiz konuşma dosyasının yolu.
        noise_path (str): Gürültü dosyasının yolu.
        output_path (str): Gürültülü çıktının kaydedileceği yol.
        target_snr_db_range (list): Hedef SNR'nin rastgele seçileceği [min_db, max_db] aralığı.
        sr (int): Örnekleme hızı.
    Returns:
        tuple: (bool success, float actual_snr_db if success else None)
               İşlemin başarılı olup olmadığı ve başarılıysa kullanılan SNR değeri.
    """
    try:
        # 1. Ses dosyalarını yükle
        speech, sr_speech = librosa.load(speech_path, sr=sr)
        noise, sr_noise = librosa.load(noise_path, sr=sr)

        # Örnekleme hızlarının uyumluluğunu kontrol et
        if sr_speech != sr or sr_noise != sr: 
            print(f"Uyarı: Örnekleme hızları farklı olabilir. Hedef: {sr}. Yeniden örnekleniyor olabilir.")
            

        # 2. Gürültü dosyasını konuşma dosyasıyla aynı uzunluğa getir
        if len(noise) < len(speech):
            num_repeats = int(np.ceil(len(speech) / len(noise)))
            noise = np.tile(noise, num_repeats)[:len(speech)]
        elif len(noise) > len(speech):
            start_index = random.randint(0, len(noise) - len(speech))
            noise = noise[start_index : start_index + len(speech)]

        # 3. LKFS metre oluştur (ITU-R BS.1770-4 standardına göre)
        meter = pln.Meter(sr, block_size=0.400)

        # 4. Konuşma ve gürültü dosyalarının LKFS değerlerini hesapla
        if np.max(np.abs(speech)) < 1e-4 or np.max(np.abs(noise)) < 1e-4:
            print(f"Uyarı: '{os.path.basename(speech_path)}' veya '{os.path.basename(noise_path)}' çok sessiz. Bu çift atlanıyor.")
            return False, None # İşlem başarısız olursa False ve None döndür
        
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
        return True, target_snr_db # İşlem başarılı olursa True ve kullanılan SNR'yi döndür

    except Exception as e:
        print(f"HATA: '{os.path.basename(speech_path)}' işlenirken bir sorun oluştu: {e}")
        return False, None # İşlem başarısız olursa False ve None döndür


if __name__ == "__main__":
    
    # 1. Temiz konuşma dosyalarınızın bulunduğu klasör
    CLEAN_SPEECH_DIR = "/root/.cache/kagglehub/datasets/truthisneverlinear/turkish-speech-corpus/versions/1/ISSAI_TSC_218/"

    # 2. Gürültü dosyalarınızın bulunduğu klasör
    NOISE_DIR = "/root/.cache/kagglehub/datasets/chrisfilo/demand/versions/1/" 

    # 3. Oluşturulacak gürültülü ses dosyalarının kaydedileceği klasör
    OUTPUT_DIR = "/content/LKFS_mix/"
    
    # Çıktı klasörünün varlığını kontrol et, yoksa oluştur
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Çıktı klasörü: {OUTPUT_DIR}")

    # Ses ve gürültü dosyalarını listele (sadece .wav uzantılıları al)
    try:
        clean_files = [f for f in os.listdir(CLEAN_SPEECH_DIR) if f.lower().endswith('.wav')]
        
        # Gürültü dosyalarını alt klasörleriyle birlikte listele
        noise_files_full_paths = []
        for root, _, files in os.walk(NOISE_DIR):
            for f in files:
                if f.lower().endswith('.wav'):
                    noise_files_full_paths.append(os.path.join(root, f))
        
    except FileNotFoundError as e:
        print(f"HATA: Belirtilen klasörlerden biri bulunamadı. Lütfen yolları kontrol edin. Hata: {e}")
        exit()

    if not clean_files:
        print(f"HATA: Temiz konuşma klasöründe (.wav) dosyası bulunamadı: '{CLEAN_SPEECH_DIR}'")
        exit()
    if not noise_files_full_paths:
        print(f"HATA: Gürültü klasöründe veya alt klasörlerinde (.wav) dosyası bulunamadı: '{NOISE_DIR}'")
        exit()

    print(f"Toplam {len(clean_files)} temiz konuşma ve {len(noise_files_full_paths)} gürültü dosyası bulundu.")
    print("-" * 50)

    # Gürültü dosyalarının kullanım sayısını takip etmek için bir sayaç
    noise_usage_counter = Counter()
    # Tüm SNR değerlerini depolamak için bir liste
    all_snr_values = [] 

    processed_files_count = 0
    skipped_files_count = 0

    # Her bir temiz konuşma dosyası için döngü başlat
    for speech_filename in clean_files:
        # Rastgele bir gürültü dosyasının tam yolunu seç
        noise_path_full = random.choice(noise_files_full_paths)

        # Tam konuşma dosya yolunu oluştur
        speech_path_full = os.path.join(CLEAN_SPEECH_DIR, speech_filename)
        
        # Çıktı dosya adı için konuşma dosyasının adını ve gürültü dosyasının üst klasör ve dosya adını al
        speech_basename = os.path.splitext(speech_filename)[0]
        
        # Gürültü dosyasının üst klasör adını al (e.g., 'araba_gurultulari')
        parent_folder_name = os.path.basename(os.path.dirname(noise_path_full))
        # Gürültü dosyasının kendi adını al (e.g., 'araba_1.wav')
        noise_file_only_name = os.path.basename(noise_path_full)

        # Çıktı adı için "üst_klasör_adı_dosya_adı.wav" formatını kullan
        clean_noise_name_for_output = f"{parent_folder_name}_{os.path.splitext(noise_file_only_name)[0]}"

        output_filename = f"{speech_basename}.wav"
        output_path_full = os.path.join(OUTPUT_DIR, output_filename)

        # Konsola daha anlamlı yazdır (parent folder name + file name)
        print(f"İşleniyor: '{speech_filename}' + '{parent_folder_name}{os.sep}{noise_file_only_name}'")

        # Gürültü ekleme fonksiyonunu çağır
        success, used_snr = add_noise_to_speech( # İki değer döndürüyor
            speech_path=speech_path_full,
            noise_path=noise_path_full, 
            output_path=output_path_full,
            target_snr_db_range=[-10, 10], # SNR aralığınızı buraya tanımlayın
            sr=16000
        )
        
        if success:
            # Sayaçta "parent_folder_name/noise_file_name.wav" formatını kullanıyoruz
            noise_usage_key = f"{parent_folder_name}{os.sep}{noise_file_only_name}"
            noise_usage_counter[noise_usage_key] += 1 
            processed_files_count += 1
            if used_snr is not None: # SNR değeri döndüyse listeye ekle
                all_snr_values.append(used_snr)
        else:
            skipped_files_count += 1 

        print("-" * 25)

    print("Tüm işlemler tamamlandı. 🎉")
    print("\n" + "=" * 50)
    print("İşlem Özeti:")
    print("=" * 50)
    print(f"Toplam İşlenen Konuşma Dosyası: {processed_files_count}")
    print(f"Atlanan Dosya Çifti (sessiz veya hatalı): {skipped_files_count}")
    
    print("\n--- Gürültü Dosyası Kullanım İstatistikleri ---")
    if noise_usage_counter:
        sorted_noise_usage = sorted(noise_usage_counter.items(), key=lambda item: item[1], reverse=True)
        for noise_key, count in sorted_noise_usage:
            print(f"- '{noise_key}': {count} kez kullanıldı.")
    else:
        print("Hiçbir gürültü dosyası başarıyla kullanılmadı.")

    print("\n--- SNR İstatistikleri ---")
    if all_snr_values:
        min_snr = np.min(all_snr_values)
        max_snr = np.max(all_snr_values)
        avg_snr = np.mean(all_snr_values)
        std_snr = np.std(all_snr_values)
        
        print(f"Minimum SNR: {min_snr:.2f} dB")
        print(f"Maksimum SNR: {max_snr:.2f} dB")
        print(f"Ortalama SNR: {avg_snr:.2f} dB")
        print(f"SNR Standart Sapma: {std_snr:.2f} dB")

    else:
        print("Hiçbir SNR değeri kaydedilmedi.")
    print("=" * 50)