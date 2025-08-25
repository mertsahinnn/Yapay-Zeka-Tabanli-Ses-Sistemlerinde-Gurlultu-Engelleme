import wandb # W&B kütüphanesi
import os # İşletim sistemi fonksiyonları için
import numpy as np # Sayısal işlemler için
import time # Zaman ölçümü için
from metric import compare # Metrik hesaplama fonksiyonu

def evaluate_and_log_metrics(clean_speech_path, output_dir, model_name):
    # Fonksiyon, temiz ses yolu, model çıkış yolu ve model adını parametre olarak alır.

    # W&B oturumunu başlatır
    wandb.init(
        project="dose-speech-enhancement", # W&B projesinin adı
        group= model_name, # İlgili çalıştırmaları gruplamak için model adı kullanılır
        job_type="evaluation", # Çalışmanın türü (değerlendirme)
        name = f"evaluation_run_on_{os.path.basename(output_dir)}", # Oturum için benzersiz bir ad oluşturur
        config={
            "clean_speech_path": clean_speech_path, # Temiz ses dosyalarının yolu
            "output_dir": output_dir # Modelin ürettiği dosyaların yolu
        }
    )

    t1 = time.time() # Değerlendirme başlangıç zamanı
    res = compare(clean_speech_path, output_dir) # Temiz ve gürültülü ses dosyalarını karşılaştırır ve metrikleri hesaplar
    t2 = time.time() # Değerlendirme bitiş zamanı

    # Hesaplanan metrikleri (res) bir numpy dizisine dönüştürür ve ortalamasını alır
    pm = np.array([x[0:] for x in res])
    pm = np.mean(pm, axis=0) # Tüm dosyaların metrik ortalamalarını hesaplar

    # Ortalaması alınmış metrikleri bir sözlük (dictionary) haline getirir
    metrics = {
        'csig': pm[0], # Sinyal kalitesi
        'cbak': pm[1], # Arka plan gürültü kalitesi
        'covl': pm[2], # Genel kalite
        'pesq': pm[3], # Perceptual Evaluation of Speech Quality (Konuşma Kalitesinin Algısal Değerlendirilmesi)
        'ssnr': pm[4], # Segmental Signal-to-Noise Ratio (Parçasal Sinyal-Gürültü Oranı)
        'stoi': pm[5]  # Short-Time Objective Intelligibility (Kısa Süreli Konuşma Anlaşılırlığı)
    }

    # W&B'ye metrikleri ve değerlendirme süresini loglar
    wandb.log(
        {
            "evaluation_time": '%.3f' % (t2 - t1), # Değerlendirme için geçen süre
            **metrics # "metrics" sözlüğündeki tüm metrikleri ana sözlüğe ekler (dictionary unpacking)
        }
    )

    # Değerlendirme sonuçlarını konsola yazdırır
    print('time: %.3f' % (t2 - t1))
    print('ref=', clean_speech_path)
    print('deg=', output_dir)
    print('csig:%6.4f cbak:%6.4f covl:%6.4f pesq:%6.4f ssnr:%6.4f stoi:%6.4f' % tuple(pm))

    
    # W&B oturumunu sonlandırır
    wandb.finish()