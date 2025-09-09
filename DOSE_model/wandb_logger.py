from typing import Any, Dict, Optional
from pathlib import Path 

import wandb # W&B kütüphanesi
import os # İşletim sistemi fonksiyonları için
import numpy as np # Sayısal işlemler için
import time # Zaman ölçümü için
from metric import compare # Metrik hesaplama fonksiyonu

class WandBTrainingLogger:
    '''Egitim surecindeki wandb loglama islemlerini yapar'''
    
    def __init__(self, project_name: str = "Ai_based_Noise_Cancellation"):
        
        self.project_name = project_name
        self.run = None
        
    def init_run(self, run_name: str, config: Dict[str, Any], job_type: str = "train"):
        
        
        try:
            self.run = wandb.init(
                project=self.project_name,
                name=run_name,
                config=config,
                job_type=job_type
            )
            print(f"W&B run initialized: {run_name}")

        except Exception as e:
            print(f"Error initializing W&B run: {e}")
            raise
    
    def get_config(self):
        if self.run is not None:
            return self.run.config
        else:
            raise ValueError("W&B run is not initialized. Call init_run() first.")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, prefix: str = ""):
        
        try:
            log_dict = {}
            for key, value in metrics.items():
                log_key = f"{prefix}/{key}" if prefix else key
                log_dict[log_key] = value
            
            wandb.log(log_dict, step=step)
        except Exception as e:
            print(f"Error logging metrics to W&B: {e}")
            
    def log_validation_results(self, avg_metrics : Dict[str, float], avg_loss: float, epoch: int):
        
        log_dict ={
            "val/loss": avg_loss,
            "val/pesq": avg_metrics['pesq'],
            "val/stoi": avg_metrics['stoi'],
            "val/ssnr": avg_metrics['ssnr'],
            "val/csig": avg_metrics['csig'],
            "val/cbak": avg_metrics['cbak'],
            "val/covl": avg_metrics['covl'],
        }
        
        self.log_metrics(log_dict, step=epoch)
        
    def log_training_epoch(self,  train_loss: float, epoch: int):

        log_dict = {
            "train/loss": train_loss,
        }

        self.log_metrics(log_dict, step=epoch)
        
    def log_epoch_summary(self, train_loss: float, val_loss: Optional[float], epoch: int):
        
        log_dict = {
            "loss/train" : train_loss
        }
        
        if val_loss is not None:
            log_dict["loss/val"] = val_loss
            
        self.log_metrics(log_dict, step=epoch)
        
    def save_model_artifact(self, save_path: Path, filename: str, epoch: Optional[int] = None, metadata: Optional[Dict] = None):
        
        try:
            artifact_name = f"model-{filename}"
            if epoch is not None:
                artifact_name += f"_{epoch}"
            
            artifact_metadata = {
                "epoch": epoch,
                "model": filename
            }
            
            if metadata:
                artifact_metadata.update(metadata)
            
            artifact = wandb.Artifact(
                name = artifact_name,
                type = "model",
                metadata = artifact_metadata,
                description = f"Model checkpoint at epoch {epoch}" if epoch is not None else ""
            )
            
            artifact.add_file(save_path)
            wandb.log_artifact(artifact)
            print(f"Model artifact logged: {artifact_name}")
            
        except Exception as e:
            print(f"Error saving model artifact to W&B: {e}")
    
    def finish(self):
        
        try:
            wandb.finish()
            print("W&B run finished.")
        except Exception as e:
            print(f"Error finishing W&B run: {e}")

def create_training_logger(project_name: str = "dose-speech-enhancement") -> WandBTrainingLogger:
    return WandBTrainingLogger(project_name=project_name)


def evaluate_and_log_metrics(clean_speech_path, output_dir, model_name):
    # Fonksiyon, temiz ses yolu, model çıkış yolu ve model adını parametre olarak alır.

    try:
        wandb.init(
            project="Ai_based_Noise_Cancellation", # W&B projesinin adı
            group= model_name, # İlgili çalıştırmaları gruplamak için model adı kullanılır
            job_type="evaluation", # Çalışmanın türü (değerlendirme)
            name = f"evaluation_run_on_{os.path.basename(output_dir)}", # Oturum için benzersiz bir ad oluşturur
            config={
                "clean_speech_path": clean_speech_path, # Temiz ses dosyalarının yolu
                "output_dir": output_dir # Modelin ürettiği dosyaların yolu
            }
        )
        
        # Klasorün varlığını kontrol et
        if not os.path.exists(clean_speech_path):
            raise FileNotFoundError(f"Clean speech path does not exist: {clean_speech_path}")
        if not os.path.exists(output_dir):
            raise FileNotFoundError(f"Output directory does not exist: {output_dir}")
        
        t1 = time.time() # Değerlendirme başlangıç zamanı
        res = compare(clean_speech_path, output_dir) # Temiz ve gürültülü ses dosyalarını karşılaştırır ve metrikleri hesaplar
        
        if not res or len(res) == 0:
            print("No results returned from compare function.")
            return
        
        t2 = time.time() # Değerlendirme bitiş zamanı

        # Hesaplanan metrikleri (res) bir numpy dizisine dönüştürür ve ortalamasını alır
        pm = np.array([x[0:] for x in res])
        pm = np.mean(pm, axis=0) # Tüm dosyaların metrik ortalamalarını hesaplar
        
        if len(pm) < 6:
            print(f"Unexpected number of metrics returned: {len(pm)}. Expected at least 6.")
            return
        
        metrics = {
            'test/csig': pm[0], # Sinyal kalitesi
            'test/cbak': pm[1], # Arka plan gürültü kalitesi
            'test/covl': pm[2], # Genel kalite
            'test/pesq': pm[3], # Perceptual Evaluation of Speech Quality (Konuşma Kalitesinin Algısal Değerlendirilmesi)
            'test/ssnr': pm[4], # Segmental Signal-to-Noise Ratio (Parçasal Sinyal-Gürültü Oranı)
            'test/stoi': pm[5]  # Short-Time Objective Intelligibility (Kısa Süreli Konuşma Anlaşılırlığı)
        }
        
        wandb.log(
            {
                "evaluation_time": t2 - t1, # Değerlendirme için geçen süre
                **metrics # "metrics" sözlüğündeki tüm metrikleri ana sözlüğe ekler (dictionary unpacking)
            }
        )
        
        # Sonuçları konsola yazdır
        print(f'Time: {t2 - t1:.3f} seconds')
        print(f'Reference: {clean_speech_path}')
        print(f'Degraded: {output_dir}')
        print('csig:%6.4f cbak:%6.4f covl:%6.4f pesq:%6.4f ssnr:%6.4f stoi:%6.4f' % tuple(pm))
        
    except Exception as e:
        print(f"Error during evaluation and logging: {e}")
    finally:
        try:
            wandb.finish()
            print("W&B run finished.")
        except Exception as e:
            print(f"Error finishing W&B run: {e}")

    
    
    
    
    
    
    
    
    
    
    
    
    
    """
    # W&B oturumunu başlatır
    wandb.init(
        project="Ai_based_Noise_Cancellation", # W&B projesinin adı
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
        'test/csig': pm[0], # Sinyal kalitesi
        'test/cbak': pm[1], # Arka plan gürültü kalitesi
        'test/covl': pm[2], # Genel kalite
        'test/pesq': pm[3], # Perceptual Evaluation of Speech Quality (Konuşma Kalitesinin Algısal Değerlendirilmesi)
        'test/ssnr': pm[4], # Segmental Signal-to-Noise Ratio (Parçasal Sinyal-Gürültü Oranı)
        'test/stoi': pm[5]  # Short-Time Objective Intelligibility (Kısa Süreli Konuşma Anlaşılırlığı)
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
    """