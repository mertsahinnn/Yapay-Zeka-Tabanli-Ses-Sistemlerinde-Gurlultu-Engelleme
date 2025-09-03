# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Gerekli kutuphanleri ice aktar
from typing import Optional, Dict, List, Tuple
import time
from pathlib import Path
import numpy as np
import os
import torch
import torch.nn as nn
import wandb # Log icin W&B kutuphanesi

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import from_path
from model import DOSE
from params import AttrDict

from metric import compare
from metric import composite


# # İç içe geçmiş veri yapılarını (listeler, sözlükler vb.) bir fonksiyonla eşlemek için yardımcı fonksiyon.
def _nested_map(struct, map_fn):
  if isinstance(struct, tuple):
    return tuple(_nested_map(x, map_fn) for x in struct)
  if isinstance(struct, list):
    return [_nested_map(x, map_fn) for x in struct]
  if isinstance(struct, dict):
    return { k: _nested_map(v, map_fn) for k, v in struct.items() }
  return map_fn(struct)

class EMAHelper:
  """Exponential Moving Average yardimc sinif"""
  
  def __init__(self, model, decay=0.9999):
     self.decay = decay
     self.shadow = {}
     self.backup = {}
     
     # Model parametrelerinin shadow kopyasini olustur
     for name, param in model.named_parameters():
         if param.requires_grad:
             self.shadow[name] = param.data.clone()
             
  
  def update(self, model):
    """EMA parametrelerini guncelle"""
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert name in self.shadow
            new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
            self.shadow[name] = new_average.clone()
            
  
  def apply_shadow(self, model):
    """Model parametrelerini EMA parametreleriyle degistir"""
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert name in self.shadow
            self.backup[name] = param.data.clone()
            param.data = self.shadow[name]
            
  def restore(self, model):
    """Orijinal parametreleri geri yukle"""
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert name in self.backup
            param.data = self.backup[name]
    self.backup.clear()
    
  def state_dict(self):
    """EMA state'ini kaydet"""
    return {'shadow': self.shadow, 'decay': self.decay}


  def load_state_dict(self, state_dict):
    """EMA state'ini yukle"""
    self.shadow = state_dict['shadow']
    self.decay = state_dict['decay']
    

# DOSE modelini egitmek icin ana sinif
class DOSELearner:
  def __init__(self, model_dir, model, dataset, optimizer, params, *args, **kwargs):

    # Modelin kaydedileceği dizini oluştur
    os.makedirs(model_dir, exist_ok=True)
    self.model_dir = model_dir

    # Model and training components
    self.model = model
    self.dataset = dataset
    self.optimizer = optimizer
    self.params = params

    # Mixed precision eğitimi icin autocast ve scaler ayarları
    try:
      #pyTorch >= 1.10 icin yeni APi
      self.autocast = torch.amp.autocast('cuda', enabled=kwargs.get('fp16', True))
      self.scaler = torch.amp.GradScaler('cuda', enabled=kwargs.get('fp16', True))
    except AttributeError:
      # Eski pyTorch veriyonlari icin fallback
      self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get('fp16', True))
      self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get('fp16', True))
      
    self.step = 0
    self.is_master = True

    # Gürültü programını hesaplar
    beta = np.array(self.params.noise_schedule)
    noise_level = np.cumprod(1 - beta)
    self.noise_level = torch.tensor(noise_level.astype(np.float32))
    self.loss_fn = nn.L1Loss() # L1(MAE) kaybı fonksiyonunu kullanır
    self.summary_writer = None
    self.dropout = params.dropout_rate
    
    # EMA helper'i baslat
    self.use_ema = kwargs.get('use_ema', True)
    if self.use_ema:
        ema_decay = kwargs.get('ema_decay', 0.9999)
        self.ema_helper = EMAHelper(model, decay=ema_decay)
        print(f"🔄 EMA aktif edildi (decay={ema_decay})")
    else:
        self.ema_helper = None

  # Modelin ve optimizer'in durumunu kaydetmek icin sozluk dondurur
  def state_dict(self):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      model_state = self.model.module.state_dict()
    else:
      model_state = self.model.state_dict()
    state = {
        'step': self.step,
        'model': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items() },
        'optimizer': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items() },
        'params': dict(self.params),
        'scaler': self.scaler.state_dict(),
    }
    
    # EMA state'ini kaydet
    if self.ema_helper is not None:
        state['ema'] = self.ema_helper.state_dict()
        
    return state

  # Kaydedilmis durumu yukler
  def load_state_dict(self, state_dict):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      self.model.module.load_state_dict(state_dict['model'])
    else:
      self.model.load_state_dict(state_dict['model'])
      
    self.optimizer.load_state_dict(state_dict['optimizer'])
    self.scaler.load_state_dict(state_dict['scaler'])
    self.step = state_dict['step']
    
    # EMA state'ini yukle (geriye uyumluluk icin kontrol)
    if self.ema_helper is not None and 'ema' in state_dict:
      self.ema_helper.load_state_dict(state_dict['ema'])

  # Kontrol noktasini dosyaya kaydeder
  def save_to_checkpoint(self, filename='weights', epoch: Optional[int] = None):
    try:
      
      # Dosya yollarını oluştur
      model_path = Path(self.model_dir)
      model_path.mkdir(parents=True, exist_ok=True)

      # Dosya adlarını oluştur
      if epoch is not None:
          save_basename = f'{filename}-{epoch}.pt'
      else:
          save_basename = f'{filename}.pt'

      save_path = model_path / save_basename
      link_path = model_path / f'{filename}.pt'


      # Model state'ini kaydet
      state_dict = self.state_dict()
      torch.save(state_dict, save_path)

      print(f"Model kaydedildi: {save_path}")

      # Wandb artifact'ı oluştur ve kaydet
      self._save_wandb_artifact(save_path, filename, epoch)
      
      # Platform bazli link olusturma  
      if os.name == 'nt': # Windows
        torch.save(self.state_dict(), link_path)
      else:
        if os.path.islink(link_path):
          os.unlink(link_path)
        os.symlink(save_basename, link_path)

    except Exception as e:
      print(f"Model kaydetme hatasi: {e}")

    """
    save_basename = f'{filename}-{epoch}.pt' # epoch sayisi ile kaydet
    save_name = f'{self.model_dir}/{save_basename}'
    link_name = f'{self.model_dir}/{filename}.pt'
    torch.save(self.state_dict(), save_name)

    # Wandb artifact olusturma ve kaydetme
    artifact = wandb.Artifact(
      name = f"model-{filename}",
      type = "model",
      description = f"Model weights at epoch {epoch}",
      metadata = {
          "epoch": epoch,
          "model": filename
      }
    )
    artifact.add_file(save_name)
    wandb.log_artifact(artifact)
    """

  def _save_wandb_artifact(self, save_path: Path, filename: str, epoch: Optional[int]):
    # Wandb artifact'i olusturur ve kaydeder.
    try:
      artifact_name = f"model-{filename}"
      if epoch is not None:
        artifact_name+= f"_{epoch}"
        
      artifact = wandb.Artifact(
        name = artifact_name,
        type = "model",
        description = f"Model weights{f'at epoch {epoch}' if epoch is not None else ''}",
        metadata = {
            "epoch": epoch,
            "model": filename,
        }
      )

      artifact.add_file(save_path)
      wandb.log_artifact(artifact)

      print(f"Wandb artifact kaydedildi: {artifact_name}")

    except Exception as e:
      print(f"Wandb artifact kaydedilemedi: {e}")

  # Dosyadan kontrol noktasini yukler
  def restore_from_checkpoint(self, filename='weights'):
    try:
      checkpoint = torch.load(f'{self.model_dir}/{filename}.pt')
      self.load_state_dict(checkpoint)
      return True
    except FileNotFoundError:
      return False
    
    
  
  def validate(self, val_dataset, epoch: int) -> float:
    """
    Validation dataset üzerinde modelin performansını değerlendirir.
    
    Args:
        val_dataset: Validation data loader
        epoch: Mevcut epoch numarası
        
    Returns:
        float: Ortalama validation loss
    """
    # EMA parametrelerini uygula
    if self.ema_helper is not None:
      self.ema_helper.apply_shadow(self.model)
    try:
      self.model.eval()
      device = next(self.model.parameters()).device

      # Metrics initialization
      total_loss = 0.0
      total_metrics = {
          'csig': 0.0, 'cbak': 0.0, 'covl': 0.0, 
          'pesq': 0.0, 'ssnr': 0.0, 'stoi': 0.0
      }
      count = 0
      batch_metrics_list = []

      # Validation başlangıç zamanı
      val_start_time = time.time()

      try:
          with torch.no_grad():
              # Progress bar ile validation
              val_iterator = tqdm(val_dataset, desc=f'Validation Epoch {epoch}', 
                                leave=False, disable=not self.is_master)

              for batch_idx, features in enumerate(val_iterator):
                  try:
                      # Verileri device'a taşı
                      features = _nested_map(features, 
                                           lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)

                      # Batch validation
                      batch_loss, batch_metrics = self._validate_batch(features)

                      if batch_loss is not None and batch_metrics is not None:
                          total_loss += batch_loss
                          batch_metrics_list.append(batch_metrics)

                          # Accumulate metrics
                          for key in total_metrics.keys():
                              total_metrics[key] += batch_metrics.get(key, 0.0)

                          count += 1

                          # Progress bar güncelleme
                          if self.is_master:
                              val_iterator.set_postfix({
                                  'Loss': f'{batch_loss:.4f}',
                                  'PESQ': f'{batch_metrics.get("pesq", 0):.3f}',
                                  'STOI': f'{batch_metrics.get("stoi", 0):.3f}'
                              })

                  except Exception as e:
                      print(f"Validation batch {batch_idx} hatası: {e}")
                      continue
                    
      except Exception as e:
          print(f"Validation sırasında kritik hata: {e}")
          return float('inf')

      # Sonuçları hesapla
      avg_loss, avg_metrics, metric_std = self._compute_validation_results(
          total_loss, total_metrics, batch_metrics_list, count
      )

      # Validation süresini hesapla
      val_duration = time.time() - val_start_time

      if self.is_master:
        ema_tag = " (EMA)" if self.ema_helper is not None else ""
        print(f"🔍 Val Loss{ema_tag}: {avg_loss:.4f} | PESQ: {avg_metrics['pesq']:.3f} | STOI: {avg_metrics['stoi']:.3f}")

      # WandB'a logla
      self._log_to_wandb(avg_metrics, avg_loss, metric_std, epoch)

      return avg_loss
    
    finally:
      # Orijinal parametreleri geri yukle
      if self.ema_helper is not None:
        self.ema_helper.restore(self.model)
        

  def _validate_batch(self, features: Dict) -> Tuple[Optional[float], Optional[Dict]]:
    """
    Tek bir batch için validation yapar.
    
    Args:
        features: Batch verileri
        
    Returns:
        Tuple[loss, metrics]: Batch loss ve metrikleri
    """
    try:
        audio = features['clean_speech']
        noisy = features['noisy_speech']
        N, T = audio.shape
        
        # Validation için t=0 kullan (direkt denoising)
        t = torch.zeros(N, dtype=torch.long, device=audio.device)
        
        # Model inference
        predicted = self.model(noisy, t, noisy)
        loss = self.loss_fn(audio, predicted.squeeze(1))
        
        # Tüm batch için metrikleri hesapla (performans için ilk 4 sample)
        batch_metrics = self._compute_batch_metrics(
            audio, predicted, min(N, 4)
        )
        
        return loss.item(), batch_metrics
        
    except Exception as e:
        print(f"Batch validation hatası: {e}")
        return None, None

  def _compute_batch_metrics(self, clean_audio: torch.Tensor, 
                         predicted_audio: torch.Tensor, 
                         num_samples: int = 4) -> Dict[str, float]:
    """
    Batch içindeki multiple sample için metrikleri hesaplar.
    
    Args:
        clean_audio: Temiz ses [batch_size, length]
        predicted_audio: Tahmin edilen ses [batch_size, 1, length]
        num_samples: Hesaplanacak sample sayısı
        
    Returns:
        Dict: Ortalama metrikler
    """
    metrics_sum = {
        'csig': 0.0, 'cbak': 0.0, 'covl': 0.0,
        'pesq': 0.0, 'ssnr': 0.0, 'stoi': 0.0
    }
    valid_samples = 0
    
    for i in range(min(num_samples, clean_audio.shape[0])):
        try:
            # Tensor'ları numpy'a çevir
            clean_np = clean_audio[i].cpu().numpy()
            predicted_np = predicted_audio[i].squeeze().cpu().numpy()
            
            # Uzunluk eşitleme
            min_len = min(len(clean_np), len(predicted_np))
            if min_len < 1000:  # Çok kısa sample'ları atla
                continue
                
            clean_np = clean_np[:min_len]
            predicted_np = predicted_np[:min_len]
            
            # Metrik hesaplama
            res = composite(clean_np, predicted_np, self.params.sample_rate)
            
            if len(res) >= 6:  # Geçerli sonuç kontrolü
                metrics_sum['ssnr'] += res[0]
                metrics_sum['pesq'] += res[1] 
                metrics_sum['csig'] += res[2]
                metrics_sum['cbak'] += res[3]
                metrics_sum['covl'] += res[4]
                metrics_sum['stoi'] += res[5]
                valid_samples += 1
                
        except Exception as e:
            print(f"Sample {i} metrik hesaplama hatası: {e}")
            continue
    
    # Ortalama hesapla
    if valid_samples > 0:
        return {k: v / valid_samples for k, v in metrics_sum.items()}
    else:
        return {k: 0.0 for k in metrics_sum.keys()}

  def _compute_validation_results(self, total_loss: float, total_metrics: Dict, 
                              batch_metrics_list: List[Dict], 
                              count: int) -> Tuple[float, Dict, Dict]:
    """
    Validation sonuçlarını hesaplar (ortalama ve standart sapma).
    
    Returns:
        Tuple[avg_loss, avg_metrics, std_metrics]
    """
    if count == 0:
        return float('inf'), {k: 0.0 for k in total_metrics.keys()}, {}
    
    # Ortalama loss
    avg_loss = total_loss / count
    
    # Ortalama metrikler
    avg_metrics = {k: v / count for k, v in total_metrics.items()}
    
    # Standart sapma hesaplama
    metric_std = {}
    if len(batch_metrics_list) > 1:
        import numpy as np
        for key in total_metrics.keys():
            values = [batch_metrics.get(key, 0) for batch_metrics in batch_metrics_list]
            metric_std[key] = float(np.std(values))
    else:
        metric_std = {k: 0.0 for k in total_metrics.keys()}
    
    return avg_loss, avg_metrics, metric_std

  def _log_to_wandb(self, avg_metrics: Dict, avg_loss: float, metric_std: Dict, epoch: int) -> None:
    """WandB'a kapsamlı logging yapar."""
    
    log_dict = {
        # Ana metrikler
        "val/loss": avg_loss,
        "val/pesq": avg_metrics['pesq'],
        "val/stoi": avg_metrics['stoi'],
        "val/ssnr": avg_metrics['ssnr'],
        "val/csig": avg_metrics['csig'],
        "val/cbak": avg_metrics['cbak'],
        "val/covl": avg_metrics['covl'],
    }
    
    try:
      wandb.log(log_dict, step=epoch)
    except Exception as e:
      print(f"WandB logging hatası: {e}")

  # Egitim dongusunu baslatir
  def train(self, max_steps=None, max_epochs=None, val_dataset=None, early_stopping_patience=4):
    # Modeli dogru cihaza (GPU/CPU) tasir
    device = next(self.model.parameters()).device
    epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    while True:
      # Modeli egitim moduna alir
      self.model.train()
      epoch_loss = 0
      batch_count = 0
     
      if max_epochs is not None and epoch >= max_epochs:
        print(f'Maximum epoch sayisina ulasildi: {max_epochs}. Egitim durduruluyor.')
        break
        
      for features in tqdm(self.dataset, desc=f'Epoch {epoch}') if self.is_master else self.dataset:
        
        # Verileri dogru cihaza (GPU/CPU) tasir
        features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)

        loss = self.train_step(features)

        # NaN kaybı kontrolü
        if torch.isnan(loss).any():
          raise RuntimeError(f'Detected NaN loss at step {self.step}.')
        
        epoch_loss += loss.item()
        batch_count += 1
        self.step += 1

      if self.is_master:
        avg_train_loss = epoch_loss / batch_count if batch_count > 0 else 0
        
        # Epoch sonunda validation yap
        val_loss = None
        if val_dataset is not None:
            val_loss = self.validate(val_dataset, epoch)

        # Train ve validation loss'unu epoch bazında aynı grafikte daha duzgun gorunur
        log_dict = {
            "loss/train": avg_train_loss,
        }
        
        if val_loss is not None:
            log_dict["loss/validation"] = val_loss

        wandb.log(log_dict, step=epoch)  # step olarak epoch kullan

        # Early stopping ve best model kaydetme
        if val_loss is not None:
          if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            self.save_to_checkpoint(filename=f'best_weights-{best_val_loss:.4f}', epoch=epoch)
            print(f'🎉 New best validation loss: {val_loss:.4f} at epoch {epoch}')
          else:
            patience_counter += 1
            
          # Early stopping
          if patience_counter >= early_stopping_patience:
            print(f'⏹️ Early stopping at epoch {epoch} with best val loss {best_val_loss:.4f}')
            break
              
        print(f'📊 Epoch {epoch}: Train Loss = {avg_train_loss:.4f}' + 
              (f', Val Loss = {val_loss:.4f}' if val_loss is not None else ''))
              
      epoch += 1

  # Tek bir egitim adimi
  def train_step(self, features):
    # Gradyanlari sifirlar
    for param in self.model.parameters():
      param.grad = None

    audio = features['clean_speech']
    noisy = features['noisy_speech']
    audio_orig = features['clean_speech'].clone()

    N,T= audio.shape
    device = audio.device
    self.noise_level = self.noise_level.to(device)

    with self.autocast:
      # Rastgele bir gurultu seviyesi (t) secer
      t = torch.randint(0, len(self.params.noise_schedule), [N], device=audio.device)
      
      noise_scale = self.noise_level[t].unsqueeze(1)
      noise_scale_sqrt = noise_scale**0.5
      noise = torch.randn_like(audio)
      
      # Dropout (rastgele bazi verileri maskeler) uygulama
      masks = torch.bernoulli(torch.zeros(N)+self.dropout)
      
      for i in range(masks.size(0)):
        if masks[i] == 1:
          audio[i] = torch.randn_like(audio[i])
      
      # Gürültülü sesi oluşturur
      noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale)**0.5 * noise
      # Modeli çalıştırır ve tahmini alır
      predicted = self.model(noisy_audio, t, noisy)
      # Tahmin ile orijinal temiz ses arasindaki kaybi hesaplar
      loss = self.loss_fn(audio_orig, predicted.squeeze(1))

    # Geriye yayilimi (backpropagation) baslatir
    self.scaler.scale(loss).backward()
    self.scaler.unscale_(self.optimizer)
    # Gradyanlari kirpar
    self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm or 1e9)
    # Optimizasyon adimini gerceklestirir
    self.scaler.step(self.optimizer)
    self.scaler.update()
    
    # EMA guncelleme
    if self.ema_helper is not None:
      self.ema_helper.update(self.model)
      
    return loss

  # W&B loglama yapar
  
  #def _write_summary(self, step, features, loss):
  #  wandb.log({
  #    "train/loss" : loss.item(), # Egitim kaybini logla
  #    "train/grad_norm" : self.grad_norm, # Egitim gradyan normunu logla
  #    "train/features" : features,
  #    "train/step" : step,
  #  }, step = step)

# Egitim uygulamasinin ana fonksiyonu
def _train_impl(replica_id, model, dataset, args, params, val_dataset):
  # Hız optimizasyonları
  torch.backends.cudnn.benchmark = True

  # Adam optimizasyonu
  opt = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
  
  learner = DOSELearner(
    args.model_dir,
    model,
    dataset,
    opt,
    params,
    fp16=args.fp16,
    use_ema=getattr(args, 'use_ema', True), # EMA parametresi
    ema_decay=getattr(args, 'ema_decay', 0.999)
  )
  learner.is_master = (replica_id == 0)

  # Arguman belirtilmisse o dosyadan yukle, yoksa varsayilani kullan
  if args.restore_model_name:
    learner.restore_from_checkpoint(filename=args.restore_model_name)
  else:
    learner.restore_from_checkpoint()
  learner.train(max_steps=args.max_steps, max_epochs=args.max_epochs, val_dataset=val_dataset)

  if learner.is_master:
    wandb.finish() # W&B oturumunu sonlandır

# Egitim surecini baslatir
def train(args, params):
  
  wandb.init(
        project="dose-speech-enhancement", # W&B projesinin adı
        job_type="train", # Çalışmanın türü (eğitim)
        name = f"train_run_on_{args.train_noisy_speech_dir} - {args.train_clean_speech_dir}", # Oturum için benzersiz bir ad oluşturur
        config= {
          'learning_rate': params.learning_rate,
          'dropout_rate': params.dropout_rate,
          'step1': params.step1,
          'step2': params.step2,
          'use_ema': params.use_ema,
          'ema_decay': params.ema_decay
        } # Parametreleri W&B ile paylaş
    )

  # wandb.config'i params icine uygula
  config = wandb.config
  params.learning_rate = config.learning_rate
  params.dropout_rate = config.dropout_rate
  params.step1 = config.step1
  params.step2 = config.step2

  # Gürültülü ve temiz ses dosyalarını yükle
  dataset = from_path(args.train_noisy_speech_dir, args.train_clean_speech_dir, params)
  val_dataset = from_path(args.val_noisy_speech_dir, args.val_clean_speech_dir, params) if hasattr(args, 'val_noisy_speech_dir') and hasattr(args, 'val_clean_speech_dir') else None
  # Cihazı ayarla
  device = torch.device('cuda', args.device_num)
  # DOSE modelini baslatir ve cihaza tasir
  model = DOSE(params).to(device)

  # Modeli ve parametreleri izlemeye başla
  # wandb.watch(model, log="all", log_freq=500)

  _train_impl(0, model, dataset, args, params, val_dataset)

