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


# # İç içe geçmiş veri yapılarını (listeler, sözlükler vb.) bir fonksiyonla eşlemek için yardımcı fonksiyon.
def _nested_map(struct, map_fn):
  if isinstance(struct, tuple):
    return tuple(_nested_map(x, map_fn) for x in struct)
  if isinstance(struct, list):
    return [_nested_map(x, map_fn) for x in struct]
  if isinstance(struct, dict):
    return { k: _nested_map(v, map_fn) for k, v in struct.items() }
  return map_fn(struct)


# DOSE modelini egitmek icin ana sinif
class DOSELearner:
  def __init__(self, model_dir, model, dataset, optimizer, params, *args, **kwargs):
    # Modelin kaydedileceği dizini oluştur
    os.makedirs(model_dir, exist_ok=True)
    self.model_dir = model_dir
    self.model = model
    self.dataset = dataset
    self.optimizer = optimizer
    self.params = params
    # Mixed precision eğitimi icin autocast ve scaler ayarlari
    self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get('fp16', False))
    self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get('fp16', False))
    self.step = 0
    self.is_master = True

    # Gürültü programını hesaplar
    beta = np.array(self.params.noise_schedule)
    noise_level = np.cumprod(1 - beta)
    self.noise_level = torch.tensor(noise_level.astype(np.float32))
    self.loss_fn = nn.L1Loss() # L1 kaybı fonksiyonunu kullanır
    self.summary_writer = None
    self.dropout = params.dropout_rate

  # Modelin ve optimizer'in durumunu kaydetmek icin sozluk dondurur
  def state_dict(self):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      model_state = self.model.module.state_dict()
    else:
      model_state = self.model.state_dict()
    return {
        'step': self.step,
        'model': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items() },
        'optimizer': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items() },
        'params': dict(self.params),
        'scaler': self.scaler.state_dict(),
    }

  # Kaydedilmis durumu yukler
  def load_state_dict(self, state_dict):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      self.model.module.load_state_dict(state_dict['model'])
    else:
      self.model.load_state_dict(state_dict['model'])
    self.optimizer.load_state_dict(state_dict['optimizer'])
    self.scaler.load_state_dict(state_dict['scaler'])
    self.step = state_dict['step']

  # Kontrol noktasini dosyaya kaydeder
  def save_to_checkpoint(self, filename='weights'):
    save_basename = f'{filename}-{self.step}.pt'
    save_name = f'{self.model_dir}/{save_basename}'
    link_name = f'{self.model_dir}/{filename}.pt'
    torch.save(self.state_dict(), save_name)
    if os.name == 'nt':
      torch.save(self.state_dict(), link_name)
    else:
      if os.path.islink(link_name):
        os.unlink(link_name)
      os.symlink(save_basename, link_name)

  # Dosyadan kontrol noktasini yukler
  def restore_from_checkpoint(self, filename='weights'):
    try:
      checkpoint = torch.load(f'{self.model_dir}/{filename}.pt')
      self.load_state_dict(checkpoint)
      return True
    except FileNotFoundError:
      return False
    
  def _write_test_summary(self, step, loss):
    writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
    writer.add_scalar('test/loss', loss, step)
    writer.flush()
    self.summary_writer = writer  

    return
  
  def validate(self, val_dataset, epoch):
    self.model.eval()
    device = next(self.model.parameters()).device
    total_loss = 0
    total_metrics = {'csig' : 0, 'cbak' : 0, 'covl' : 0, 'pesq' : 0, 'ssnr' : 0, 'stoi' : 0}
    count = 0
    with torch.no_grad():
      for features in val_dataset:
        features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
        audio = features['clean_speech']
        noisy = features['noisy_speech']
        N, T = audio.shape
        t = torch.randint(0, len(self.params.noise_schedule), [N], device=audio.device)
        noise_scale = self.noise_level[t].unsqueeze(1)
        noise_scale_sqrt = noise_scale**0.5
        noise = torch.randn_like(audio)
        noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale)**0.5 * noise
        predicted = self.model(noisy_audio, t, noisy)
        loss = self.loss_fn(audio, predicted.squeeze(1))
        total_loss += loss.item()
        # Sadece ilk ornek icin metrikleri hesapla
        clean_np = audio[0].cpu().numpy()
        predicted_np = predicted[0].cpu().numpy()
        ssnr, pesq, csig, cbak, covl, stoi = compare(clean_np, predicted_np, self.params.sample_rate)
        total_metrics["csig"] = csig
        total_metrics["cbak"] = cbak
        total_metrics["covl"] = covl
        total_metrics["pesq"] = pesq
        total_metrics["ssnr"] = ssnr
        total_metrics["stoi"] = stoi
        count += 1
    avg_loss = total_loss / count if count > 0 else 0
    avg_metrics= {k: v / count if count > 0 else 0 for k, v in total_metrics.items()}
    wandb.log({
      "val/loss" : avg_loss,
      "val/csig" : avg_metrics["csig"],
      "val/cbak" : avg_metrics["cbak"],
      "val/covl" : avg_metrics["covl"],
      "val/pesq" : avg_metrics["pesq"],
      "val/ssnr" : avg_metrics["ssnr"],
      "val/stoi" : avg_metrics["stoi"],
    }, step=self.epoch)
    self._write_test_summary(self.step, avg_loss)
    return avg_loss

  # Egitim dongusunu baslatir
  def train(self, max_steps=None, val_dataset=None, early_stopping_patience=10):
    # Modeli dogru cihaza (GPU/CPU) tasir
    device = next(self.model.parameters()).device
    epoch = 0
    
    while True:
      # Modeli egitim moduna alir
      self.model.train()
      epoch_loss = 0
      batch_count = 0
      best_val_loss = float('inf')
      patience_counter = 0

      for features in tqdm(self.dataset, desc=f'Epoch {self.step // len(self.dataset)}') if self.is_master else self.dataset:
        if max_steps is not None and self.step >= max_steps:
          return
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
          wandb.log({
            "train/loss": avg_train_loss,
            "train/epoch": epoch
          }, step = epoch)
          if val_dataset is not None:
            val_loss = self.validate(val_dataset)
            wandb.log({
              "val/loss": val_loss,
              "val/epoch": epoch
            }, step = epoch)

            # best model kaydetme
            if val_loss < best_val_loss:
              best_val_loss = val_loss
              patience_counter = 0
              self.save_to_checkpoint(filename='best_weights')
            else:
              patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
              print(f'Early stopping at epoch {epoch} with best val loss {best_val_loss:.4f}')
              break
      self.save_to_checkpoint()
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

  learner = DOSELearner(args.model_dir, model, dataset, opt, params, fp16=args.fp16)
  learner.is_master = (replica_id == 0)

  # Arguman belirtilmisse o dosyadan yukle, yoksa varsayilani kullan
  if args.restore_model_name:
    learner.restore_from_checkpoint(filename=args.restore_model_name)
  else:
    learner.restore_from_checkpoint()
  learner.train(max_steps=args.max_steps, val_dataset=val_dataset)

  if learner.is_master:
    wandb.finish() # W&B oturumunu sonlandır

# Egitim surecini baslatir
def train(args, params):
  
  wandb.init(
        project="dose-speech-enhancement", # W&B projesinin adı
        job_type="train", # Çalışmanın türü (eğitim)
        name = f"train_run_on_{args.noisy_speech_dir} - {args.clean_speech_dir} - {params.model_name}", # Oturum için benzersiz bir ad oluşturur
        config= params # Parametreleri W&B ile paylaş
    )

  # Gürültülü ve temiz ses dosyalarını yükle
  dataset = from_path(args.noisy_speech_dir, args.clean_speech_dir, params)
  val_dataset = from_path(args.val_noisy_speech_dir, args.val_clean_speech_dir, params) if hasattr(args, 'val_noisy_speech_dir') and hasattr(args, 'val_clean_speech_dir') else None 
  # Cihazı ayarla
  device = torch.device('cuda', args.device_num)
  # DOSE modelini baslatir ve cihaza tasir
  model = DOSE(params).to(device)

  # Modeli ve parametreleri izlemeye başla
  wandb.watch(model, log="all", log_freq=500)

  _train_impl(0, model, dataset, args, params, val_dataset)

