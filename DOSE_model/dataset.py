# Gerekli kütüphaneleri içe aktar
import numpy as np
import os
import random
import torch
import torch.nn.functional as F
import torchaudio

import glob
from torch.utils.data.distributed import DistributedSampler
import librosa

from torch.utils.data import SubsetRandomSampler


# Gurulutulu ve temiz ses verilerini birleştiren sinif
class ConcatDataset(torch.utils.data.Dataset):
  def __init__(self, noisy_root, clean_root, params):
    super().__init__()
    # Gurultulu ve temiz ses dosyalarinin kok dizinlerini saklar
    self.noisy_root = noisy_root
    self.clean_root = clean_root
    # Parametreleri saklar
    self.params = params
    # Kok dizin icindeki tum .wav dosyalarinin yollarini alir.
    self.raw_paths = [x.split('/')[-1] for x in glob.glob(noisy_root + '/*.wav')]
    print(self.noisy_root)

  # Veri kümesinin uzunluğunu döndürür
  def __len__(self):

    return len(self.raw_paths)

  # Belirtilen indeksteki gurultulu ve temiz ses verilerini döndürür
  def __getitem__(self, index):
    
    raw_paths = self.raw_paths
    noisy, _ = librosa.load(os.path.join(self.noisy_root, raw_paths[index]), sr=16000)  
    clean, _ = librosa.load(os.path.join(self.clean_root, raw_paths[index]), sr=16000)
    
    # Mel-spektrogram hesapla (conditional mode icin)
    mel_spec = librosa.feature.melspectrogram(
      y=noisy,
      sr=self.params.sample_rate,
      n_fft=self.params.n_fft,
      hop_length=self.params.hop_samples,
      n_mels=self.params.n_mels
    )
    
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)


    # Gurultulu ve temiz ses verilerini iceren bir sozluk dondurur
    return {
            'noisy_speech': noisy,
            'clean_speech': clean,
            'mel_spectrogram': mel_spec
        }


# Veri birlestirme sinifi
class Collator:
  def __init__(self, params):
    # Modelin parametrelerini saklar
    self.params = params


  def concat_collate(self, minibatch):
    for record in minibatch: 
        if self.params.unconditional:
          # Ses uzunlugu belirlenen degerden kisa olanlari doldurma (padding) yapar
            if len(record['clean_speech']) < self.params.audio_len:
                # print(len(record['clean_speech']))
                start = 0
                end = start + self.params.audio_len
                record['clean_speech'] = np.pad(record['clean_speech'], (0, (end - start) - len(record['clean_speech'])), mode='constant')
                record['noisy_speech'] = np.pad(record['noisy_speech'], (0, (end - start) - len(record['noisy_speech'])), mode='constant')
                # continue
            # todo 不删除而是进行填补
            # Rastgele bir baslangic noktasi belirleyerek sesi kirpar
            start = random.randint(0, record['clean_speech'].shape[-1] - self.params.audio_len)
            end = start + self.params.audio_len
            # Belirlenen uzunlukta ses parcalarini alir
            record['clean_speech'] = record['clean_speech'][start:end]            
            record['noisy_speech'] = record['noisy_speech'][start:end]   

        else:
          # Conditional mode icin hem ses verilerini hem de mel-spektrogramu boyutlarini esitle
          mel_spec = record['mel_spectrogram']
          
          # Ses verilerinin uzunluğunu belirle
          clean_len = len(record['clean_speech'])
          noisy_len = len(record['noisy_speech'])
          
          # Eğer audio_len parametresi varsa onu kullan, yoksa minimum uzunluğu kullan
          if hasattr(self.params, 'audio_len') and self.params.audio_len is not None:
            target_len = self.params.audio_len
          else:
            target_len = min(clean_len, noisy_len)
          
          # Ses verilerini padding veya cropping ile standart boyuta getir
          if clean_len < target_len:
            # Padding
            record['clean_speech'] = np.pad(record['clean_speech'], (0, target_len - clean_len), mode='constant')
          elif clean_len > target_len:
            # Cropping (rastgele başlangıç noktası)
            if clean_len > target_len:
              start = random.randint(0, clean_len - target_len) if clean_len > target_len else 0
            else:
              start = 0
            record['clean_speech'] = record['clean_speech'][start:start + target_len]
          
          if noisy_len < target_len:
            # Padding
            record['noisy_speech'] = np.pad(record['noisy_speech'], (0, target_len - noisy_len), mode='constant')
          elif noisy_len > target_len:
            # Cropping (aynı başlangıç noktasını kullan)
            if clean_len > target_len:
              start = random.randint(0, noisy_len - target_len) if noisy_len > target_len else 0
            else:
              start = 0
            record['noisy_speech'] = record['noisy_speech'][start:start + target_len]
          
          # Mel-spektrogram boyutlarını ayarla
          audio_len = target_len
          expected_spec_len = audio_len // self.params.hop_samples + 1
          
          if mel_spec.shape[1] > expected_spec_len:
            mel_spec = mel_spec[:, :expected_spec_len]
          elif mel_spec.shape[1] < expected_spec_len:
            pad_width = expected_spec_len - mel_spec.shape[1]
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)))
        
          record['mel_spectrogram'] = mel_spec
          
          
          
    # Minibatch'teki gurulutlu ve temiz sesleri yigin (stack ) haline getirir.
    clean_speech = np.stack([record['clean_speech'] for record in minibatch if 'clean_speech' in record])
    noisy_speech = np.stack([record['noisy_speech'] for record in minibatch if 'noisy_speech' in record])
    
    if self.params.unconditional:
        # Verileri PyTorch tensorlerine donusturur.
        return {
            'clean_speech': torch.from_numpy(clean_speech),
            'noisy_speech': torch.from_numpy(noisy_speech),
        }
    
    else:
      # Conditional mode icin spektrogram da dondur
      mel_spectrogram = np.stack([record['mel_spectrogram'] for record in minibatch if 'mel_spectrogram' in record])
      return {
          'clean_speech': torch.from_numpy(clean_speech),
          'noisy_speech': torch.from_numpy(noisy_speech),
          'mel_spectrogram': torch.from_numpy(mel_spectrogram)
      }

# Veri setini ve veri yükleyicisini oluşturur
def from_path(noisy_root,clean_root, params, is_distributed=False):

    # Veri kümesini oluşturur
    dataset = ConcatDataset(noisy_root,clean_root, params)
    
    # PyTorch DataLoader'i olusturur ve dondurur.
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=params.batch_size,
        collate_fn=Collator(params).concat_collate,
        shuffle=False,
        num_workers=os.cpu_count(),
        sampler=DistributedSampler(dataset) if is_distributed else None,
        pin_memory=True,
        drop_last=False)

