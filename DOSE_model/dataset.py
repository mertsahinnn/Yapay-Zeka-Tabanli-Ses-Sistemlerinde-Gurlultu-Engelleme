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
  def __init__(self, noisy_root, clean_root):
    super().__init__()
    # Gurultulu ve temiz ses dosyalarinin kok dizinlerini saklar
    self.noisy_root = noisy_root
    self.clean_root = clean_root
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


    # Gurultulu ve temiz ses verilerini iceren bir sozluk dondurur
    return {
            'noisy_speech': noisy,
            'clean_speech': clean,
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
    
    # Minibatch'teki gurulutlu ve temiz sesleri yigin (stack ) haline getirir.
    clean_speech = np.stack([record['clean_speech'] for record in minibatch if 'clean_speech' in record])
    noisy_speech = np.stack([record['noisy_speech'] for record in minibatch if 'noisy_speech' in record])
    
    if self.params.unconditional:
        # Verileri PyTorch tensorlerine donusturur.
        return {
            'clean_speech': torch.from_numpy(clean_speech),
            'noisy_speech': torch.from_numpy(noisy_speech),
        }
    

# Veri setini ve veri yükleyicisini oluşturur
def from_path(noisy_root,clean_root, params, is_distributed=False):

    # Veri kümesini oluşturur
    dataset = ConcatDataset(noisy_root,clean_root)
    
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

