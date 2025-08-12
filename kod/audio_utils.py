import numpy as np
from scipy.io import wavfile

def read_wav_mono(path):
    """Reads a WAV file and converts it to a normalized, mono float signal."""
    # 1. Ses dosyasını ve örnekleme frekansını oku
    fs, data = wavfile.read(path)
    
    # 2. Veri tipini kontrol et ve normalize et
    if data.dtype.kind == 'i':
        # Tamsayı (integer) ise, [-1.0, 1.0] aralığında ondalıklı (float) sayıya dönüştür
        max_val = float(np.iinfo(data.dtype).max)
        data = data.astype(float) / (max_val + 1.0)
    elif data.dtype.kind == 'f':
        # Zaten ondalıklı ise, sadece tipini float olarak doğrula
        data = data.astype(float)

    # 3. Çok kanallı (stereo vb.) ise tek kanala (mono) indirge
    if data.ndim > 1:
        data = data.mean(axis=1)

    # 4. Örnekleme frekansını ve işlenmiş mono sinyali döndür
    return fs, data