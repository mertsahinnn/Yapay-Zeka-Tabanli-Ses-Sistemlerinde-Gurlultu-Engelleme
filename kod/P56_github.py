"""
asl_P56_py
Python implementation of ITU ASL P56.

Code is ported from the MATLAB implementation 
found in the attachement to 
Philipos C. Loizou's Speech Enhancement Book.

George Close (glclose1@sheffield.ac.uk) 2022
"""

import numpy as np
from scipy.signal import lfilter

def asl_P56(x,fs,nbits):
    """
    ITU-T P.56'ya göre Aktif Konuşma Seviyesini (ASL) hesaplar.
    Bu kod, orijinal MATLAB kodunun doğrudan bir çevirisidir ve bazı verimsizlikler içerir.

    Args:
        x (np.array): Giriş sinyali (float, -1.0 ile 1.0 arasında normalize edilmiş varsayılır).
        fs (float): Örnekleme frekansı (Hz).
        nbits (int): Sinyalin bit derinliği (genellikle 16).

    Returns:
        tuple: (asl_ms, asl, c0)
            - asl_ms (float): Aktif konuşma gücü (mean-square). RMS seviyesi için karekökü alınmalıdır.
            - asl (float): Aktivite faktörü (0-1 arası).
            - c0 (float): Tespit edilen aktivite eşiği (lo).
    """
    # --- 1. Parametreleri Ayarla ---
    T = 0.03 # Zaman sabitesi (P.56 standardına göre 0.03s). Orijinal kodda 0.23s idi.
    H = 0.2  # Hangover süresi (saniye).
    M = 15.9 # Marj (dB).
    
    # Sinyalin float olduğundan emin ol
    x = np.asarray(x, dtype=float)

    # --- 2. Eşikleri Oluştur ---
    # Orijinal kodda bu bölüm hatalıydı. Düzeltilmiş hali:
    # 16-bit için 2**-15'ten 2**-1'e kadar 15 eşik oluşturulur.
    num_thresholds = nbits - 1 
    c = 2.0 ** np.arange(-num_thresholds, 0) # 2**-15, 2**-14, ..., 2**-1

    # --- 3. Aktivite ve Hangover Sayıcılarını Hazırla ---
    I = np.ceil(fs * H) # Hangover süresi (örnek sayısı cinsinden)
    a = np.zeros(num_thresholds) # Her eşik için aktivite sayacı
    hang = np.full(num_thresholds, I) # Her eşik için hangover sayacı

    # --- 4. Sinyal Zarfını Hesapla ---
    mean_square_energy = np.mean(x**2)
    if mean_square_energy == 0:
        return -np.inf, 0.0, 0.0

    total_power = np.sum(x**2) # Orijinal koddaki 'sq'
    x_len = len(x)

    g = np.exp(-1 / (fs * T)) # Yumuşatma faktörü
    x_abs = np.abs(x)
    # Orijinal koddaki lfilter parametresi [1, g] şeklindeydi, doğrusu [1, -g]'dir.
    p = lfilter([1 - g], [1, -g], x_abs) 
    q = lfilter([1 - g], [1, -g], p) 

    # --- 5. Aktiviteyi Say (Hangover ile birlikte) ---
    # UYARI: Bu döngü çok verimsizdir ve uzun sinyaller için yavaştır.
    for k in range(x_len):
        for j in range(num_thresholds):
            if (q[k] >= c[j]):
                a[j] = a[j] + 1
                hang[j] = 0
            elif hang[j] < I:
                a[j] = a[j] + 1
                hang[j] = hang[j] + 1
            else:
                break

    # --- 6. Kesişim Noktasını Bul ---
    if a[0] == 0:
        # Hiç aktivite bulunamadı
        return -np.inf, 0.0, 0.0
    
    # Aktivite (AdB) ve Eşik (CdB) eğrilerini dB cinsinden hesapla
    # 1e-10 gibi küçük bir değer log(0) hatasını önlemek için eklenir.
    AdB = 10 * np.log10(total_power / (a + 1e-10))
    CdB = 20 * np.log10(c + 1e-10)
    Delta = AdB - CdB

    # İlk eşik zaten marjın altındaysa, sonuç bulunamaz.
    if Delta[0] < M:
        # Genellikle sinyal çok kısa veya sessiz olduğunda olur.
        return -np.inf, 0.0, 0.0
    
    asl_ms = 0
    asl = 0
    c0 = 0
    
    # Delta'nın M'nin altına düştüğü ilk aralığı bul
    for j in range(1, num_thresholds):
       if a[j] == 0: # Aktivite olmayan bir eşiğe geldiysek dur
           break
       
       if Delta[j] <= M:
           # Kesişim j-1 ve j arasında. İnterpolasyon yap.
           asl_ms_log, c10 = bin_interp(AdB[j], AdB[j-1], CdB[j], CdB[j-1], M, 0.5)
           
           # Sonuçları logaritmik'ten lineer'e çevir
           asl_ms = 10**(asl_ms_log / 10)
           
           # Aktivite faktörü = (toplam güç / sinyal uzunluğu) / aktif konuşma gücü
           asl = mean_square_energy / asl_ms
           
           c0 = 10**(c10 / 20)
           break
           
    return asl_ms, asl, c0




def bin_interp(upcount,lwcount,upthr,lwthr,Margin,tol):
    if tol < 0:
        tol = -tol

    # Check if extreme counts are not already the true active value
    iterno = 1
    if (np.abs(upcount-upthr - Margin) < tol):
        asl_ms_log = upcount
        cc = upthr
        return asl_ms_log,cc
    if (np.abs(lwcount-lwthr - Margin) < tol):
        asl_ms_log = lwcount
        cc = lwthr
        return asl_ms_log,cc
    
    # Initalize first middle for given (initial) bounds
    midcount = (upcount + lwcount)/2.0
    midthr = (upthr + lwthr)/2.0

    # Repeats loop until `diff' falls inside the tolerance (-tol<=diff<=tol
    while True:
        diff = midcount  - midthr - Margin
        if np.abs(diff) <= tol:
            break
        # if tolerance is not met up to 20 iteractions, then relax the
        # tolerance by 10%
        iterno = iterno + 1

        if iterno > 20:
            tol = tol*1.1

        if diff > tol: # then the new bounds are...
            midcount = (upcount+midcount)/2.0
            # upper and middle activitiy
            midthr = (upthr+midthr)/2.0
            # and the thresholds
        elif diff < -tol: # then the new bounds are...
            midcount = (midcount+lwcount)/2.0
            # middle and lower activity
            midthr = (midthr+lwthr)/2.0
            # and the thresholds
    # Since the tolerance has been satisfied, midcount is selected 
    # as the interpolated value with a tol [dB] tolerance.
    asl_ms_log = midcount
    cc = midthr
    return asl_ms_log,cc




if __name__ == '__main__':
    import soundfile as sf
    try:
        dosya_yolu = 'C:\\D----------------\\Staj\\makale\\ses\\21.wav'
        # Örnek bir dosya ile test etme
        speech, fs = sf.read(dosya_yolu)
        # Sinyali -1 ile 1 arasına normalize et (eğer tamsayı ise)
        if np.issubdtype(speech.dtype, np.integer):
            speech = speech / np.iinfo(speech.dtype).max

        # Fonksiyonu çağır ve sonuçları anlamlı değişkenlere ata
        asl_ms, activity_factor, threshold = asl_P56(speech, float(fs), 16)

        # Sonuçları okunaklı bir şekilde yazdır
        print("\n--- P.56 Aktif Konuşma Seviyesi Analizi Sonuçları ---")
        print(f"Dosya: {dosya_yolu}\n")

        if asl_ms > 0 and activity_factor > 0:
            # Daha sezgisel değerler hesapla: RMS ve dB
            asl_rms = np.sqrt(asl_ms)
            asl_db = 20 * np.log10(asl_rms)

            print(f"Aktif Konuşma Gücü (asl_ms) : {asl_ms:.6f}")
            print(f"Aktif Konuşma Seviyesi (RMS): {asl_rms:.4f}")
            print(f"Aktif Konuşma Seviyesi (dB) : {asl_db:.2f} dB (FS'e göre)")
            print(f"Aktivite Faktörü (asl)     : {activity_factor*100:.2f} % (Sinyalin ne kadarının aktif olduğunu gösterir)")
            print(f"Tespit Edilen Eşik (c0)     : {threshold:.6f}")
        else:
            print("Sinyalde anlamlı bir aktif konuşma tespit edilemedi.")

    except FileNotFoundError:
        print(f"Hata: Test için '{dosya_yolu}' dosyası bulunamadı.")