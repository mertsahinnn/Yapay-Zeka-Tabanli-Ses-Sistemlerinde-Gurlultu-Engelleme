"""
ITU-T P.56 Method B - Active Speech Level (Python)
References: ITU-T P.56 and Kabal - "Measuring Speech Activity".
(See citations in assistant response)

Requires:
  pip install numpy scipy

Usage:
  python p56_active_level.py path/to/file.wav
"""
import sys
import numpy as np
from scipy.io import wavfile
from audio_utils import read_wav_mono

def active_speech_level(x, fs, time_constant=0.03, hangover=0.2, M_dB=15.9, b=2.0):
    """
    Compute active speech level using ITU-T P.56 Method B (Kabal description).

    Inputs:
      x            : 1-D numpy array (floating, signal; can be -1..1 or integer range)
      fs           : sampling frequency (Hz)
      time_constant: envelope smoothing time constant T (seconds) — default 0.03 s per P.56.
      hangover     : hangover time H (seconds) — default 0.2 s per P.56.
      M_dB         : margin M in dB (15.9 dB in P.56)
      b            : geometric step for thresholds (2.0 per P.56 for 16-bit examples)

    Returns:
      act_level_db : active level in dB relative to full-scale (dBov if x is FS-normalized)
      activity     : fractional activity a(lo) (0..1)
      info dict    : internal values (lo, thresholds, counts, etc.)
    """
    # --- 1. Başlangıç Kontrolleri ---
    x = np.asarray(x, dtype=float)
    Ns = x.size
    if Ns == 0:
        raise ValueError("Empty signal")

    # Sinyalin ortalama gücünü (enerjisini) hesapla
    Ex = np.mean(x**2)
    if Ex == 0:
        return -np.inf, 0.0, {"reason": "zero-energy"}

    # --- 2. Sinyal Zarfını (Envelope) Hesaplama ---
    # P.56'ya göre çift üstel düzeltme (double exponential smoothing) uygulanır.
    t = 1.0 / fs
    g = np.exp(-t / time_constant)
    absx = np.abs(x)
    p = np.empty_like(absx)
    q = np.empty_like(absx)
    p_prev = 0.0
    q_prev = 0.0
    one_minus_g = (1.0 - g)
    for i, v in enumerate(absx):
        p_curr = g * p_prev + one_minus_g * v
        q_curr = g * q_prev + one_minus_g * p_curr
        p[i] = p_curr
        q[i] = q_curr
        p_prev = p_curr
        q_prev = q_curr

    # --- 3. Hangover Uygulaması ---
    # Zarf (q), konuşma sonlarındaki düşüşleri yumuşatmak için hareketli maksimum filtresinden geçirilir.
    I = int(np.ceil(hangover / t))
    if I < 1:
        q_tilde = q.copy()
    else:
        # efficient moving maximum (simple implementation)
        from collections import deque
        q_tilde = np.empty_like(q)
        dq = deque()
        for i, val in enumerate(q):
            # pop from right while smaller than val
            while dq and dq[-1][0] <= val:
                dq.pop()
            dq.append((val, i))
            # remove outdated
            while dq and dq[0][1] < i - I + 1:
                dq.popleft()
            q_tilde[i] = dq[0][0]

    # --- 4. Eşik Seviyelerini Oluşturma ---
    # Aktiviteyi ölçmek için geometrik bir dizi halinde eşik seviyeleri (cj) oluşturulur.
    qmax = q_tilde.max()
    if qmax <= 0:
        return -np.inf, 0.0, {"reason": "no-envelope-energy"}

    Nlevels = 30
    c_max = qmax
    c_min = c_max / (b ** (Nlevels - 1))
    cj = c_min * (b ** np.arange(Nlevels))

    # --- 5. Aktiviteyi Hesaplama ---
    # Her bir eşik (cj) için, zarfın (q_tilde) o eşiğin üzerinde kaldığı örnek sayısı (aj) sayılır.
    aj = np.array([(q_tilde >= c).sum() for c in cj], dtype=float)
    a_frac = aj / float(Ns)

    # --- 6. Kesişim Noktasını Bulma ---
    # Kabal'ın tanımına göre, logaritmik alanda iki eğrinin kesişim noktası bulunur.
    # Bu nokta, aktif konuşma seviyesini belirler.
    eps = 1e-300
    a_nonzero = np.maximum(a_frac, eps)
    Alnj = np.log(Ex / a_nonzero)  # natural log
    Clnj = 2.0 * np.log(np.maximum(cj, eps))

    # M in natural log units: Mln = ln(m) where m = 10^(M/10)
    Mln = np.log(10 ** (M_dB / 10.0))
    dAC = Alnj - Clnj
    ActLev = 0.0
    lo = 0.0
    activity = 0.0

    # Kesişim noktasını bulmak için eşikler üzerinde döngüye girilir.
    # Eğer ilk aralıkta veya sonraki aralıklarda kesişim bulunursa, interpolasyon yapılır.
    found = False
    prev_d = None
    prev_Aln = None
    for j in range(len(dAC)):
        if a_frac[j] == 0.0:
            break
        d = dAC[j]
        if j == 0:
            if d <= Mln:
                # crossing in first interval
                Alno = Alnj[j]
                ActLev = np.exp(Alno / 2.0)
                lo = cj[j]
                activity = a_frac[j]
                found = True
                break
        else:
            if d <= Mln:
                # j-1 ve j arasında lineer interpolasyon yaparak hassas kesişim noktası bulunur.
                d_prev = prev_d
                Aln_prev = prev_Aln
                alpha = (Mln - d_prev) / (d - d_prev + 1e-300)
                Alno = Aln_prev + alpha * (Alnj[j] - Aln_prev)
                ActLev = np.exp(Alno / 2.0)
                lo = ActLev / np.sqrt(10 ** (M_dB / 10.0))
                # activity a(lo) from Eq (17): a(lo) = Ex / (m * lo^2)
                m = 10 ** (M_dB / 10.0)
                activity = Ex / (m * (lo ** 2) + 1e-300)
                found = True
                break
        prev_d = d
        prev_Aln = Alnj[j]

    if not found:
        # Kesişim bulunamazsa, aktivite sıfır kabul edilir.
        return -np.inf, 0.0, {"reason": "no_crossing", "dAC": dAC}

    # --- 7. Sonuçları Döndürme ---
    # Hesaplanan aktif seviyeyi (RMS genliği) desibel (dB) cinsine çevir.
    ref = 1.0  # Referans olarak tam ölçek (Full-Scale) kabul edilir.
    act_level_db = 20.0 * np.log10(ActLev / ref + 1e-300)

    # Detaylı analiz sonuçlarını bir sözlükte topla.
    info = {
        "Ex": Ex,
        "ActLev_sample_units": ActLev,
        "lo": lo,
        "M_dB": M_dB,
        "thresholds": cj,
        "activity_counts": aj,
        "activity_fraction": a_frac,
        "dAC": dAC,
        "q_tilde": q_tilde, # Görselleştirme ve doğrulama için
        "active_mask": q_tilde >= lo # Doğrulama için
    }
    return act_level_db, float(activity), info

if __name__ == "__main__":
    # --- Örnek Kullanım ---
    # Belirtilen WAV dosyasını oku ve aktif konuşma seviyesini hesapla.
    path = "C:\\D----------------\\Staj\\Kod deneme\\1.wav"
    fs, sig = read_wav_mono(path)
    level_db, activity, info = active_speech_level(sig, fs)
    # Hesaplanan değerleri ekrana yazdır.
    print(f"Active speech level: {level_db:.2f} dB (relative to FS / dBov if WAV normalized)")
    print(f"Fractional activity: {activity*100:.2f} %")
