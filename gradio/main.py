import gradio as gr
import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend ayarı
import torch
import torchaudio
import sys
import os
import tempfile
import warnings

# Uyarıları bastır
warnings.filterwarnings("ignore")

# DOSE model için gerekli importlar
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'DOSE_model'))
from params import AttrDict, params as base_params
from model import DOSE
from inference import predict

# Model global değişkenleri
models = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_waveform_plot(audio_path, title="Dalga Formu"):
    """Matplotlib ile dalga formu oluştur - HİÇ OPTİMİZASYON YOK"""
    try:
        print(f"Grafik oluşturuluyor: {audio_path}")
        
        # Ses dosyasını yükle - TAM BOYUT
        y, sr = librosa.load(audio_path, sr=16000)
        print(f"Ses yüklendi: {len(y)/sr:.2f} saniye, {len(y)} örnek")
        
        # Zaman eksenini oluştur - TAM BOYUT
        time = np.linspace(0, len(y) / sr, num=len(y))
        
        # Matplotlib figürü oluştur - TAM VERİ
        plt.figure(figsize=(16, 8))
        plt.plot(time, y, linewidth=0.5, color='#2E86AB', alpha=0.8)
        plt.title(f"{title} ({len(y)/sr:.1f} saniye)", fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Zaman (saniye)', fontsize=12)
        plt.ylabel('Genlik', fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # Stil iyileştirmeleri
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().set_facecolor('#f8f9fa')
        
        print(f"Matplotlib figürü oluşturuldu - {len(time)} nokta")
        return plt.gcf()
        
    except Exception as e:
        print(f"Dalga formu hatası: {e}")
        return None

def compare_waveforms(original_path, enhanced_path):
    """Matplotlib ile karşılaştırmalı dalga formu - HİÇ OPTİMİZASYON YOK"""
    try:
        print("Karşılaştırma grafiği oluşturuluyor...")
        
        # Orijinal ses - TAM BOYUT
        y_orig, sr_orig = librosa.load(original_path, sr=16000)
        time_orig = np.linspace(0, len(y_orig) / sr_orig, num=len(y_orig))
        print(f"Orijinal ses: {len(y_orig)/sr_orig:.2f} saniye, {len(y_orig)} örnek")
        
        # İşlenmiş ses - TAM BOYUT
        y_enh, sr_enh = librosa.load(enhanced_path, sr=16000)
        time_enh = np.linspace(0, len(y_enh) / sr_enh, num=len(y_enh))
        print(f"İşlenmiş ses: {len(y_enh)/sr_enh:.2f} saniye, {len(y_enh)} örnek")
        
        # Subplot oluştur - BÜYÜK BOYUT
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))
        
        # Orijinal dalga formu - TAM VERİ
        ax1.plot(time_orig, y_orig, linewidth=0.5, color='#E74C3C', alpha=0.8)
        ax1.set_title(f'Orijinal Ses ({len(y_orig)/sr_orig:.1f} saniye)', fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel('Zaman (saniye)', fontsize=11)
        ax1.set_ylabel('Genlik', fontsize=11)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_facecolor('#f8f9fa')
        
        # İşlenmiş dalga formu - TAM VERİ
        ax2.plot(time_enh, y_enh, linewidth=0.5, color='#3498DB', alpha=0.8)
        ax2.set_title(f'Gürültüsü Engellenmiş Ses ({len(y_enh)/sr_enh:.1f} saniye)', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Zaman (saniye)', fontsize=11)
        ax2.set_ylabel('Genlik', fontsize=11)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_facecolor('#f8f9fa')
        
        # Stil iyileştirmeleri
        for ax in [ax1, ax2]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.suptitle('Ses Karşılaştırması', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        print(f"Karşılaştırma grafiği oluşturuldu - Orijinal: {len(time_orig)} nokta, İşlenmiş: {len(time_enh)} nokta")
        return fig
        
    except Exception as e:
        print(f"Karşılaştırma hatası: {e}")
        return None

def create_info_page():
    with gr.Blocks() as info_page:
        gr.Markdown("""
        # Ses Sistemlerinde Gürültü Engelleme Projesi
        
        Bu proje, ses sistemlerinde gürültü engelleme üzerine yapılan bir çalışmadır.
        
        ## Proje Hakkında
        - Yapay zeka tabanlı gürültü engelleme (DOSE Modeli)
        - Ses kalitesini iyileştirme
        
        ## Nasıl Kullanılır?
        1. "Dalga Formu" sekmesine geçin
        2. Bir ses dosyası yükleyin (herhangi bir uzunluk)
        3. Tam çözünürlük dalga formunu inceleyin
        4. "Model" sekmesinde gürültü engelleme işlemini gerçekleştirin
        
        ## Kullanılan Model
        - **DOSE (Denoising Diffusion for Ordinary Speech Enhancement)**
        - 16kHz örnekleme frekansında çalışır
        - Diffusion-based denoising algoritması kullanır
        
        ## Teknik Özellikler
        - ✅ **Sınırsız ses uzunluğu** - herhangi bir uzunlukta ses dosyası
        - ✅ **Tam çözünürlük** - hiçbir optimizasyon yok, tüm veri gösteriliyor
        - ✅ **Gerçek boyut** - ses dosyasının tam boyutu korunuyor
        - Desteklenen format: WAV
        - İşlem süresi: Ses uzunluğuna bağlı (GPU/CPU'ya göre)
        - **Matplotlib** grafikleri (tam detay)
        
        ## Grafik Özellikleri
        - **Tam çözünürlük:** Hiçbir veri kaybı yok
        - **Gerçek boyut:** Tüm ses örnekleri gösteriliyor
        
        ## ⚠️ Uyarı
        - Uzun ses dosyaları büyük grafikler oluşturabilir
        - Grafik yükleme süresi ses uzunluğuna bağlıdır
        - Tarayıcı performansı etkilenebilir
        """)
    return info_page

def show_waveform(audio):
    if audio is None:
        return None
    return create_waveform_plot(audio, "Tam Çözünürlük Ses Dalga Formu")

def create_waveform_page():
    with gr.Blocks() as waveform_page:
        gr.Markdown("""
        ## Tam Çözünürlük Ses Dalga Formu Görüntüleyici
        
        Ses dosyanızı yükleyerek **tam çözünürlük** dalga formunu görüntüleyebilirsiniz.
        
        **Özellikler:**
        - ✅ **Sınırsız uzunluk** - herhangi bir uzunlukta ses dosyası
        - ✅ **Tam detay** - hiçbir veri kaybı yok
        - ✅ **Gerçek boyut** - tüm ses örnekleri görüntüleniyor
        - 🎨 **Temiz matplotlib görünümü**
        - 📊 **Yüksek çözünürlük** grafikler
        
        **Desteklenen formatlar:** WAV, MP3, FLAC, M4A, OGG, AIFF
        
        **Not:** Uzun ses dosyaları büyük grafikler oluşturur, yükleme süresi uzayabilir.
        """)
        
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(type="filepath", label="Ses Dosyası Yükleyin (Sınırsız uzunluk)")
                
            with gr.Column():
                waveform_output = gr.Plot(label="Tam Çözünürlük Dalga Formu")
        
        audio_input.change(
            fn=show_waveform,
            inputs=audio_input,
            outputs=waveform_output
        )
        
    return waveform_page

def create_model_page():
    with gr.Blocks() as model_page:
        gr.Markdown("""
        ## DOSE Gürültü Engelleme Modeli
        
        Bu sayfa DOSE (Denoising Diffusion for Ordinary Speech Enhancement) modelini kullanarak 
        gürültülü ses dosyalarını temizler.
        
        **Önemli Notlar:**
        - Model 16kHz örnekleme frekansında çalışır
        - ✅ **Sınırsız ses uzunluğu** - herhangi bir uzunlukta ses işlenir
        - ✅ **Tam çözünürlük** karşılaştırma grafikleri
        - İşlem süresi ses uzunluğu ile doğru orantılıdır
        
        **Mevcut Modeller:**
        - **Original Model :** Makale orijinal model
        - **Turkish Model :** Kendim eğittiğim Türkçe ağırlıklı model

        **Uyarı:** Çok uzun ses dosyaları işlem süresini artırır ve büyük grafikler oluşturur.
        """)
        
        with gr.Row():
            with gr.Column():
                input_audio = gr.Audio(
                    type="filepath", 
                    label="Gürültülü Ses Dosyası (Sınırsız uzunluk)"
                )
                
                # Model seçimi için iki buton
                with gr.Row():
                    model1_btn = gr.Button(
                        "Original Model ile İşle", 
                        variant="primary", 
                        size="lg",
                        scale=1
                    )
                    model2_btn = gr.Button(
                        "Turkish Model ile İşle", 
                        variant="secondary", 
                        size="lg",
                        scale=1
                    )
                
                status_text = gr.Textbox(
                    label="Durum", 
                    interactive=False,
                    lines=4
                )
                
            with gr.Column():
                output_audio = gr.Audio(label="Gürültüsü Engellenmiş Ses")
                output_plot = gr.Plot(label="Tam Çözünürlük Karşılaştırmalı Dalga Formu")
        
        # Model yolları
        model1_path = os.path.join(os.path.dirname(__file__), '..', 'DOSE_model', 'weights', 'original_weight.pt')
        model2_path = os.path.join(os.path.dirname(__file__), '..', 'DOSE_model', 'weights', 'turkish_weight.pt')
        
        # İşleme fonksiyonu - Model parametresi eklendi
        def process_and_compare(audio_path, model_path, model_name):
            if audio_path is None:
                return None, None, "❌ Lütfen bir ses dosyası seçin!"
            
            try:
                # Ses dosyası bilgisi
                y_temp, sr_temp = librosa.load(audio_path, sr=16000)
                duration = len(y_temp) / sr_temp
                samples = len(y_temp)
                
                status_msg = f"🔄 {model_name} ile işleniyor...\nSes uzunluğu: {duration:.1f} saniye\nÖrnek sayısı: {samples:,}\nModel: {model_name}"
                
                # Seçilen model ile işle
                enhanced_path, status = process_audio_with_dose_custom(audio_path, model_path)
                if enhanced_path is None:
                    return None, None, f"❌ {status}"
                
                # Karşılaştırmalı grafik oluştur - TAM BOYUT
                comparison_plot = compare_waveforms(audio_path, enhanced_path)
                
                return enhanced_path, comparison_plot, f"✅ {status}\nKullanılan model: {model_name}"
                
            except Exception as e:
                return None, None, f"❌ Hata: {str(e)}"
        
        # Model 1 buton olayları
        model1_btn.click(
            fn=lambda audio_path: process_and_compare(audio_path, model1_path, "Original Model"),
            inputs=[input_audio],
            outputs=[output_audio, output_plot, status_text],
            show_progress=True
        )
        
        # Model 2 buton olayları
        model2_btn.click(
            fn=lambda audio_path: process_and_compare(audio_path, model2_path, "Turkish Model"),
            inputs=[input_audio],
            outputs=[output_audio, output_plot, status_text],
            show_progress=True
        )
        
    return model_page

def process_audio_with_dose_custom(audio_path, model_path):
    """DOSE modeli ile ses dosyasını işle - Özel model yolu ile"""
    try:
        # Modeli yükle
        if not load_dose_model_custom(model_path):
            return None, "Model yüklenemedi!"
        
        # Ses dosyasını yükle - HİÇ SINIR YOK
        condition, sr = librosa.load(audio_path, sr=16000)
        print(f"Orijinal ses uzunluğu: {len(condition)/sr:.2f} saniye")
        
        condition = torch.tensor(condition).unsqueeze(0).to(device)
        
        # Inference işlemi
        enhanced_audio, output_sr = predict(
            condition=condition, 
            model_dir=model_path, 
            params=base_params,
            device=device,
            fast_sampling=False
        )
        
        # Çıktıyı numpy array'e çevir
        enhanced_audio = enhanced_audio.cpu().numpy()
        
        # Geçici dosya oluştur
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            output_path = tmp_file.name
        
        torchaudio.save(output_path, torch.tensor(enhanced_audio), sample_rate=output_sr)
        
        model_name = os.path.basename(model_path)
        return output_path, f"Gürültü engelleme işlemi başarılı! İşlenen süre: {len(enhanced_audio)/output_sr:.2f} saniye"
        
    except Exception as e:
        return None, f"Hata oluştu: {str(e)}"

def load_dose_model_custom(model_path):
    """DOSE modelini yükle - Özel model yolu ile"""
    global models
    if model_path not in models:
        try:
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=device, weights_only=True)
                model = DOSE(AttrDict(base_params)).to(device)
                model.load_state_dict(checkpoint['model'])
                model.eval()
                models[model_path] = model
                print(f"Model yüklendi: {os.path.basename(model_path)}")
                return True
            else:
                print(f"Model dosyası bulunamadı: {model_path}")
                return False
        except Exception as e:
            print(f"Model yükleme hatası: {e}")
            return False
    return True

# Sayfaları oluştur
info_page = create_info_page()
waveform_page = create_waveform_page()
model_page = create_model_page()

# Çoklu sayfa arayüzünü oluştur
demo = gr.TabbedInterface(
    [info_page, waveform_page, model_page],
    ["📋 Proje Bilgisi", "📊 Dalga Formu", "🤖 Model"],
    title="DOSE - Ses Sistemlerinde Gürültü Engelleme Projesi",
)

if __name__ == "__main__":
    print(f"🔧 Kullanılan cihaz: {device}")
    
    # Gradio uygulamasını başlat
    demo.launch(
        share=True, 
        debug=False,
        server_name="127.0.0.1",
        server_port=7860,
        max_file_size="500mb",  # Büyük dosyalar için
        show_error=True
    )