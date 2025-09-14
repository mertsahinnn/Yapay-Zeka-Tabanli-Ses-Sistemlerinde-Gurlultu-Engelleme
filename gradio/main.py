import gradio as gr
import librosa
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import torch
import torchaudio
import sys
import os

# DOSE model için gerekli importlar
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'DOSE_model'))
from params import AttrDict, params as base_params
from model import DOSE
from inference import predict

# Model global değişkenleri
models = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'DOSE_model', 'weights', 'weights.pt')

def load_dose_model():
    """DOSE modelini yükle"""
    global models
    if MODEL_PATH not in models:
        try:
            if os.path.exists(MODEL_PATH):
                checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
                model = DOSE(AttrDict(base_params)).to(device)
                model.load_state_dict(checkpoint['model'])
                model.eval()
                models[MODEL_PATH] = model
                return True
            else:
                print(f"Model dosyası bulunamadı: {MODEL_PATH}")
                return False
        except Exception as e:
            print(f"Model yükleme hatası: {e}")
            return False
    return True

def process_audio_with_dose(audio_path):
    """DOSE modeli ile ses dosyasını işle"""
    try:
        # Modeli yükle
        if not load_dose_model():
            return None, "Model yüklenemedi!"
        
        # Ses dosyasını yükle
        condition, sr = librosa.load(audio_path, sr=16000)
        condition = torch.tensor(condition).unsqueeze(0).to(device)
        
        # Inference işlemi
        enhanced_audio, output_sr = predict(
            condition=condition, 
            model_dir=MODEL_PATH, 
            params=base_params,
            device=device,
            fast_sampling=False
        )
        
        # Çıktıyı numpy array'e çevir
        enhanced_audio = enhanced_audio.cpu().numpy()
        
        # Geçici dosya oluştur
        output_path = "temp_enhanced.wav"
        torchaudio.save(output_path, torch.tensor(enhanced_audio), sample_rate=output_sr)
        
        return output_path, "Gürültü engelleme işlemi başarılı!"
        
    except Exception as e:
        return None, f"Hata oluştu: {str(e)}"

def compare_waveforms(original_path, enhanced_path):
    """Orijinal ve işlenmiş ses dosyalarını karşılaştır"""
    try:
        # Orijinal ses
        y_orig, sr_orig = librosa.load(original_path, sr=16000)
        time_orig = np.linspace(0, len(y_orig) / sr_orig, num=len(y_orig))
        
        # İşlenmiş ses
        y_enh, sr_enh = librosa.load(enhanced_path, sr=16000)
        time_enh = np.linspace(0, len(y_enh) / sr_enh, num=len(y_enh))
        
        # Karşılaştırmalı grafik oluştur
        fig = make_subplots(
            rows=2, cols=1, 
            subplot_titles=["Orijinal Ses", "Gürültüsü Engellenmiş Ses"]
        )
        
        # Orijinal dalga formu
        fig.add_trace(
            go.Scatter(x=time_orig, y=y_orig, mode='lines', name='Orijinal', line=dict(color='red')), 
            row=1, col=1
        )
        
        # İşlenmiş dalga formu
        fig.add_trace(
            go.Scatter(x=time_enh, y=y_enh, mode='lines', name='İşlenmiş', line=dict(color='blue')), 
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Zaman (saniye)", row=1, col=1)
        fig.update_xaxes(title_text="Zaman (saniye)", row=2, col=1)
        fig.update_yaxes(title_text="Genlik", row=1, col=1)
        fig.update_yaxes(title_text="Genlik", row=2, col=1)
        
        fig.update_layout(height=600, showlegend=False, title_text="Ses Karşılaştırması")
        
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
        - Gerçek zamanlı ses işleme
        
        ## Nasıl Kullanılır?
        1. "Dalga Formu" sekmesine geçin
        2. Bir ses dosyası yükleyin
        3. Dalga formunu inceleyin
        4. "Model" sekmesinde gürültü engelleme işlemini gerçekleştirin
        
        ## Kullanılan Model
        - **DOSE (Denoising Diffusion for Ordinary Speech Enhancement)**
        - 16kHz örnekleme frekansında çalışır
        - Diffusion-based denoising algoritması kullanır
        """)
    return info_page

def show_waveform(audio):
    if audio is None:
        return None
        
    # Ses dosyasını yükle
    y, sr = librosa.load(audio)
    
    time = np.linspace(0, len(y) / sr, num=len(y))
    fig = make_subplots(rows=2, cols=1, subplot_titles=["Dalga Formu", "Spektrogram"])

    fig.add_trace(go.Scatter(x=time, y=y, mode='lines', name='Dalga Formu'), row=1, col=1)
    fig.update_xaxes(title_text="Zaman (saniye)", row=1, col=1)
    fig.update_yaxes(title_text="Genlik", row=1, col=1)

    # Spektrumu oluştur
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = go.Heatmap(
        z=D,
        x=librosa.times_like(D, sr=sr),
        y=librosa.fft_frequencies(sr=sr),
        colorscale="Jet",
        colorbar=dict(title='dB')
    )

    fig.add_trace(img, row=2, col=1)
    fig.update_xaxes(title_text="Zaman (saniye)", row=2, col=1)
    fig.update_yaxes(title_text="Frekans (Hz)", type="log", row=2, col=1)

    fig.update_layout(height=800, showlegend=False, title_text="Ses Dalga Formu ve Spektrumu")

    return fig

def create_waveform_page():
    with gr.Blocks() as waveform_page:
        gr.Markdown("## Ses Dalga Formu Görüntüleyici")
        with gr.Row():
            audio_input = gr.Audio(type="filepath", label="Ses Dosyası Yükleyin")
            waveform_output = gr.Plot(label="Dalga Formu")
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
        
        **Not:** Model 16kHz örnekleme frekansında çalışır, ses dosyaları otomatik olarak bu frekansa dönüştürülür.
        """)
        
        with gr.Row():
            with gr.Column():
                input_audio = gr.Audio(type="filepath", label="Gürültülü Ses Dosyası")
                process_btn = gr.Button("Gürültüyü Engelle", variant="primary")
                status_text = gr.Textbox(label="Durum", interactive=False)
                
            with gr.Column():
                output_audio = gr.Audio(label="Gürültüsü Engellenmiş Ses")
                output_plot = gr.Plot(label="Karşılaştırmalı Dalga Formu")
        
        # İşleme fonksiyonu
        def process_and_compare(audio_path):
            if audio_path is None:
                return None, None, "Lütfen bir ses dosyası seçin!"
            
            # DOSE ile işle
            enhanced_path, status = process_audio_with_dose(audio_path)
            
            if enhanced_path is None:
                return None, None, status
            
            # Karşılaştırmalı grafik oluştur
            comparison_plot = compare_waveforms(audio_path, enhanced_path)
            
            return enhanced_path, comparison_plot, status
        
        process_btn.click(
            fn=process_and_compare,
            inputs=[input_audio],
            outputs=[output_audio, output_plot, status_text]
        )
        
    return model_page

# Sayfaları oluştur
info_page = create_info_page()
waveform_page = create_waveform_page()
model_page = create_model_page()

# Çoklu sayfa arayüzünü oluştur ve başlat
demo = gr.TabbedInterface(
    [info_page, waveform_page, model_page],
    ["Proje Bilgisi", "Dalga Formu", "Model"],
    title="DOSE - Ses Sistemlerinde Gürültü Engelleme",
)

if __name__ == "__main__":
    print(f"Kullanılan cihaz: {device}")
    print(f"Model yolu: {MODEL_PATH}")
    demo.launch(share=False, debug=True)

