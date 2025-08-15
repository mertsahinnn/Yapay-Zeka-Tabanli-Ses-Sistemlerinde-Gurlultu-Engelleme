import gradio as gr
import librosa
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go



def create_info_page():
    with gr.Blocks() as info_page:
        gr.Markdown("""
        # Ses Sistemlerinde Gürültü Engelleme Projesi
        
        Bu proje, ses sistemlerinde gürültü engelleme üzerine yapılan bir çalışmadır.
        
        ## Proje Hakkında
        - Yapay zeka tabanlı gürültü engelleme
        - Ses kalitesini iyileştirme
        - Gerçek zamanlı ses işleme
        
        ## Nasıl Kullanılır?
        1. "Dalga Formu" sekmesine geçin
        2. Bir ses dosyası yükleyin
        3. Dalga formunu inceleyin
        4. "Model" sekmesinde gürültü engelleme işlemini gerçekleştirin
        """)
    return info_page

def show_waveform(audio):
    # Ses dosyasını yükle
    y, sr = librosa.load(audio)
    
    time = np.linspace(0, len(y) / sr, num=len(y))
    fig = make_subplots(rows=2, cols=1, subplot_titles=["Dalga Formu", "Spektrum"])

    fig.add_trace(go.Scatter(x=time, y=y, mode='lines', name='Dalga Formu'), row=1, col=1)
    fig.update_xaxes(title_text="Zaman (saniye)", row=1, col=1)
    fig.update_yaxes(title_text="Genlik", row=1, col=1)

    # Spektrumu oluştur
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = go.Heatmap(
        z=D,
        x=librosa.times_like(D, sr=sr),
        y=librosa.fft_frequencies(sr=sr),
        colorscale= "Jet",
        colorbar=dict(title='dB')
    )

    fig.add_trace(img, row=2, col=1)
    fig.update_xaxes(title_text="Zaman (saniye)", row=2, col=1)
    fig.update_yaxes(title_text="Frekans (Hz)", type =  "log" , row=2, col=1)

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
        gr.Markdown("## Gürültü Engelleme Modeli")
        with gr.Row():
            with gr.Column():
                input_audio = gr.Audio(type="filepath", label="Gürültülü Ses Dosyası")
                process_btn = gr.Button("Gürültüyü Engelle")
            with gr.Column():
                output_audio = gr.Audio(label="Gürültüsü Engellenmiş Ses")
                output_plot = gr.Plot(label="Karşılaştırmalı Dalga Formu")
    return model_page




# Sayfaları oluştur
info_page = create_info_page()
waveform_page = create_waveform_page()
model_page = create_model_page()

# Çoklu sayfa arayüzünü oluştur ve başlat
demo = gr.TabbedInterface(
    [info_page, waveform_page, model_page],
    ["Proje Bilgisi", "Dalga Formu", "Model"],
    title="Ses Sistemlerinde Gürültü Engelleme",
)

if __name__ == "__main__":
    demo.launch()

