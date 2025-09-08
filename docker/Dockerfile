# CUDA destekli temel imaj kullanimi
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Sistem paketlerini guncelle ve gerekli paketleri yukle
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Python'u varsayilan yap
RUN ln -s /usr/bin/python3 /usr/bin/python

# Pip'i guncelle
RUN pip3 install --upgrade pip

# Calisma dizinini olustur ve ayarla
WORKDIR /workspace

# Once build dependency'leri yukle
RUN pip install --no-cache-dir Cython numpy

# PESQ kütüphanesini metric.py dosyasındaki talimatlara göre kur
# İlk olarak dev branch'ini dene, başarısız olursa master branch'i kullan
RUN pip install --no-cache-dir https://github.com/ludlows/python-pesq/archive/refs/heads/dev.zip || \
    pip install --no-cache-dir https://github.com/ludlows/python-pesq/archive/master.zip

# PySTOI kütüphanesini yükle
RUN pip install --no-cache-dir pystoi

# Tum requirements.txt dosyasindaki paketleri yukle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Proje dosyalarını konteynıra kopyala
COPY . .

# Jupyter Notebook icin port ac
EXPOSE 8888

# Gradio icin port ac
EXPOSE 7860

# Varsayilan komut olarak Jupyter Notebook'u calistir
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root"]