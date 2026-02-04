FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf2.0-0 \
    libffi-dev \
    shared-mime-info \
    libxml2 \
    libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    ultralytics

RUN pip install --no-cache-dir \
    pillow==10.0.1 \
    numpy==1.26.4 \
    opencv-python-headless==4.10.0.84 \
    cairosvg \
    tqdm \
    requests \
    beautifulsoup4 \
    scikit-learn

COPY . .
