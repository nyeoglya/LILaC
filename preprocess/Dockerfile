FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    "requests" \
    "beautifulsoup4"

RUN pip install --no-cache-dir \
    transformers \
    opencv-python-headless \
    scikit-learn \
    pillow \
    numpy \
    tqdm

COPY . .
