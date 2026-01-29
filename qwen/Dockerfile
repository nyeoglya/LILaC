FROM nvcr.io/nvidia/pytorch:23.11-py3

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir \
    "torch==2.4.1" \
    "torchvision==0.19.1" \
    "transformers>=4.48.0" \
    "accelerate>=1.1.0" \
    "bitsandbytes>=0.45.0" \
    "sentencepiece" \
    "tiktoken" \
    "einops"

RUN pip install --no-cache-dir \
    "fastapi" \
    "uvicorn" \
    "pydantic"

COPY . .
