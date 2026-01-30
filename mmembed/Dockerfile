FROM nvcr.io/nvidia/pytorch:23.11-py3

WORKDIR /app

RUN apt update
RUN apt install -y libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    "transformers==4.41.2" \
    "peft==0.10.0" \
    "accelerate==1.12.0" \
    "safetensors==0.7.0" \
    "tokenizers==0.19" \
    "datasets>=2.18.0,<3.0.0"

RUN pip install --no-cache-dir \
    "cairosvg"

RUN pip install --no-cache-dir \
    "fastapi" \
    "uvicorn" \
    "pydantic"

COPY . .
