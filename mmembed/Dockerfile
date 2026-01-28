FROM nvcr.io/nvidia/pytorch:23.11-py3

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    "transformers==4.41.2" \
    "peft==0.10.0" \
    "accelerate" \
    "datasets"


RUN pip install --no-cache-dir \
    "fastapi" \
    "uvicorn" \
    "pydantic"

COPY . .
