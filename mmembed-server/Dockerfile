FROM nvcr.io/nvidia/pytorch:23.11-py3

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir \
    "transformers==4.40.2" \
    "accelerate==0.30.1" \
    "bitsandbytes==0.43.1" \
    "sentencepiece" \
    "tiktoken" \
    "einops" \
    "fastapi" \
    "uvicorn" \
    "pydantic"

COPY . .

CMD ["python", "server.py"]
