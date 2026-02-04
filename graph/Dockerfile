FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    "tqdm"

RUN pip install --no-cache-dir \
    "numpy" \
    "requests" \
    "pysbd"

RUN pip install --no-cache-dir \
    "fastapi" \
    "uvicorn" \
    "pydantic"

COPY . .
