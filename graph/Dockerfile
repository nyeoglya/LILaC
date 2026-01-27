FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "requests" \
    "beautifulsoup4" \
    "pysbd"

COPY . .

CMD ["python", "main.py"]
