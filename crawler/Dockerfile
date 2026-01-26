FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "Wikipedia-API" \
    "pandas" \
    "requests" \
    "beautifulsoup4" \
    "wikitextparser"

COPY . .

CMD ["python", "main.py"]
