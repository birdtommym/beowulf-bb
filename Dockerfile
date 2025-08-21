# Dockerfile
FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# System deps for OpenCV + Tesseract OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Generate app code & install Python deps
COPY beowulf_bootstrap_v2.py .
RUN python beowulf_bootstrap_v2.py
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 10000
CMD ["bash","-lc","streamlit run beowulf_app/app.py --server.port ${PORT:-10000} --server.headless true"]
