# Python 3.11 Lightweight Slim Image
FROM python:3.11-slim

# System dependencies for Audio processing (FFmpeg, PortAudio) & C compilers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    portaudio19-dev \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY . .

# Expose FastAPI Port
EXPOSE 8000

# Start FastAPI application without reload in production/docker to prevent startup hang on HF model fetch
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-reload"]