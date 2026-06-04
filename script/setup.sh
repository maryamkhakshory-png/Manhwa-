#!/bin/bash

echo "🧠 Installing Arvan AI dependencies..."

# System packages
pkg install -y \
    clang cmake make git \
    curl libcurl openssl \
    opencv ffmpeg \
    tesseract tesseract-data-eng tesseract-data-fas \
    eigen nlohmann-json \
    python python-pip

# Python packages for training
pip install torch transformers datasets openai-whisper

# Download models
echo "📥 Downloading models..."
mkdir -p models data

# Download pre-trained weights
wget -O models/yolov4.weights https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
wget -O models/whisper_tiny.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin

echo "✅ Setup complete!"
