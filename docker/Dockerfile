# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages directly without a requirements file
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    matplotlib \
    seaborn \
    pillow \
    tqdm \
    torch>=2.3 \
    torchvision \
    torchaudio \
    transformers>=4.46 \
    tokenizers \
    datasets>=2.20 \
    accelerate>=1.0 \
    evaluate \
    huggingface-hub \
    safetensors \
    sentencepiece \
    peft>=0.13 \
    trl>=0.16 \
    diffusers \
    timm \
    nltk \
    sentence-transformers \
    'scikit-multilearn; python_version < "3.13"' \
    tensorboard \
    wandb \
    python-dotenv \
    pyyaml \
    rich \
    bitsandbytes>=0.43 \
    jupyter \
    jupyterlab \
    ipywidgets \
    ipykernel \
    mlflow>=2.14 \
    protobuf

# Copy application code
COPY . .

# Create a non-root user for security
RUN useradd -m runner
USER runner

# Default command
CMD ["python", "-v"]
