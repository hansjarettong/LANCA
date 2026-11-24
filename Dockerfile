# Start from an official NVIDIA CUDA base image
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install essentials like Python 3.10, git, and system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    screen \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy your requirements file first to leverage Docker's layer caching
COPY requirements.txt .

# Install Python packages
RUN pip3 install --no-cache-dir -r requirements.txt

# Keep the container running
CMD ["tail", "-f", "/dev/null"]