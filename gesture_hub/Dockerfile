# Use official Python 3.11 slim image (Debian-based, works well with AI wheels)
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Required for PyQt6 to find the X11 platform plugin
ENV QT_QPA_PLATFORM=xcb
ENV DISPLAY=:0

# Install heavy system dependencies for OpenCV, PyQt6, and Pygame Audio
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-xfixes0 \
    libxcb-shape0 \
    libxcb-cursor0 \
    libasound2 \
    pulseaudio \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the gesture hub application
COPY . .

# Command to run the application
CMD ["python", "main.py"]
