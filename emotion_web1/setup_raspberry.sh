#!/bin/bash

# Setup script for Raspberry Pi Emotion Detection System
# Run with: chmod +x setup_raspberry.sh && ./setup_raspberry.sh

echo "Setting up Raspberry Pi for Emotion Detection System..."

# Update system
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
echo "Installing system dependencies..."
sudo apt install -y \
    python3-pip \
    python3-picamera2 \
    python3-libcamera \
    python3-opencv \
    portaudio19-dev \
    python3-pyaudio \
    libatlas-base-dev \
    python3-tk \
    python3-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libavresample-dev \
    libavfilter-dev

# Install Python packages
echo "Installing Python packages..."
pip3 install --upgrade pip
pip3 install -r requirements_raspberry.txt

# Enable camera interface
echo "Enabling camera interface..."
sudo raspi-config nonint do_camera 0

# Enable I2C for potential sensor integration
echo "Enabling I2C interface..."
sudo raspi-config nonint do_i2c 0

# Configure audio
echo "Configuring audio..."
# Set default audio output to headphone jack or HDMI
# Uncomment the appropriate line based on your setup
# amixer cset numid=3 1  # Headphone jack
# amixer cset numid=3 2  # HDMI

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p uploads
mkdir -p data_emotion
mkdir -p models

# Set permissions
echo "Setting permissions..."
chmod +x test2.py

# Configure camera settings
echo "Configuring camera settings..."
# Add camera configuration to /boot/config.txt if needed
# echo "camera_auto_detect=1" | sudo tee -a /boot/config.txt

echo "Setup completed!"
echo ""
echo "To run the emotion detection system:"
echo "1. Make sure your camera is connected"
echo "2. Run: python3 test2.py"
echo "3. Or run in background: nohup python3 test2.py &"
echo ""
echo "To run the Flask server:"
echo "1. Run: python3 app.py"
echo "2. Or run in background: nohup python3 app.py &"
echo ""
echo "Note: If running headless, you may need to comment out cv2.imshow() lines in test2.py" 