# Raspberry Pi Emotion Detection System

Hệ thống nhận diện cảm xúc chạy trên Raspberry Pi sử dụng Picamera2 và xử lý âm thanh real-time.

## Yêu cầu hệ thống

- Raspberry Pi 3/4/5 (khuyến nghị Pi 4 với 4GB RAM)
- Camera Module (Pi Camera v2/v3)
- Microphone USB hoặc built-in
- Thẻ SD 16GB+ (Class 10)
- Kết nối mạng (WiFi/Ethernet)

## Cài đặt

### 1. Cài đặt nhanh (Tự động)

```bash
# Clone hoặc copy code vào Raspberry Pi
cd /path/to/emotion_web1

# Chạy script cài đặt tự động
chmod +x setup_raspberry.sh
./setup_raspberry.sh
```

### 2. Cài đặt thủ công

```bash
# Cập nhật hệ thống
sudo apt update && sudo apt upgrade -y

# Cài đặt dependencies hệ thống
sudo apt install -y \
    python3-pip \
    python3-picamera2 \
    python3-libcamera \
    python3-opencv \
    portaudio19-dev \
    python3-pyaudio \
    libatlas-base-dev \
    python3-tk

# Cài đặt Python packages
pip3 install -r requirements_raspberry.txt

# Bật camera interface
sudo raspi-config nonint do_camera 0

# Khởi động lại
sudo reboot
```

## Cấu hình

### 1. Cấu hình Camera

```bash
# Kiểm tra camera
vcgencmd get_camera

# Test camera
libcamera-still -o test.jpg
```

### 2. Cấu hình Audio

```bash
# Kiểm tra audio devices
arecord -l

# Test microphone
arecord -D hw:1,0 -f S16_LE -r 16000 -c 1 test.wav
```

### 3. Cấu hình Network

Đảm bảo Raspberry Pi có thể kết nối đến Flask server:

```bash
# Test kết nối
ping localhost
curl http://localhost:5000
```

## Sử dụng

### 1. Chạy với màn hình (Desktop Mode)

```bash
# Chạy Flask server
python3 app.py

# Trong terminal khác, chạy client
python3 test2.py
```

### 2. Chạy headless (Không có màn hình)

```bash
# Chạy Flask server trong background
nohup python3 app.py > server.log 2>&1 &

# Chạy client headless
nohup python3 test2_headless.py > client.log 2>&1 &
```

### 3. Chạy như service

Tạo systemd service để tự động khởi động:

```bash
# Tạo service file
sudo nano /etc/systemd/system/emotion-detection.service
```

```ini
[Unit]
Description=Emotion Detection System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/emotion_web1
ExecStart=/usr/bin/python3 test2_headless.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable và start service
sudo systemctl enable emotion-detection.service
sudo systemctl start emotion-detection.service

# Kiểm tra status
sudo systemctl status emotion-detection.service
```

## Cấu hình nâng cao

### 1. Tối ưu hiệu suất

```bash
# Tăng GPU memory
echo "gpu_mem=128" | sudo tee -a /boot/config.txt

# Tối ưu CPU governor
echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### 2. Cấu hình camera nâng cao

```bash
# Thêm vào /boot/config.txt
camera_auto_detect=1
camera_interface=1
```

### 3. Monitoring và Logging

```bash
# Xem logs real-time
tail -f client.log

# Kiểm tra CPU/Memory usage
htop

# Kiểm tra temperature
vcgencmd measure_temp
```

## Troubleshooting

### 1. Camera không hoạt động

```bash
# Kiểm tra camera
vcgencmd get_camera

# Kiểm tra permissions
ls -la /dev/video*

# Restart camera service
sudo systemctl restart camera.service
```

### 2. Audio không hoạt động

```bash
# Kiểm tra audio devices
arecord -l
aplay -l

# Test microphone
arecord -D hw:1,0 -f S16_LE -r 16000 -c 1 test.wav
aplay test.wav
```

### 3. Network issues

```bash
# Kiểm tra kết nối
ping localhost
curl http://localhost:5000

# Kiểm tra firewall
sudo ufw status
```

### 4. Performance issues

```bash
# Kiểm tra CPU usage
top

# Kiểm tra memory
free -h

# Kiểm tra temperature
vcgencmd measure_temp
```

## Files quan trọng

- `test2.py` - Client với display (desktop mode)
- `test2_headless.py` - Client không có display (headless mode)
- `app.py` - Flask server
- `setup_raspberry.sh` - Script cài đặt tự động
- `requirements_raspberry.txt` - Dependencies Python

## Cấu hình mạng

Để kết nối từ máy khác, thay đổi `FLASK_SERVER` trong `test2.py`:

```python
# Thay localhost bằng IP của Raspberry Pi
FLASK_SERVER = 'http://192.168.1.100:5000'
```

## Monitoring

### 1. System monitoring

```bash
# CPU temperature
watch -n 1 vcgencmd measure_temp

# Memory usage
watch -n 1 free -h

# Network traffic
iftop
```

### 2. Application monitoring

```bash
# View logs
tail -f client.log
tail -f server.log

# Check processes
ps aux | grep python
```

## Backup và Recovery

```bash
# Backup data
tar -czf emotion_data_backup.tar.gz data_emotion/

# Backup configuration
cp /boot/config.txt config_backup.txt
```

## Support

Nếu gặp vấn đề, kiểm tra:
1. Logs trong `client.log` và `server.log`
2. System logs: `journalctl -u emotion-detection.service`
3. Camera status: `vcgencmd get_camera`
4. Network connectivity: `ping` và `curl` 