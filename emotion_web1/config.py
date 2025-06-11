# Configuration file for Raspberry Pi Emotion Detection System

# Server Configuration
FLASK_SERVER = 'http://localhost:5000'

# Audio Configuration
AUDIO_RATE = 16000
AUDIO_DURATION = 3  # seconds
SEND_INTERVAL = 3  # Interval between audio recordings

# Image Configuration
IMAGE_SEND_FREQUENCY = 0.08  # seconds (12.5 FPS)
CAMERA_FPS = 30  # Camera frame rate

# Threading Configuration
IMAGE_QUEUE_SIZE = 50  # Buffer size for images
AUDIO_QUEUE_SIZE = 10  # Buffer size for audio
WORKER_SLEEP_EMPTY = 0.01  # Sleep time when queue is empty
WORKER_SLEEP_ERROR = 0.1  # Sleep time on error

# Network Configuration
IMAGE_TIMEOUT = 2  # seconds
AUDIO_TIMEOUT = 10  # seconds
SIGNAL_TIMEOUT = 5  # seconds
MAX_RETRY_ATTEMPTS = 2  # for images
AUDIO_RETRY_ATTEMPTS = 3  # for audio

# Camera Configuration
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FORMAT = "RGB888"

# Face Detection Configuration
FACE_SCALE_FACTOR = 1.1
FACE_MIN_NEIGHBORS = 5
FACE_SIZE = (48, 48)

# Performance Tuning
ENABLE_QUEUE_MONITORING = True  # Log queue status
ENABLE_PERFORMANCE_LOGGING = False  # Detailed performance logs
MAX_CONCURRENT_REQUESTS = 5  # Limit concurrent HTTP requests

# Headless Mode Configuration
HEADLESS_MODE = False  # Set to True for headless operation
DISABLE_DISPLAY = True  # Disable cv2.imshow() calls

# Logging Configuration
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# Error Handling
AUTO_RESTART_ON_ERROR = True  # Restart system on unexpected errors
RESTART_DELAY = 10  # seconds to wait before restart

# Development Mode
DEBUG_MODE = False  # Enable additional debug information 