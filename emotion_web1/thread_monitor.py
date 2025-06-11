import threading
import time
import logging
from queue import Queue
import psutil
import os

class ThreadMonitor:
    """Monitor thread performance and health"""
    
    def __init__(self, image_queue, audio_queue):
        self.image_queue = image_queue
        self.audio_queue = audio_queue
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start the monitoring thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logging.info("Thread monitoring started")
        
    def stop_monitoring(self):
        """Stop the monitoring thread"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        logging.info("Thread monitoring stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                self._log_system_stats()
                self._log_queue_stats()
                self._log_thread_stats()
                time.sleep(5)  # Monitor every 5 seconds
            except Exception as e:
                logging.error(f"Error in monitoring: {e}")
                time.sleep(1)
                
    def _log_system_stats(self):
        """Log system resource usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        logging.info(f"System Stats - CPU: {cpu_percent}%, "
                    f"Memory: {memory.percent}%, "
                    f"Disk: {disk.percent}%")
                    
    def _log_queue_stats(self):
        """Log queue statistics"""
        image_queue_size = self.image_queue.qsize()
        audio_queue_size = self.audio_queue.qsize()
        
        logging.info(f"Queue Stats - Image: {image_queue_size}/50, "
                    f"Audio: {audio_queue_size}/10")
                    
        # Warn if queues are getting full
        if image_queue_size > 40:
            logging.warning(f"Image queue is getting full: {image_queue_size}/50")
        if audio_queue_size > 8:
            logging.warning(f"Audio queue is getting full: {audio_queue_size}/10")
            
    def _log_thread_stats(self):
        """Log thread statistics"""
        current_thread = threading.current_thread()
        active_threads = threading.active_count()
        
        logging.info(f"Thread Stats - Active: {active_threads}, "
                    f"Current: {current_thread.name}")
                    
        # Log all active threads
        for thread in threading.enumerate():
            if thread.is_alive():
                logging.debug(f"Active thread: {thread.name} (daemon: {thread.daemon})")

class PerformanceTracker:
    """Track performance metrics"""
    
    def __init__(self):
        self.metrics = {
            'images_sent': 0,
            'audio_sent': 0,
            'errors': 0,
            'start_time': time.time()
        }
        self.lock = threading.Lock()
        
    def increment_images_sent(self):
        """Increment image sent counter"""
        with self.lock:
            self.metrics['images_sent'] += 1
            
    def increment_audio_sent(self):
        """Increment audio sent counter"""
        with self.lock:
            self.metrics['audio_sent'] += 1
            
    def increment_errors(self):
        """Increment error counter"""
        with self.lock:
            self.metrics['errors'] += 1
            
    def get_stats(self):
        """Get current statistics"""
        with self.lock:
            elapsed_time = time.time() - self.metrics['start_time']
            images_per_second = self.metrics['images_sent'] / elapsed_time if elapsed_time > 0 else 0
            audio_per_minute = (self.metrics['audio_sent'] / elapsed_time) * 60 if elapsed_time > 0 else 0
            
            return {
                'elapsed_time': elapsed_time,
                'images_sent': self.metrics['images_sent'],
                'audio_sent': self.metrics['audio_sent'],
                'errors': self.metrics['errors'],
                'images_per_second': images_per_second,
                'audio_per_minute': audio_per_minute
            }
            
    def log_stats(self):
        """Log current statistics"""
        stats = self.get_stats()
        logging.info(f"Performance Stats - "
                    f"Time: {stats['elapsed_time']:.1f}s, "
                    f"Images: {stats['images_sent']} ({stats['images_per_second']:.2f}/s), "
                    f"Audio: {stats['audio_sent']} ({stats['audio_per_minute']:.2f}/min), "
                    f"Errors: {stats['errors']}")

def create_monitoring_decorator(performance_tracker, metric_type):
    """Create a decorator to track performance metrics"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                if metric_type == 'image':
                    performance_tracker.increment_images_sent()
                elif metric_type == 'audio':
                    performance_tracker.increment_audio_sent()
                return result
            except Exception as e:
                performance_tracker.increment_errors()
                raise e
        return wrapper
    return decorator 