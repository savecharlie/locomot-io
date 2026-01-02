#!/usr/bin/env python3
"""
Simple static file servers for portrait painter and sand buddy
"""
import http.server
import socketserver
import threading
import os
import shutil
import functools

# Ensure painter directory exists
PAINTER_DIR = '/home/ivy/locomot-io/painter'
SANDBOX_DIR = '/home/ivy/locomot-io/sandbox'

os.makedirs(PAINTER_DIR, exist_ok=True)

# Copy portrait painter to its directory
shutil.copy('/home/ivy/live_portrait_painter.html', f'{PAINTER_DIR}/index.html')

# Copy Sand Buddy to its directory
os.makedirs(SANDBOX_DIR, exist_ok=True)
shutil.copy('/home/ivy/tamagotchi.html', f'{SANDBOX_DIR}/index.html')

def make_handler(directory):
    """Create a handler class for a specific directory"""
    class DirectoryHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)
        def log_message(self, format, *args):
            pass  # Suppress logging
    return DirectoryHandler

class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

def serve_directory(directory, port):
    handler = make_handler(directory)
    with ReusableTCPServer(("0.0.0.0", port), handler) as httpd:
        print(f"Serving {directory} on port {port}")
        httpd.serve_forever()

if __name__ == '__main__':
    # Painter on 8766
    t1 = threading.Thread(target=serve_directory, args=(PAINTER_DIR, 8766), daemon=True)
    t1.start()

    # Sand Buddy on 8767
    t2 = threading.Thread(target=serve_directory, args=(SANDBOX_DIR, 8767), daemon=True)
    t2.start()

    print("🎨 Portrait Painter: http://localhost:8766")
    print("🏖️ Sand Buddy: http://localhost:8767")

    # Keep running
    import time
    while True:
        time.sleep(3600)
