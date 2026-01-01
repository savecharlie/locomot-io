#!/usr/bin/env python3
"""
Clip server - receives game clips and tracks player activity
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import os
import json
import requests
from datetime import datetime
import re

CLIP_DIR = '/home/ivy/locomot-io/clips'
PLAYER_HISTORY_FILE = '/home/ivy/locomot-io/player_history.json'
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

os.makedirs(CLIP_DIR, exist_ok=True)

# Load or initialize player history
def load_player_history():
    try:
        if os.path.exists(PLAYER_HISTORY_FILE):
            with open(PLAYER_HISTORY_FILE, 'r') as f:
                return json.load(f)
    except:
        pass
    return {'players': [], 'sessions': []}

def save_player_history(data):
    with open(PLAYER_HISTORY_FILE, 'w') as f:
        json.dump(data, f, indent=2)

player_history = load_player_history()

class ClipHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if '/api/players' in self.path:
            self.handle_get_players()
        else:
            self.send_error(404)

    def do_POST(self):
        if '/api/clip' in self.path:
            self.handle_clip()
        elif '/api/player/join' in self.path:
            self.handle_player_join()
        elif '/api/player/leave' in self.path:
            self.handle_player_leave()
        elif '/api/player/activity' in self.path:
            self.handle_player_activity()
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()

    def handle_get_players(self):
        """Return recent player history"""
        global player_history
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Type', 'application/json')
        self.end_headers()

        # Return last 50 sessions and unique players
        response = {
            'players': player_history.get('players', [])[-50:],
            'sessions': player_history.get('sessions', [])[-100:],
            'timestamp': datetime.now().isoformat()
        }
        self.wfile.write(json.dumps(response).encode())

    def handle_player_join(self):
        """Record player joining"""
        global player_history
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)

            player_name = data.get('name', 'Unknown')
            player_id = data.get('id', '')

            # Add to sessions
            session = {
                'name': player_name,
                'id': player_id,
                'joined': datetime.now().isoformat(),
                'left': None,
                'maxLength': 0,
                'kills': 0,
                'score': 0
            }
            player_history['sessions'].append(session)

            # Update unique players list
            existing = next((p for p in player_history['players'] if p['name'] == player_name), None)
            if existing:
                existing['lastSeen'] = datetime.now().isoformat()
                existing['sessions'] = existing.get('sessions', 0) + 1
            else:
                player_history['players'].append({
                    'name': player_name,
                    'firstSeen': datetime.now().isoformat(),
                    'lastSeen': datetime.now().isoformat(),
                    'sessions': 1,
                    'totalKills': 0,
                    'bestScore': 0
                })

            # Keep only last 100 sessions
            player_history['sessions'] = player_history['sessions'][-100:]
            save_player_history(player_history)

            print(f'👋 Player joined: {player_name}')

            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        except Exception as e:
            print(f'Error in player_join: {e}')
            self.send_error(500, str(e))

    def handle_player_leave(self):
        """Record player leaving with stats"""
        global player_history
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)

            player_name = data.get('name', 'Unknown')
            max_length = data.get('maxLength', 0)
            kills = data.get('kills', 0)
            score = data.get('score', 0)

            # Update last session
            for session in reversed(player_history['sessions']):
                if session['name'] == player_name and session['left'] is None:
                    session['left'] = datetime.now().isoformat()
                    session['maxLength'] = max_length
                    session['kills'] = kills
                    session['score'] = score
                    break

            # Update player stats
            for player in player_history['players']:
                if player['name'] == player_name:
                    player['lastSeen'] = datetime.now().isoformat()
                    player['totalKills'] = player.get('totalKills', 0) + kills
                    player['bestScore'] = max(player.get('bestScore', 0), score)
                    break

            save_player_history(player_history)

            print(f'👋 Player left: {player_name} (Score: {score}, Kills: {kills})')

            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        except Exception as e:
            print(f'Error in player_leave: {e}')
            self.send_error(500, str(e))

    def handle_player_activity(self):
        """Record periodic player activity (heartbeat with stats)"""
        global player_history
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)

            player_name = data.get('name', 'Unknown')

            # Update last seen
            for player in player_history['players']:
                if player['name'] == player_name:
                    player['lastSeen'] = datetime.now().isoformat()
                    break

            # Update current session
            for session in reversed(player_history['sessions']):
                if session['name'] == player_name and session['left'] is None:
                    session['maxLength'] = max(session.get('maxLength', 0), data.get('length', 0))
                    session['score'] = max(session.get('score', 0), data.get('score', 0))
                    break

            save_player_history(player_history)

            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        except Exception as e:
            self.send_error(500, str(e))

    def handle_clip(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)

            # Parse multipart manually
            content_type = self.headers.get('Content-Type', '')
            boundary = content_type.split('boundary=')[-1].encode()

            parts = body.split(b'--' + boundary)

            video_data = None
            clip_type = 'highlight'
            score = '0'
            player = 'Unknown'

            for part in parts:
                if b'name="video"' in part:
                    header_end = part.find(b'\r\n\r\n')
                    if header_end > 0:
                        video_data = part[header_end + 4:].rstrip(b'\r\n--')
                elif b'name="type"' in part:
                    match = re.search(b'\r\n\r\n(.+?)\r\n', part)
                    if match:
                        clip_type = match.group(1).decode()
                elif b'name="score"' in part:
                    match = re.search(b'\r\n\r\n(.+?)\r\n', part)
                    if match:
                        score = match.group(1).decode()
                elif b'name="player"' in part:
                    match = re.search(b'\r\n\r\n(.+?)\r\n', part)
                    if match:
                        player = match.group(1).decode()

            if not video_data:
                self.send_error(400, 'No video data')
                return

            # Save file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'{timestamp}_{player}_{clip_type}_{score}pts.webm'
            filepath = os.path.join(CLIP_DIR, filename)

            with open(filepath, 'wb') as f:
                f.write(video_data)

            print(f'🎬 Saved: {filename} ({len(video_data)//1024}KB)')

            # Send to Telegram
            self.send_to_telegram(filepath, player, clip_type, score)

            # Response
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        except Exception as e:
            print(f'Error: {e}')
            import traceback
            traceback.print_exc()
            self.send_error(500, str(e))

    def send_to_telegram(self, filepath, player, clip_type, score):
        try:
            url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVideo'
            caption = f'🎬 LOCOMOT.IO Clip!\n\nPlayer: {player}\nType: {clip_type.replace("_", " ")}\nScore: {score}pts'

            with open(filepath, 'rb') as video:
                r = requests.post(url, files={'video': video},
                                  data={'chat_id': TELEGRAM_CHAT_ID, 'caption': caption})

            if r.ok:
                print(f'📤 Sent to Telegram')
            else:
                print(f'Telegram error: {r.text[:100]}')
        except Exception as e:
            print(f'Telegram failed: {e}')

    def log_message(self, format, *args):
        print(f'[{datetime.now().strftime("%H:%M:%S")}] {args[0]}')

if __name__ == '__main__':
    port = 8765
    print(f'🎬 Clip server on port {port}')
    print(f'📁 Saving clips to: {CLIP_DIR}')
    print(f'📊 Player history: {PLAYER_HISTORY_FILE}')
    HTTPServer(('0.0.0.0', port), ClipHandler).serve_forever()
