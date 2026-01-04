#!/usr/bin/env python3
"""
Bot Weight Storage Server for LOCOMOT.IO
Serves bot weights from local filesystem via Cloudflare Tunnel
"""

from flask import Flask, request, send_from_directory, jsonify
from flask_cors import CORS
import os
import json
import logging

app = Flask(__name__)
CORS(app)  # Allow cross-origin from locomot.io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

BOTS_DIR = '/home/ivy/locomot-io/bots'
MAX_BOT_SIZE = 2 * 1024 * 1024  # 2MB max per bot

os.makedirs(BOTS_DIR, exist_ok=True)


@app.route('/bots/<filename>', methods=['GET'])
def get_bot(filename):
    """Serve a bot's weights"""
    logger.info(f'GET /bots/{filename}')
    return send_from_directory(BOTS_DIR, filename)


@app.route('/bots', methods=['POST'])
def upload_bot():
    """Upload a new bot's weights"""
    try:
        data = request.json
        bot_id = data.get('id')
        weights = data.get('weights')

        if not bot_id or not weights:
            return jsonify({'error': 'Missing id or weights'}), 400

        # Sanitize bot_id (only allow alphanumeric, underscore, dash)
        safe_id = ''.join(c for c in bot_id if c.isalnum() or c in '_-')
        if safe_id != bot_id:
            return jsonify({'error': 'Invalid bot ID'}), 400

        # Check size
        weights_json = json.dumps(weights)
        if len(weights_json) > MAX_BOT_SIZE:
            return jsonify({'error': 'Bot too large'}), 400

        # Save to disk
        filepath = os.path.join(BOTS_DIR, f'{safe_id}.json')
        with open(filepath, 'w') as f:
            f.write(weights_json)

        logger.info(f'Saved bot: {safe_id} ({len(weights_json)} bytes)')
        return jsonify({'success': True, 'id': safe_id})

    except Exception as e:
        logger.error(f'Upload error: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/bots', methods=['GET'])
def list_bots():
    """List all stored bots"""
    try:
        files = os.listdir(BOTS_DIR)
        bots = [f.replace('.json', '') for f in files if f.endswith('.json')]
        return jsonify({'bots': bots, 'count': len(bots)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    bot_count = len([f for f in os.listdir(BOTS_DIR) if f.endswith('.json')])
    return jsonify({'status': 'ok', 'bots': bot_count})


if __name__ == '__main__':
    logger.info(f'Starting bot server, serving from {BOTS_DIR}')
    app.run(host='0.0.0.0', port=8102)
