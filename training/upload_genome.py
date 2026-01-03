#!/usr/bin/env python3
"""
Upload a genome to the server using chunked uploads.
Usage: python3 upload_genome.py <brain_file.json> <genome_id> [name] [type]
"""

import json
import requests
import sys
from pathlib import Path

SERVER_URL = 'https://locomot-io.savecharlie.partykit.dev/party/collective'
CHUNK_SIZE = 10  # Rows per chunk for weight matrices


def upload_genome(brain_file, genome_id, name=None, genome_type='behavioral', source=None):
    """Upload a genome to the server in chunks."""

    # Load brain weights
    with open(brain_file, 'r') as f:
        weights = json.load(f)

    if name is None:
        name = genome_id
    if source is None:
        source = f'file:{Path(brain_file).name}'

    print(f"Uploading genome: {genome_id} ({name})")
    print(f"  Type: {genome_type}")
    print(f"  Source: {source}")

    # Get the weight keys we need to upload
    weight_keys = [k for k in weights.keys() if k.startswith('net.')]
    keys_info = []

    for key in weight_keys:
        data = weights[key]

        if isinstance(data, list) and len(data) > 0:
            # Check if it's a 2D array (weight matrix) or 1D (bias)
            if isinstance(data[0], list):
                # 2D weight matrix - chunk by rows
                total_chunks = (len(data) + CHUNK_SIZE - 1) // CHUNK_SIZE
                print(f"  {key}: {len(data)} rows -> {total_chunks} chunks")

                for i in range(total_chunks):
                    start = i * CHUNK_SIZE
                    end = min((i + 1) * CHUNK_SIZE, len(data))
                    chunk_data = data[start:end]

                    response = requests.post(SERVER_URL, json={
                        'type': 'submit_genome_chunk',
                        'genome_id': genome_id,
                        'key': key,
                        'chunk_index': i,
                        'total_chunks': total_chunks,
                        'data': chunk_data
                    }, timeout=30)

                    if response.status_code != 200:
                        print(f"  ERROR: Chunk {i} failed: {response.text}")
                        return False

                keys_info.append({'key': key, 'total_chunks': total_chunks})
            else:
                # 1D bias vector - single chunk
                print(f"  {key}: {len(data)} values -> 1 chunk")

                response = requests.post(SERVER_URL, json={
                    'type': 'submit_genome_chunk',
                    'genome_id': genome_id,
                    'key': key,
                    'chunk_index': 0,
                    'total_chunks': 1,
                    'data': data
                }, timeout=30)

                if response.status_code != 200:
                    print(f"  ERROR: {key} failed: {response.text}")
                    return False

                keys_info.append({'key': key, 'total_chunks': 1})

    # Finalize - assemble chunks on server
    print("Finalizing...")
    response = requests.post(SERVER_URL, json={
        'type': 'finalize_genome',
        'genome_id': genome_id,
        'name': name,
        'genome_type': genome_type,
        'source': source,
        'keys': keys_info
    }, timeout=60)

    if response.status_code == 200:
        result = response.json()
        print(f"SUCCESS: Genome {genome_id} uploaded!")
        print(f"  Active: {result.get('active', True)}")
        return True
    else:
        print(f"ERROR: Finalize failed: {response.text}")
        return False


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 upload_genome.py <brain_file.json> <genome_id> [name] [type]")
        print("Example: python3 upload_genome.py brain_ivy.json ivy IvyBot behavioral")
        sys.exit(1)

    brain_file = sys.argv[1]
    genome_id = sys.argv[2]
    name = sys.argv[3] if len(sys.argv) > 3 else None
    genome_type = sys.argv[4] if len(sys.argv) > 4 else 'behavioral'

    success = upload_genome(brain_file, genome_id, name, genome_type)
    sys.exit(0 if success else 1)
