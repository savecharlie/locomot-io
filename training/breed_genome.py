#!/usr/bin/env python3
"""
Genome Breeding System for LOCOMOT.IO
Creates child genomes from parent genomes via crossover and mutation.

Usage:
  python3 breed_genome.py <parent1_id> <parent2_id> [child_name]
  python3 breed_genome.py ivy ivy "IvyBot Jr"  # Self-breed with mutation
  python3 breed_genome.py ivy charlie "Ivy×Charlie"  # Crossbreed
"""

import json
import requests
import sys
import random
import numpy as np
from datetime import datetime

SERVER_URL = 'https://locomot-io.savecharlie.partykit.dev/party/collective'

# Breeding parameters
MUTATION_RATE = 0.1      # Probability of mutating each weight
MUTATION_STRENGTH = 0.1  # Standard deviation of mutation noise
CROSSOVER_RATE = 0.5     # Probability of taking from parent B vs A


def fetch_genome_weights(genome_id):
    """Fetch all weight parts for a genome and reassemble."""
    # Get manifest to find keys
    response = requests.post(SERVER_URL, json={'type': 'get_genome_manifest'}, timeout=30)
    manifest = response.json()

    genome_info = next((g for g in manifest['genomes'] if g['id'] == genome_id), None)
    if not genome_info:
        raise ValueError(f"Genome {genome_id} not found in manifest")

    weights = {}
    base_keys = set()

    # Find base keys (without _chunk/_meta suffixes)
    for key in genome_info.get('keys', []):
        if '_chunk' in key or '_meta' in key:
            base_keys.add(key.split('_chunk')[0].split('_meta')[0])
        else:
            base_keys.add(key)

    # Fetch each key
    for base_key in base_keys:
        meta_key = f"{base_key}_meta"
        if meta_key in genome_info.get('keys', []):
            # Chunked - fetch meta first
            meta_resp = requests.post(SERVER_URL, json={
                'type': 'get_genome_part',
                'genome_id': genome_id,
                'key': meta_key
            }, timeout=30)
            meta = meta_resp.json()['data']

            # Fetch all chunks
            rows = []
            for i in range(meta['chunks']):
                chunk_resp = requests.post(SERVER_URL, json={
                    'type': 'get_genome_part',
                    'genome_id': genome_id,
                    'key': f"{base_key}_chunk{i}"
                }, timeout=30)
                rows.extend(chunk_resp.json()['data'])
            weights[base_key] = rows
        else:
            # Direct fetch
            part_resp = requests.post(SERVER_URL, json={
                'type': 'get_genome_part',
                'genome_id': genome_id,
                'key': base_key
            }, timeout=30)
            weights[base_key] = part_resp.json()['data']

    return weights


def crossover(parent_a, parent_b, rate=CROSSOVER_RATE):
    """Crossover two weight arrays. Works for both 1D and 2D arrays."""
    if isinstance(parent_a[0], list):
        # 2D array - crossover by row
        child = []
        for i in range(len(parent_a)):
            if random.random() < rate:
                child.append(parent_b[i].copy() if i < len(parent_b) else parent_a[i].copy())
            else:
                child.append(parent_a[i].copy())
        return child
    else:
        # 1D array - crossover by element
        child = []
        for i in range(len(parent_a)):
            if random.random() < rate:
                child.append(parent_b[i] if i < len(parent_b) else parent_a[i])
            else:
                child.append(parent_a[i])
        return child


def mutate(weights, rate=MUTATION_RATE, strength=MUTATION_STRENGTH):
    """Add random mutations to weights."""
    if isinstance(weights[0], list):
        # 2D array
        mutated = []
        for row in weights:
            new_row = []
            for val in row:
                if random.random() < rate:
                    new_row.append(val + random.gauss(0, strength))
                else:
                    new_row.append(val)
            mutated.append(new_row)
        return mutated
    else:
        # 1D array
        mutated = []
        for val in weights:
            if random.random() < rate:
                mutated.append(val + random.gauss(0, strength))
            else:
                mutated.append(val)
        return mutated


def breed(parent_a_weights, parent_b_weights):
    """Breed two genomes to create a child."""
    child_weights = {}

    for key in parent_a_weights:
        if key not in parent_b_weights:
            # Parent B doesn't have this key, just mutate A
            child_weights[key] = mutate(parent_a_weights[key])
        else:
            # Crossover then mutate
            crossed = crossover(parent_a_weights[key], parent_b_weights[key])
            child_weights[key] = mutate(crossed)

    return child_weights


def upload_child(child_id, child_name, child_weights, parent_a_id, parent_b_id, generation):
    """Upload child genome to server using chunked upload."""
    from upload_genome import upload_genome_weights

    # Upload weights
    success = upload_genome_weights(child_weights, child_id, child_name,
                                     genome_type='bred',
                                     source=f'breed:{parent_a_id}×{parent_b_id}',
                                     extra_metadata={
                                         'parents': [parent_a_id, parent_b_id],
                                         'generation': generation
                                     })
    return success


def get_generation(genome_id, manifest):
    """Get the generation of a genome (0 for original players, +1 for each breeding)."""
    genome_info = next((g for g in manifest['genomes'] if g['id'] == genome_id), None)
    if not genome_info:
        return 0
    return genome_info.get('generation', 0)


def breed_genomes(parent_a_id, parent_b_id, child_name=None):
    """Main breeding function."""
    print(f"Breeding {parent_a_id} × {parent_b_id}")

    # Fetch parent weights
    print(f"  Fetching {parent_a_id}...")
    parent_a_weights = fetch_genome_weights(parent_a_id)

    if parent_a_id == parent_b_id:
        print(f"  Self-breeding (mutation only)")
        parent_b_weights = parent_a_weights
    else:
        print(f"  Fetching {parent_b_id}...")
        parent_b_weights = fetch_genome_weights(parent_b_id)

    # Breed
    print("  Crossover + mutation...")
    child_weights = breed(parent_a_weights, parent_b_weights)

    # Get generation
    manifest_resp = requests.post(SERVER_URL, json={'type': 'get_genome_manifest'}, timeout=30)
    manifest = manifest_resp.json()
    gen_a = get_generation(parent_a_id, manifest)
    gen_b = get_generation(parent_b_id, manifest)
    child_generation = max(gen_a, gen_b) + 1

    # Generate child ID and name
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    if parent_a_id == parent_b_id:
        child_id = f"{parent_a_id}_jr_{timestamp[-6:]}"
        if not child_name:
            child_name = f"{parent_a_id.title()}Bot Jr"
    else:
        child_id = f"{parent_a_id}x{parent_b_id}_{timestamp[-6:]}"
        if not child_name:
            # Get parent names from manifest
            name_a = next((g['name'] for g in manifest['genomes'] if g['id'] == parent_a_id), parent_a_id)
            name_b = next((g['name'] for g in manifest['genomes'] if g['id'] == parent_b_id), parent_b_id)
            # Shorten names if needed
            short_a = name_a.replace('Bot', '').strip()[:8]
            short_b = name_b.replace('Bot', '').strip()[:8]
            child_name = f"{short_a}×{short_b}"

    print(f"  Child: {child_id} ({child_name}) - Generation {child_generation}")

    # Save locally first
    local_file = f"brain_{child_id}.json"
    with open(local_file, 'w') as f:
        json.dump(child_weights, f)
    print(f"  Saved locally: {local_file}")

    # Upload to server
    print("  Uploading to server...")

    # Use the upload script logic
    MAX_CHUNK_SIZE = 50000
    stored_keys = []

    for key in child_weights:
        data = child_weights[key]

        if isinstance(data[0], list):
            # 2D array - may need chunking
            rows = len(data)
            row_size = len(json.dumps(data[0]))
            rows_per_chunk = max(1, MAX_CHUNK_SIZE // row_size)
            num_chunks = (rows + rows_per_chunk - 1) // rows_per_chunk

            if num_chunks > 1:
                for i in range(num_chunks):
                    start = i * rows_per_chunk
                    end = min((i + 1) * rows_per_chunk, rows)
                    chunk_data = data[start:end]
                    chunk_key = f"{key}_chunk{i}"

                    response = requests.post(SERVER_URL, json={
                        'type': 'store_genome_part',
                        'genome_id': child_id,
                        'key': chunk_key,
                        'data': chunk_data
                    }, timeout=60)

                    if response.status_code != 200:
                        print(f"    Chunk {i} FAILED")
                        return None
                    stored_keys.append(chunk_key)

                # Store meta
                stored_keys.append(f"{key}_meta")
                requests.post(SERVER_URL, json={
                    'type': 'store_genome_part',
                    'genome_id': child_id,
                    'key': f"{key}_meta",
                    'data': {'chunks': num_chunks, 'rows_per_chunk': rows_per_chunk, 'total_rows': rows}
                }, timeout=30)
                continue

        # Small enough to store directly
        response = requests.post(SERVER_URL, json={
            'type': 'store_genome_part',
            'genome_id': child_id,
            'key': key,
            'data': data
        }, timeout=60)

        if response.status_code == 200:
            stored_keys.append(key)

    # Register the child genome with lineage info
    response = requests.post(SERVER_URL, json={
        'type': 'register_genome',
        'genome_id': child_id,
        'name': child_name,
        'genome_type': 'bred',
        'source': f'breed:{parent_a_id}×{parent_b_id}',
        'keys': stored_keys,
        'parents': [parent_a_id, parent_b_id],
        'generation': child_generation
    }, timeout=30)

    if response.status_code == 200:
        print(f"  SUCCESS: {child_name} born!")
        return child_id
    else:
        print(f"  FAILED: {response.text}")
        return None


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 breed_genome.py <parent1_id> <parent2_id> [child_name]")
        print("Example: python3 breed_genome.py ivy ivy 'IvyBot Jr'")
        print("Example: python3 breed_genome.py ivy charlie 'Ivy×Charlie'")
        sys.exit(1)

    parent_a = sys.argv[1]
    parent_b = sys.argv[2]
    child_name = sys.argv[3] if len(sys.argv) > 3 else None

    result = breed_genomes(parent_a, parent_b, child_name)
    sys.exit(0 if result else 1)
