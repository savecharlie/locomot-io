#!/usr/bin/env python3
"""
LOCOMOT.IO Arena Training System
Bots fight each other. Winners get fine-tuned. Losers get culled. Dynasties emerge.

Usage:
  python3 arena.py --tournament --matches 100
  python3 arena.py --finetune ivy --episodes 1000
  python3 arena.py --daemon --cycle-hours 6
  python3 arena.py --daemon --cycles 4 --cycle-hours 2
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import json
import requests
import argparse
import time
import os
from collections import deque
from copy import deepcopy
from datetime import datetime

# Rich for pretty output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich import print as rprint
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'rich', '-q'])
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich import print as rprint

console = Console()

# ========== CONFIG ==========
SERVER_URL = 'https://locomot-io.savecharlie.partykit.dev/party/collective'

# Network architecture (must match game)
INPUT_SIZE = 133
HIDDEN1_SIZE = 192
HIDDEN2_SIZE = 96
OUTPUT_SIZE = 3

# Tournament settings
DEFAULT_MATCHES = 100
MATCH_DURATION = 500  # steps per match

# Fine-tuning settings
FINETUNE_EPISODES = 1000
BATCH_SIZE = 128
LEARNING_RATE = 0.0003
GAMMA = 0.99
EPSILON_START = 0.3
EPSILON_MIN = 0.02
EPSILON_DECAY = 0.997

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ========== NEURAL NETWORK ==========
class NeuralNetwork(nn.Module):
    def __init__(self, weights=None, output_size=None):
        super().__init__()
        # Detect output size from weights if provided
        if weights and output_size is None:
            output_size = len(weights['net.4.bias'])
        self.output_size = output_size or OUTPUT_SIZE

        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN1_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN1_SIZE, HIDDEN2_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN2_SIZE, self.output_size)
        )
        if weights:
            self.load_weights(weights)

    def forward(self, x):
        return self.net(x)

    def get_action(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randint(0, 2)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.forward(state_t)
            action = q_values.argmax(dim=1).item()
            # Map 4-output networks: 0=left, 1=straight, 2=right, 3=boost (treat as straight)
            if self.output_size == 4:
                if action == 3:
                    action = 1  # Boost -> straight
            return min(action, 2)  # Clamp to 0-2

    def load_weights(self, weights):
        state_dict = {
            'net.0.weight': torch.FloatTensor(weights['net.0.weight']),
            'net.0.bias': torch.FloatTensor(weights['net.0.bias']),
            'net.2.weight': torch.FloatTensor(weights['net.2.weight']),
            'net.2.bias': torch.FloatTensor(weights['net.2.bias']),
            'net.4.weight': torch.FloatTensor(weights['net.4.weight']),
            'net.4.bias': torch.FloatTensor(weights['net.4.bias']),
        }
        self.load_state_dict(state_dict)

    def to_weights(self):
        state_dict = self.state_dict()
        return {
            'net.0.weight': state_dict['net.0.weight'].cpu().numpy().tolist(),
            'net.0.bias': state_dict['net.0.bias'].cpu().numpy().tolist(),
            'net.2.weight': state_dict['net.2.weight'].cpu().numpy().tolist(),
            'net.2.bias': state_dict['net.2.bias'].cpu().numpy().tolist(),
            'net.4.weight': state_dict['net.4.weight'].cpu().numpy().tolist(),
            'net.4.bias': state_dict['net.4.bias'].cpu().numpy().tolist(),
        }


# ========== ENVIRONMENT (from train_local.py) ==========
POWERUP_SPEED = 0
POWERUP_SHIELD = 1
POWERUP_MAGNET = 2
POWERUP_DURATION = 30

class ArenaEnv:
    WORLD_COLS = 50
    WORLD_ROWS = 40
    DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # UP, RIGHT, DOWN, LEFT
    RAY_OFFSETS = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]

    def __init__(self, num_agents=4):
        self.num_agents = num_agents
        self.projectiles = []
        self.reset()

    def reset(self):
        self.agents = []
        self.gun_pickups = {}
        self.powerup_pickups = {}
        self.segment_map = {}
        self.step_count = 0
        self.projectiles = []
        self.leader_grave = None
        self.kill_log = []  # Track kills: [(killer_idx, victim_idx, step)]

        for i in range(self.num_agents):
            x = random.randint(8, self.WORLD_COLS - 8)
            y = random.randint(8, self.WORLD_ROWS - 8)
            direction = random.randint(0, 3)
            dx, dy = self.DIRECTIONS[direction]

            segments = []
            for j in range(4):
                seg = {'x': x - j*dx, 'y': y - j*dy, 'hp': 100 if j > 0 else float('inf'),
                       'maxHp': 100, 'gun_type': 0 if j > 0 else -1}
                segments.append(seg)

            self.agents.append({
                'segments': segments, 'direction': direction, 'alive': True,
                'score': 0, 'prev_tail_pos': None, 'current_gun': 0,
                'last_turn_time': 0, 'last_turns': [0, 0],
                'invincible_until': 0, 'base_speed': 8,
                'speed_until': 0, 'shield_until': 0, 'magnet_until': 0,
                'survival_time': 0
            })
            for seg_idx, seg in enumerate(segments):
                self.segment_map[(seg['x'], seg['y'])] = (i, seg_idx)

        return [self.get_vision(i) for i in range(self.num_agents)]

    def get_vision(self, agent_idx):
        agent = self.agents[agent_idx]
        if not agent['alive']:
            return np.zeros(INPUT_SIZE, dtype=np.float32)

        head = agent['segments'][0]
        head_x, head_y = head['x'], head['y']
        current_dir = agent['direction']
        my_length = len(agent['segments'])
        my_segments = set((s['x'], s['y']) for s in agent['segments'][1:])

        vision = np.zeros(INPUT_SIZE, dtype=np.float32)
        projectile_positions = set((int(p['x']), int(p['y'])) for p in self.projectiles)

        for ray_idx in range(8):
            rotated_idx = (ray_idx + current_dir * 2) % 8
            dx, dy = self.RAY_OFFSETS[rotated_idx]

            pickup_dist = self_danger = wall_dist = 0.0
            enemy_smaller = enemy_bigger = enemy_body = nearest_enemy_gun = 0.0
            gun_dists = [0.0, 0.0, 0.0, 0.0]
            projectile_danger = ally_head = ally_body = 0.0
            local_crowding = 0

            for dist in range(1, 16):
                cx, cy = head_x + dx * dist, head_y + dy * dist
                if cx < 0 or cx >= self.WORLD_COLS or cy < 0 or cy >= self.WORLD_ROWS:
                    if wall_dist == 0: wall_dist = 1.0 / dist
                    break

                if self_danger == 0 and (cx, cy) in my_segments:
                    self_danger = 1.0 / dist
                    local_crowding += 1

                if projectile_danger == 0 and (cx, cy) in projectile_positions:
                    projectile_danger = 1.0 / dist

                pos = (cx, cy)
                if pos in self.gun_pickups:
                    if pickup_dist == 0:
                        pickup_dist = 1.0 / dist
                    if gun_dists[self.gun_pickups[pos]] == 0:
                        gun_dists[self.gun_pickups[pos]] = 1.0 / dist

                if pos in self.powerup_pickups:
                    if pickup_dist == 0:
                        pickup_dist = 1.0 / dist

                if self.leader_grave and pos == (self.leader_grave['x'], self.leader_grave['y']):
                    if pickup_dist == 0:
                        pickup_dist = 0.8 / dist

                if pos in self.segment_map:
                    other_idx, seg_idx = self.segment_map[pos]
                    if other_idx != agent_idx and self.agents[other_idx]['alive']:
                        other = self.agents[other_idx]
                        local_crowding += 1

                        if seg_idx == 0:
                            if len(other['segments']) < my_length and enemy_smaller == 0:
                                enemy_smaller = 1.0 / dist
                            elif enemy_bigger == 0:
                                enemy_bigger = 1.0 / dist
                        elif enemy_body == 0:
                            enemy_body = 1.0 / dist

            local_crowding = min(local_crowding / 15, 1.0)

            base = ray_idx * 15
            vision[base:base+5] = [pickup_dist] + gun_dists
            vision[base+5:base+11] = [self_danger, wall_dist, enemy_smaller, enemy_bigger, enemy_body, nearest_enemy_gun]
            vision[base+11:base+15] = [projectile_danger, ally_head, ally_body, local_crowding]

        total_hp = sum(s['hp'] for s in agent['segments'][1:] if s['hp'] != float('inf'))
        total_max = sum(s['maxHp'] for s in agent['segments'][1:] if s['maxHp'] != float('inf'))
        health_ratio = total_hp / total_max if total_max > 0 else 1.0

        dist_to_edge = min(head_x, head_y, self.WORLD_COLS - 1 - head_x, self.WORLD_ROWS - 1 - head_y)
        arena_pos = min(dist_to_edge / (min(self.WORLD_COLS, self.WORLD_ROWS) / 2), 1.0)

        threat = 0.0
        for i, other in enumerate(self.agents):
            if i == agent_idx or not other['alive']: continue
            oh = other['segments'][0]
            dist = abs(oh['x'] - head_x) + abs(oh['y'] - head_y)
            if dist < 15:
                size_factor = 2.0 if len(other['segments']) > my_length else 0.5
                threat += size_factor / max(dist, 1)
        threat = min(threat / 3, 1.0)

        time_since_turn = min((self.step_count - agent['last_turn_time']) / 20, 1.0)
        last_turn_1 = (agent['last_turns'][0] + 1) / 2
        last_turn_2 = (agent['last_turns'][1] + 1) / 2
        invincibility = max(0, agent['invincible_until'] - self.step_count) / 30
        relative_speed = 0.5

        vision[120:133] = [
            health_ratio, arena_pos, threat, min(my_length/20, 1), 0.25,
            0, 0, 1, time_since_turn, last_turn_1, last_turn_2, invincibility, relative_speed
        ]
        return vision

    def step(self, actions):
        self.step_count += 1

        # Update survival time for alive agents
        for agent in self.agents:
            if agent['alive']:
                agent['survival_time'] += 1

        # Apply turns
        for i, agent in enumerate(self.agents):
            if not agent['alive']: continue
            if actions[i] == 0:
                agent['direction'] = (agent['direction'] - 1) % 4
                turn_dir = -1
            elif actions[i] == 2:
                agent['direction'] = (agent['direction'] + 1) % 4
                turn_dir = 1
            else:
                turn_dir = 0

            if turn_dir != 0:
                agent['last_turn_time'] = self.step_count
                agent['last_turns'][1] = agent['last_turns'][0]
                agent['last_turns'][0] = turn_dir

        # Projectiles
        if random.random() < 0.1 and len(self.projectiles) < 20:
            alive_agents = [a for a in self.agents if a['alive'] and len(a['segments']) > 1]
            if alive_agents:
                shooter = random.choice(alive_agents)
                head = shooter['segments'][0]
                dx, dy = self.DIRECTIONS[shooter['direction']]
                self.projectiles.append({'x': head['x'] + dx*2, 'y': head['y'] + dy*2, 'vx': dx, 'vy': dy, 'life': 10})

        new_projectiles = []
        for p in self.projectiles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            if p['life'] > 0 and 0 <= p['x'] < self.WORLD_COLS and 0 <= p['y'] < self.WORLD_ROWS:
                new_projectiles.append(p)
        self.projectiles = new_projectiles

        # Move
        for i, agent in enumerate(self.agents):
            if not agent['alive']: continue
            old_tail = agent['segments'][-1]
            agent['prev_tail_pos'] = (old_tail['x'], old_tail['y'])
            if agent['prev_tail_pos'] in self.segment_map and self.segment_map[agent['prev_tail_pos']][0] == i:
                del self.segment_map[agent['prev_tail_pos']]
            dx, dy = self.DIRECTIONS[agent['direction']]
            head = agent['segments'][0]
            new_head = {'x': head['x'] + dx, 'y': head['y'] + dy, 'hp': float('inf'), 'maxHp': float('inf'), 'gun_type': -1}
            agent['segments'].insert(0, new_head)
            agent['segments'].pop()
            for seg_idx, seg in enumerate(agent['segments']):
                self.segment_map[(seg['x'], seg['y'])] = (i, seg_idx)

        # Collisions
        for i, agent in enumerate(self.agents):
            if not agent['alive']: continue
            head = agent['segments'][0]
            hx, hy = head['x'], head['y']
            has_shield = agent['shield_until'] > self.step_count

            # Wall
            if hx < 0 or hx >= self.WORLD_COLS or hy < 0 or hy >= self.WORLD_ROWS:
                agent['alive'] = False
                self.kill_log.append((-1, i, self.step_count))  # -1 = wall/self kill
                continue

            # Self collision
            my_body = set((s['x'], s['y']) for s in agent['segments'][1:])
            if (hx, hy) in my_body and not has_shield:
                agent['alive'] = False
                self.kill_log.append((-1, i, self.step_count))
                continue

            # Other agents
            for j, other in enumerate(self.agents):
                if i == j or not other['alive']: continue
                oh = other['segments'][0]
                other_has_shield = other['shield_until'] > self.step_count

                if hx == oh['x'] and hy == oh['y']:
                    # Head-on collision
                    if len(agent['segments']) > len(other['segments']):
                        if not other_has_shield:
                            other['alive'] = False
                            self.kill_log.append((i, j, self.step_count))
                    elif len(agent['segments']) < len(other['segments']):
                        if not has_shield:
                            agent['alive'] = False
                            self.kill_log.append((j, i, self.step_count))
                    else:
                        # Tie
                        if not has_shield:
                            agent['alive'] = False
                            self.kill_log.append((j, i, self.step_count))
                        if not other_has_shield:
                            other['alive'] = False
                            self.kill_log.append((i, j, self.step_count))
                    break

                other_body = set((s['x'], s['y']) for s in other['segments'][1:])
                if (hx, hy) in other_body and not has_shield:
                    agent['alive'] = False
                    self.kill_log.append((j, i, self.step_count))
                    break

        # Drop guns when agents die
        for i, agent in enumerate(self.agents):
            if not agent['alive'] and len(agent['segments']) > 0:
                dying_length = len(agent['segments'])
                head = agent['segments'][0]
                drop_pos = (head['x'], head['y'])

                max_alive_length = max((len(a['segments']) for a in self.agents if a['alive']), default=0)
                if dying_length > max_alive_length and self.leader_grave is None:
                    self.leader_grave = {
                        'x': head['x'], 'y': head['y'],
                        'spawned': 0, 'last_spawn': self.step_count
                    }

                if 0 <= drop_pos[0] < self.WORLD_COLS and 0 <= drop_pos[1] < self.WORLD_ROWS:
                    if drop_pos not in self.gun_pickups:
                        self.gun_pickups[drop_pos] = random.randint(0, 3)
                agent['segments'] = []

        # Leader Grave spawns powerups
        if self.leader_grave and self.leader_grave['spawned'] < 5:
            if self.step_count - self.leader_grave['last_spawn'] >= 15:
                angle = (self.leader_grave['spawned'] / 5) * 2 * 3.14159
                dist = 2 + self.leader_grave['spawned']
                px = int(self.leader_grave['x'] + dist * np.cos(angle))
                py = int(self.leader_grave['y'] + dist * np.sin(angle))
                px = max(0, min(self.WORLD_COLS - 1, px))
                py = max(0, min(self.WORLD_ROWS - 1, py))
                if (px, py) not in self.powerup_pickups:
                    self.powerup_pickups[(px, py)] = random.randint(0, 2)
                self.leader_grave['spawned'] += 1
                self.leader_grave['last_spawn'] = self.step_count

        # Pickups
        for i, agent in enumerate(self.agents):
            if not agent['alive']: continue
            head = agent['segments'][0]
            pos = (head['x'], head['y'])

            if pos in self.gun_pickups:
                gun_type = self.gun_pickups[pos]
                del self.gun_pickups[pos]
                for seg in agent['segments'][1:]:
                    seg['gun_type'] = gun_type
                agent['current_gun'] = gun_type
                prev_tail = agent['prev_tail_pos']
                agent['segments'].append({'x': prev_tail[0], 'y': prev_tail[1], 'hp': 100, 'maxHp': 100, 'gun_type': gun_type})
                self.segment_map[prev_tail] = (i, len(agent['segments']) - 1)
                agent['score'] += 1

            if pos in self.powerup_pickups:
                powerup_type = self.powerup_pickups[pos]
                del self.powerup_pickups[pos]
                if powerup_type == POWERUP_SPEED:
                    agent['speed_until'] = self.step_count + POWERUP_DURATION
                elif powerup_type == POWERUP_SHIELD:
                    agent['shield_until'] = self.step_count + POWERUP_DURATION
                elif powerup_type == POWERUP_MAGNET:
                    agent['magnet_until'] = self.step_count + POWERUP_DURATION

        observations = [self.get_vision(i) for i in range(self.num_agents)]
        dones = [not a['alive'] for a in self.agents]
        alive = sum(1 for a in self.agents if a['alive'])
        game_over = alive <= 1 or self.step_count >= MATCH_DURATION

        return observations, dones, game_over


# ========== GENOME LOADING & UPLOADING ==========
def fetch_manifest():
    """Fetch genome manifest from server."""
    try:
        response = requests.post(SERVER_URL, json={'type': 'get_genome_manifest'}, timeout=30)
        return response.json()
    except Exception as e:
        console.print(f"[red]Failed to fetch manifest: {e}[/red]")
        return None


def upload_to_pool(weights, genome_id, name, genome_type='rl_trained',
                   parent_id=None, generation=1):
    """Upload genome weights to the server pool."""
    console.print(f"[yellow]Uploading {genome_id} to pool...[/yellow]")

    MAX_CHUNK_SIZE = 50000
    stored_keys = []

    try:
        for key in weights:
            data = weights[key]

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
                            'genome_id': genome_id,
                            'key': chunk_key,
                            'data': chunk_data
                        }, timeout=60)

                        if response.status_code != 200:
                            console.print(f"[red]Chunk {i} failed[/red]")
                            return False
                        stored_keys.append(chunk_key)

                    # Store meta
                    stored_keys.append(f"{key}_meta")
                    requests.post(SERVER_URL, json={
                        'type': 'store_genome_part',
                        'genome_id': genome_id,
                        'key': f"{key}_meta",
                        'data': {'chunks': num_chunks, 'rows_per_chunk': rows_per_chunk, 'total_rows': rows}
                    }, timeout=30)
                    continue

            # Small enough to store directly
            response = requests.post(SERVER_URL, json={
                'type': 'store_genome_part',
                'genome_id': genome_id,
                'key': key,
                'data': data
            }, timeout=60)

            if response.status_code == 200:
                stored_keys.append(key)

        # Register the genome with lineage info
        parents = [parent_id] if parent_id else []
        response = requests.post(SERVER_URL, json={
            'type': 'register_genome',
            'genome_id': genome_id,
            'name': name,
            'genome_type': genome_type,
            'source': f'finetune:{parent_id}' if parent_id else 'training',
            'keys': stored_keys,
            'parents': parents,
            'generation': generation
        }, timeout=30)

        if response.status_code == 200:
            console.print(f"[green]Uploaded: {genome_id}[/green]")
            return True
        else:
            console.print(f"[red]Registration failed: {response.text}[/red]")
            return False

    except Exception as e:
        console.print(f"[red]Upload failed: {e}[/red]")
        return False


def fetch_genome_weights(genome_id):
    """Fetch all weight parts for a genome and reassemble."""
    manifest = fetch_manifest()
    if not manifest:
        return None

    genome_info = next((g for g in manifest['genomes'] if g['id'] == genome_id), None)
    if not genome_info:
        console.print(f"[red]Genome {genome_id} not found in manifest[/red]")
        return None

    weights = {}
    base_keys = set()

    # Find base keys
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


def load_genome(genome_id):
    """Load a genome as a NeuralNetwork."""
    # Try server first
    weights = fetch_genome_weights(genome_id)
    if weights:
        return NeuralNetwork(weights).to(device)

    # Fallback to local file
    local_path = f"brain_{genome_id}.json"
    if os.path.exists(local_path):
        with open(local_path, 'r') as f:
            weights = json.load(f)
        return NeuralNetwork(weights).to(device)

    console.print(f"[red]Could not load genome: {genome_id}[/red]")
    return None


# ========== TOURNAMENT ==========
class Tournament:
    def __init__(self, genome_ids=None):
        self.manifest = fetch_manifest()
        if not self.manifest:
            raise RuntimeError("Could not fetch genome manifest - check network connection")

        if genome_ids:
            self.genome_ids = genome_ids
        else:
            # Get all active genomes
            self.genome_ids = [g['id'] for g in self.manifest['genomes'] if g.get('active', True)]

        console.print(f"[cyan]Loading {len(self.genome_ids)} genomes...[/cyan]")

        self.genomes = {}
        for gid in self.genome_ids:
            brain = load_genome(gid)
            if brain:
                self.genomes[gid] = brain
                console.print(f"  [green]+[/green] {gid}")
            else:
                console.print(f"  [red]x[/red] {gid}")

        self.stats = {gid: {
            'kills': 0, 'deaths': 0, 'score': 0, 'games': 0,
            'survival_time': 0, 'wins': 0
        } for gid in self.genomes.keys()}

    def run_match(self):
        """Run a single match with all genomes."""
        if len(self.genomes) < 2:
            console.print("[red]Need at least 2 genomes for a match[/red]")
            return None

        # Pick participants (up to 4)
        participants = list(self.genomes.keys())
        if len(participants) > 4:
            participants = random.sample(participants, 4)
        while len(participants) < 4:
            participants.append(random.choice(list(self.genomes.keys())))

        random.shuffle(participants)

        env = ArenaEnv(num_agents=len(participants))
        obs = env.reset()

        while True:
            actions = []
            for i, gid in enumerate(participants):
                if env.agents[i]['alive']:
                    actions.append(self.genomes[gid].get_action(obs[i], 0.0))
                else:
                    actions.append(1)

            obs, dones, done = env.step(actions)
            if done:
                break

        # Process results
        results = {'participants': participants, 'kills': {}, 'survivor': None}

        # Count kills from log
        kill_counts = {gid: 0 for gid in participants}
        for killer_idx, victim_idx, _ in env.kill_log:
            if killer_idx >= 0 and killer_idx < len(participants):
                kill_counts[participants[killer_idx]] += 1

        results['kills'] = kill_counts

        # Update stats
        for i, gid in enumerate(participants):
            self.stats[gid]['games'] += 1
            self.stats[gid]['score'] += env.agents[i]['score']
            self.stats[gid]['survival_time'] += env.agents[i]['survival_time']
            self.stats[gid]['kills'] += kill_counts[gid]

            if not env.agents[i]['alive']:
                self.stats[gid]['deaths'] += 1
            else:
                results['survivor'] = gid
                self.stats[gid]['wins'] += 1

        return results

    def run_tournament(self, matches=DEFAULT_MATCHES):
        """Run multiple matches and rank genomes."""
        console.print(Panel.fit(
            f"[bold cyan]ARENA TOURNAMENT[/bold cyan]\n"
            f"[dim]{len(self.genomes)} genomes | {matches} matches[/dim]",
            border_style="cyan"
        ))

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Fighting...", total=matches)

            for _ in range(matches):
                self.run_match()
                progress.update(task, advance=1)

        return self.get_rankings()

    def calculate_fitness(self, gid):
        """Calculate fitness score for a genome."""
        s = self.stats[gid]
        if s['games'] == 0:
            return 0

        kd_ratio = s['kills'] / max(s['deaths'], 1)
        avg_score = s['score'] / s['games']
        avg_survival = s['survival_time'] / s['games']
        win_rate = s['wins'] / s['games']

        # Weighted fitness
        fitness = (
            0.3 * kd_ratio +
            0.2 * (avg_score / 10) +  # Normalize score
            0.2 * (avg_survival / MATCH_DURATION) +  # Normalize survival
            0.3 * win_rate
        )
        return fitness

    def get_rankings(self):
        """Get genomes ranked by fitness."""
        rankings = []
        for gid in self.genomes.keys():
            fitness = self.calculate_fitness(gid)
            s = self.stats[gid]
            rankings.append({
                'id': gid,
                'fitness': fitness,
                'games': s['games'],
                'kills': s['kills'],
                'deaths': s['deaths'],
                'kd': s['kills'] / max(s['deaths'], 1),
                'win_rate': s['wins'] / s['games'] if s['games'] > 0 else 0,
                'avg_score': s['score'] / s['games'] if s['games'] > 0 else 0,
                'avg_survival': s['survival_time'] / s['games'] if s['games'] > 0 else 0
            })

        rankings.sort(key=lambda x: x['fitness'], reverse=True)
        return rankings

    def display_rankings(self, rankings):
        """Display tournament results."""
        table = Table(title="Tournament Results", border_style="green")
        table.add_column("#", style="dim", width=3)
        table.add_column("Genome", style="cyan")
        table.add_column("Fitness", style="green")
        table.add_column("K/D", style="yellow")
        table.add_column("Win%", style="magenta")
        table.add_column("Kills", style="red")
        table.add_column("Games", style="dim")

        for i, r in enumerate(rankings):
            rank = f"#{i+1}"
            fitness = f"{r['fitness']:.3f}"
            kd = f"{r['kd']:.2f}"
            win_rate = f"{r['win_rate']:.0%}"
            kills = str(r['kills'])
            games = str(r['games'])

            style = "bold" if i == 0 else None
            table.add_row(rank, r['id'], fitness, kd, win_rate, kills, games, style=style)

        console.print(table)


# ========== FINE-TUNING ==========
class FineTuner:
    def __init__(self, genome_id, pool_genomes=None):
        self.genome_id = genome_id
        self.model = load_genome(genome_id)
        if not self.model:
            raise ValueError(f"Could not load genome: {genome_id}")

        self.target = NeuralNetwork(output_size=self.model.output_size).to(device)
        self.target.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=50000)
        self.epsilon = EPSILON_START

        # Load opponent pool
        self.opponents = []
        if pool_genomes:
            for gid in pool_genomes:
                if gid != genome_id:
                    opp = load_genome(gid)
                    if opp:
                        self.opponents.append((gid, opp))

        if not self.opponents:
            # Just use self as opponent
            self.opponents.append((genome_id, deepcopy(self.model)))

        console.print(f"[cyan]Loaded {len(self.opponents)} opponents for fine-tuning[/cyan]")

    def get_opponents(self, count=3):
        """Sample opponents for a match."""
        sampled = []
        for _ in range(count):
            # 50% top performers, 30% similar, 20% random
            gid, opp = random.choice(self.opponents)
            sampled.append(deepcopy(opp))
        return sampled

    def play_episode(self):
        """Play one episode against pool opponents."""
        env = ArenaEnv(num_agents=4)
        obs = env.reset()
        opponents = self.get_opponents(3)
        total_reward = 0

        while True:
            actions = [self.model.get_action(obs[0], self.epsilon)]
            for i, opp in enumerate(opponents):
                if env.agents[i + 1]['alive']:
                    actions.append(opp.get_action(obs[i + 1], 0.0))
                else:
                    actions.append(1)

            next_obs, dones, done = env.step(actions)

            # Calculate reward for agent 0
            reward = 0.01  # Survival bonus
            for killer_idx, victim_idx, _ in env.kill_log:
                if killer_idx == 0:
                    reward += 3.0  # Kill reward
            if dones[0]:
                reward -= 5.0  # Death penalty

            if env.agents[0]['alive'] or dones[0]:
                self.memory.append((obs[0], actions[0], reward, next_obs[0], dones[0]))
                total_reward += reward

            obs = next_obs
            if done:
                break

        return total_reward, env.agents[0]['alive']

    def train_step(self):
        """Train on a batch from memory."""
        if len(self.memory) < BATCH_SIZE:
            return 0

        batch = random.sample(self.memory, BATCH_SIZE)
        states = torch.FloatTensor(np.array([b[0] for b in batch])).to(device)
        actions = torch.LongTensor([b[1] for b in batch]).to(device)
        rewards = torch.FloatTensor([b[2] for b in batch]).to(device)
        next_states = torch.FloatTensor(np.array([b[3] for b in batch])).to(device)
        dones = torch.FloatTensor([b[4] for b in batch]).to(device)

        q = self.model(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1)
            next_q = self.target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = rewards + GAMMA * next_q * (1 - dones)

        loss = nn.MSELoss()(q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        """Soft update target network."""
        tau = 0.01
        for tp, cp in zip(self.target.parameters(), self.model.parameters()):
            tp.data.copy_(tau * cp.data + (1 - tau) * tp.data)

    def finetune(self, episodes=FINETUNE_EPISODES):
        """Run fine-tuning."""
        console.print(Panel.fit(
            f"[bold cyan]FINE-TUNING: {self.genome_id}[/bold cyan]\n"
            f"[dim]{episodes} episodes | {len(self.opponents)} opponents[/dim]",
            border_style="cyan"
        ))

        wins = 0
        total_games = 0
        recent_rewards = deque(maxlen=50)

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Training...", total=episodes)

            for ep in range(episodes):
                reward, won = self.play_episode()
                recent_rewards.append(reward)
                total_games += 1
                if won:
                    wins += 1

                for _ in range(2):
                    self.train_step()

                self.update_target()
                self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

                avg_r = np.mean(recent_rewards) if recent_rewards else 0
                win_rate = wins / max(1, total_games)
                progress.update(task, advance=1,
                    description=f"[cyan]R:{avg_r:+.1f} Win:{win_rate:.0%} e:{self.epsilon:.2f}")

        # Save improved genome
        timestamp = datetime.now().strftime('%H%M%S')
        new_id = f"{self.genome_id}_ft_{timestamp}"
        weights = self.model.to_weights()

        # Save locally
        local_file = f"brain_{new_id}.json"
        with open(local_file, 'w') as f:
            json.dump(weights, f)
        console.print(f"[green]Saved: {local_file}[/green]")

        # Get parent generation for lineage
        manifest = fetch_manifest()
        parent_gen = 0
        if manifest:
            parent_info = next((g for g in manifest['genomes'] if g['id'] == self.genome_id), None)
            if parent_info:
                parent_gen = parent_info.get('generation', 0)

        # Upload to server pool
        upload_to_pool(
            weights,
            genome_id=new_id,
            name=f"{self.genome_id} (trained)",
            genome_type='rl_trained',
            parent_id=self.genome_id,
            generation=parent_gen + 1
        )

        return new_id, weights


# ========== CULLING ==========
def cull_weakest(rankings, keep_top=10):
    """Mark bottom performers as inactive. Never cull Gen 0 original player genomes."""
    manifest = fetch_manifest()
    if not manifest:
        return []

    culled = []

    # Sort by fitness (rankings already sorted)
    for i, r in enumerate(rankings):
        if i < keep_top:
            continue  # Keep top performers

        genome_id = r['id']
        genome_info = next((g for g in manifest['genomes'] if g['id'] == genome_id), None)
        if not genome_info:
            continue

        # Never cull Gen 0 (original player genomes)
        if genome_info.get('generation', 0) == 0:
            console.print(f"[yellow]  Skipping {genome_id} (Gen 0 - protected)[/yellow]")
            continue

        # Mark as inactive
        console.print(f"[red]  Culling {genome_id}[/red]")
        try:
            response = requests.post(SERVER_URL, json={
                'type': 'update_genome_status',
                'genome_id': genome_id,
                'active': False
            }, timeout=30)
            if response.status_code == 200:
                culled.append(genome_id)
        except Exception as e:
            console.print(f"[red]Failed to cull {genome_id}: {e}[/red]")

    return culled


def upload_tournament_results(rankings):
    """Upload tournament results to server for tracking."""
    try:
        for r in rankings:
            response = requests.post(SERVER_URL, json={
                'type': 'update_genome_stats',
                'genome_id': r['id'],
                'stats': {
                    'arena_fitness': r['fitness'],
                    'arena_kills': r['kills'],
                    'arena_deaths': r['deaths'],
                    'arena_kd': r['kd'],
                    'arena_win_rate': r['win_rate'],
                    'arena_games': r['games'],
                    'last_tournament': datetime.now().isoformat()
                }
            }, timeout=30)
        console.print("[green]Uploaded tournament results to server[/green]")
    except Exception as e:
        console.print(f"[red]Failed to upload results: {e}[/red]")


# ========== MAIN ==========
def main():
    parser = argparse.ArgumentParser(description='LOCOMOT.IO Arena Training')
    parser.add_argument('--tournament', action='store_true', help='Run tournament')
    parser.add_argument('--matches', type=int, default=DEFAULT_MATCHES, help='Matches per tournament')
    parser.add_argument('--finetune', type=str, help='Fine-tune a specific genome')
    parser.add_argument('--episodes', type=int, default=FINETUNE_EPISODES, help='Fine-tuning episodes')
    parser.add_argument('--daemon', action='store_true', help='Run in daemon mode')
    parser.add_argument('--cycles', type=int, default=0, help='Number of cycles (0 = infinite)')
    parser.add_argument('--cycle-hours', type=float, default=6, help='Hours per cycle')
    args = parser.parse_args()

    if args.tournament:
        tournament = Tournament()
        rankings = tournament.run_tournament(args.matches)
        tournament.display_rankings(rankings)

    elif args.finetune:
        # Get pool genomes for opponents
        manifest = fetch_manifest()
        pool_ids = [g['id'] for g in manifest['genomes'] if g.get('active', True)]

        finetuner = FineTuner(args.finetune, pool_ids)
        new_id, weights = finetuner.finetune(args.episodes)
        console.print(f"[green]Fine-tuned genome: {new_id}[/green]")

    elif args.daemon:
        console.print(Panel.fit(
            "[bold cyan]ARENA DAEMON[/bold cyan]\n"
            "[dim]Training bots while you sleep...[/dim]",
            border_style="cyan"
        ))

        cycle = 0
        while args.cycles == 0 or cycle < args.cycles:
            cycle += 1
            console.print(f"\n[yellow]━━━ Cycle {cycle} ━━━[/yellow]")

            # 1. Run tournament
            tournament = Tournament()
            rankings = tournament.run_tournament(args.matches)
            tournament.display_rankings(rankings)

            # 2. Upload results to server
            upload_tournament_results(rankings)

            # 3. Fine-tune top 3
            if len(rankings) >= 3:
                for r in rankings[:3]:
                    console.print(f"\n[cyan]Fine-tuning {r['id']}...[/cyan]")
                    try:
                        finetuner = FineTuner(r['id'], [x['id'] for x in rankings])
                        finetuner.finetune(500)
                    except Exception as e:
                        console.print(f"[red]Fine-tuning failed: {e}[/red]")

            # 4. Breed top performers
            if len(rankings) >= 2:
                from breed_genome import breed_genomes
                console.print(f"\n[magenta]Breeding {rankings[0]['id']} x {rankings[1]['id']}...[/magenta]")
                try:
                    breed_genomes(rankings[0]['id'], rankings[1]['id'])
                except Exception as e:
                    console.print(f"[red]Breeding failed: {e}[/red]")

            # 5. Cull bottom performers (keep top 10)
            console.print("\n[yellow]Culling weak performers...[/yellow]")
            culled = cull_weakest(rankings, keep_top=10)
            if culled:
                console.print(f"[red]Culled: {', '.join(culled)}[/red]")

            # Sleep until next cycle
            if args.cycles == 0 or cycle < args.cycles:
                sleep_secs = args.cycle_hours * 3600
                console.print(f"\n[dim]Sleeping {args.cycle_hours} hours until next cycle...[/dim]")
                time.sleep(sleep_secs)

        console.print("[green]Daemon complete![/green]")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
