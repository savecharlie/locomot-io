#!/usr/bin/env python3
"""
LOCOMOT.IO Local Training Script
Run: python3 train_local.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import json
from collections import deque
from copy import deepcopy
import time
import os
from datetime import datetime

# Rich progress bar
try:
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import print as rprint
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("Installing rich for pretty output...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'rich', '-q'])
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import print as rprint
    HAS_RICH = True

console = Console()

# ========== CONFIG ==========
EPISODES = 2000
NEW_INPUT_SIZE = 96
HIDDEN1_SIZE = 128
HIDDEN2_SIZE = 64
OUTPUT_SIZE = 3
BATCH_SIZE = 64
LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON_START = 0.5
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== NEURAL NETWORK ==========
class LocomotNetwork(nn.Module):
    def __init__(self, input_size=NEW_INPUT_SIZE):
        super().__init__()
        self.input_size = input_size
        self.net = nn.Sequential(
            nn.Linear(input_size, HIDDEN1_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN1_SIZE, HIDDEN2_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN2_SIZE, OUTPUT_SIZE)
        )

    def forward(self, x):
        return self.net(x)

    def get_action(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randint(0, 2)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.forward(state_t)
            return q_values.argmax(dim=1).item()

# ========== ENVIRONMENT ==========
class LocomotEnv:
    WORLD_COLS = 50
    WORLD_ROWS = 40
    DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # UP, RIGHT, DOWN, LEFT
    RAY_OFFSETS = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]

    def __init__(self, num_agents=4):
        self.num_agents = num_agents
        self.reset()

    def reset(self):
        self.agents = []
        self.health_pickups = set()
        self.gun_pickups = {}
        self.segment_map = {}
        self.step_count = 0

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
                'score': 0, 'prev_tail_pos': None, 'current_gun': 0
            })
            for seg_idx, seg in enumerate(segments):
                self.segment_map[(seg['x'], seg['y'])] = (i, seg_idx)

        for _ in range(15):
            self.gun_pickups[(random.randint(0, self.WORLD_COLS-1), random.randint(0, self.WORLD_ROWS-1))] = random.randint(0, 3)
        for _ in range(5):
            self.health_pickups.add((random.randint(0, self.WORLD_COLS-1), random.randint(0, self.WORLD_ROWS-1)))

        return [self.get_vision(i) for i in range(self.num_agents)]

    def get_vision(self, agent_idx):
        agent = self.agents[agent_idx]
        if not agent['alive']:
            return np.zeros(NEW_INPUT_SIZE, dtype=np.float32)

        head = agent['segments'][0]
        head_x, head_y = head['x'], head['y']
        current_dir = agent['direction']
        my_length = len(agent['segments'])
        my_segments = set((s['x'], s['y']) for s in agent['segments'][1:])

        vision = np.zeros(NEW_INPUT_SIZE, dtype=np.float32)

        for ray_idx in range(8):
            rotated_idx = (ray_idx + current_dir * 2) % 8
            dx, dy = self.RAY_OFFSETS[rotated_idx]

            health_dist = self_danger = wall_dist = 0.0
            enemy_smaller = enemy_bigger = enemy_body = nearest_enemy_gun = 0.0
            gun_dists = [0.0, 0.0, 0.0, 0.0]

            for dist in range(1, 16):
                cx, cy = head_x + dx * dist, head_y + dy * dist
                if cx < 0 or cx >= self.WORLD_COLS or cy < 0 or cy >= self.WORLD_ROWS:
                    if wall_dist == 0: wall_dist = 1.0 / dist
                    break
                if self_danger == 0 and (cx, cy) in my_segments: self_danger = 1.0 / dist
                if health_dist == 0 and (cx, cy) in self.health_pickups: health_dist = 1.0 / dist
                pos = (cx, cy)
                if pos in self.gun_pickups and gun_dists[self.gun_pickups[pos]] == 0:
                    gun_dists[self.gun_pickups[pos]] = 1.0 / dist
                if pos in self.segment_map:
                    other_idx, seg_idx = self.segment_map[pos]
                    if other_idx != agent_idx and self.agents[other_idx]['alive']:
                        other = self.agents[other_idx]
                        if seg_idx == 0:
                            if len(other['segments']) < my_length and enemy_smaller == 0: enemy_smaller = 1.0 / dist
                            elif enemy_bigger == 0: enemy_bigger = 1.0 / dist
                        elif enemy_body == 0: enemy_body = 1.0 / dist

            base = ray_idx * 11
            vision[base:base+5] = [health_dist] + gun_dists
            vision[base+5:base+11] = [self_danger, wall_dist, enemy_smaller, enemy_bigger, enemy_body, nearest_enemy_gun]

        # State inputs
        total_hp = sum(s['hp'] for s in agent['segments'][1:] if s['hp'] != float('inf'))
        total_max = sum(s['maxHp'] for s in agent['segments'][1:] if s['maxHp'] != float('inf'))
        health_ratio = total_hp / total_max if total_max > 0 else 1.0

        dist_to_edge = min(head_x, head_y, self.WORLD_COLS - 1 - head_x, self.WORLD_ROWS - 1 - head_y)
        arena_pos = min(dist_to_edge / (min(self.WORLD_COLS, self.WORLD_ROWS) / 2), 1.0)

        vision[88:96] = [health_ratio, arena_pos, 0.3, min(my_length/20, 1), 0.25, 0, 0, 1]
        return vision

    def step(self, actions):
        self.step_count += 1
        rewards = np.zeros(self.num_agents, dtype=np.float32)

        # Apply turns
        for i, agent in enumerate(self.agents):
            if not agent['alive']: continue
            if actions[i] == 0: agent['direction'] = (agent['direction'] - 1) % 4
            elif actions[i] == 2: agent['direction'] = (agent['direction'] + 1) % 4

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
            rewards[i] += 0.01

        # Collisions
        for i, agent in enumerate(self.agents):
            if not agent['alive']: continue
            head = agent['segments'][0]
            hx, hy = head['x'], head['y']

            # Wall
            if hx < 0 or hx >= self.WORLD_COLS or hy < 0 or hy >= self.WORLD_ROWS:
                agent['alive'] = False
                rewards[i] -= 5.0
                continue

            # Self
            my_body = set((s['x'], s['y']) for s in agent['segments'][1:])
            if (hx, hy) in my_body:
                agent['alive'] = False
                rewards[i] -= 5.0
                continue

            # Other agents
            for j, other in enumerate(self.agents):
                if i == j or not other['alive']: continue
                oh = other['segments'][0]
                if hx == oh['x'] and hy == oh['y']:
                    if len(agent['segments']) > len(other['segments']):
                        other['alive'] = False
                        rewards[j] -= 5.0
                        rewards[i] += 3.0
                    elif len(agent['segments']) < len(other['segments']):
                        agent['alive'] = False
                        rewards[i] -= 5.0
                        rewards[j] += 3.0
                    else:
                        agent['alive'] = other['alive'] = False
                        rewards[i] = rewards[j] = -3.0
                    break

                other_body = set((s['x'], s['y']) for s in other['segments'][1:])
                if (hx, hy) in other_body:
                    agent['alive'] = False
                    rewards[i] -= 5.0
                    rewards[j] += 2.0
                    break

        # Pickups
        for i, agent in enumerate(self.agents):
            if not agent['alive']: continue
            head = agent['segments'][0]
            pos = (head['x'], head['y'])

            if pos in self.health_pickups:
                self.health_pickups.remove(pos)
                for seg in agent['segments'][1:]:
                    if seg['hp'] != float('inf'):
                        seg['hp'] = min(seg['hp'] + 30, seg['maxHp'])
                rewards[i] += 0.5
                self.health_pickups.add((random.randint(0, self.WORLD_COLS-1), random.randint(0, self.WORLD_ROWS-1)))

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
                rewards[i] += 1.0
                self.gun_pickups[(random.randint(0, self.WORLD_COLS-1), random.randint(0, self.WORLD_ROWS-1))] = random.randint(0, 3)

        observations = [self.get_vision(i) for i in range(self.num_agents)]
        dones = [not a['alive'] for a in self.agents]
        alive = sum(1 for a in self.agents if a['alive'])
        game_over = alive <= 1 or self.step_count >= 500

        if game_over and alive == 1:
            for i, a in enumerate(self.agents):
                if a['alive']: rewards[i] += 5.0

        return observations, rewards.tolist(), dones, game_over

# ========== TRAINER ==========
class Trainer:
    def __init__(self, brain_path=None):
        self.model = LocomotNetwork().to(device)
        self.target = LocomotNetwork().to(device)

        if brain_path and os.path.exists(brain_path):
            self.load_brain(brain_path)
            console.print(f"[green]Loaded brain from {brain_path}[/green]")

        self.target.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=50000)
        self.epsilon = EPSILON_START

        self.opponent_pool = [deepcopy(self.model.state_dict())]
        self.rewards_history = []
        self.wins = 0
        self.total_games = 0

    def load_brain(self, path):
        with open(path, 'r') as f:
            weights = json.load(f)
        state_dict = {
            'net.0.weight': torch.FloatTensor(weights['net.0.weight']),
            'net.0.bias': torch.FloatTensor(weights['net.0.bias']),
            'net.2.weight': torch.FloatTensor(weights['net.2.weight']),
            'net.2.bias': torch.FloatTensor(weights['net.2.bias']),
            'net.4.weight': torch.FloatTensor(weights['net.4.weight']),
            'net.4.bias': torch.FloatTensor(weights['net.4.bias']),
        }
        self.model.load_state_dict(state_dict)

    def save_brain(self, path):
        state_dict = self.model.state_dict()
        brain = {
            'input_size': self.model.input_size,
            'net.0.weight': state_dict['net.0.weight'].cpu().numpy().tolist(),
            'net.0.bias': state_dict['net.0.bias'].cpu().numpy().tolist(),
            'net.2.weight': state_dict['net.2.weight'].cpu().numpy().tolist(),
            'net.2.bias': state_dict['net.2.bias'].cpu().numpy().tolist(),
            'net.4.weight': state_dict['net.4.weight'].cpu().numpy().tolist(),
            'net.4.bias': state_dict['net.4.bias'].cpu().numpy().tolist(),
        }
        with open(path, 'w') as f:
            json.dump(brain, f)

    def get_opponents(self):
        opponents = []
        for _ in range(3):
            if random.random() < 0.3:
                opp = deepcopy(self.model)
            else:
                idx = random.randint(0, len(self.opponent_pool) - 1)
                opp = LocomotNetwork().to(device)
                opp.load_state_dict(self.opponent_pool[idx])
            opp.eval()
            opponents.append(opp)
        return opponents

    def play_episode(self):
        env = LocomotEnv(num_agents=4)
        obs = env.reset()
        opponents = self.get_opponents()
        total_reward = 0

        while True:
            actions = [self.model.get_action(obs[0], self.epsilon)]
            for i, opp in enumerate(opponents):
                if env.agents[i + 1]['alive']:
                    actions.append(opp.get_action(obs[i + 1], 0.0))
                else:
                    actions.append(1)

            next_obs, rewards, dones, done = env.step(actions)

            if env.agents[0]['alive'] or dones[0]:
                self.memory.append((obs[0], actions[0], rewards[0], next_obs[0], dones[0]))
                total_reward += rewards[0]

            obs = next_obs
            if done: break

        self.total_games += 1
        if env.agents[0]['alive']:
            self.wins += 1

        return total_reward, env.agents[0]['alive'], len(env.agents[0]['segments'])

    def train_step(self):
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
        tau = 0.01
        for tp, cp in zip(self.target.parameters(), self.model.parameters()):
            tp.data.copy_(tau * cp.data + (1 - tau) * tp.data)

    def train(self, episodes=EPISODES):
        console.print(Panel.fit(
            "[bold cyan]LOCOMOT.IO Neural Network Training[/bold cyan]\n"
            f"[dim]Device: {device} | Episodes: {episodes}[/dim]",
            border_style="cyan"
        ))

        start_time = time.time()
        recent_rewards = deque(maxlen=50)

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:

            task = progress.add_task("[cyan]Training...", total=episodes)

            for ep in range(episodes):
                reward, won, length = self.play_episode()
                recent_rewards.append(reward)

                for _ in range(2):
                    self.train_step()

                self.update_target()
                self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

                if ep % 50 == 0 and ep > 0:
                    self.opponent_pool.append(deepcopy(self.model.state_dict()))
                    if len(self.opponent_pool) > 10:
                        del self.opponent_pool[len(self.opponent_pool) // 2]

                avg_r = np.mean(recent_rewards) if recent_rewards else 0
                win_rate = self.wins / max(1, self.total_games)

                progress.update(task, advance=1,
                    description=f"[cyan]R:{avg_r:+.1f} Win:{win_rate:.0%} ε:{self.epsilon:.2f}")

                if ep % 500 == 0 and ep > 0:
                    self.save_brain(f'brain_checkpoint_{ep}.json')
                    self.wins = 0
                    self.total_games = 0

        elapsed = time.time() - start_time

        # Save final brain
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'brain_{timestamp}.json'
        self.save_brain(filename)

        # Results table
        table = Table(title="Training Complete!", border_style="green")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Time", f"{elapsed:.1f}s")
        table.add_row("Speed", f"{episodes/elapsed:.1f} ep/s")
        table.add_row("Final ε", f"{self.epsilon:.3f}")
        table.add_row("Saved to", filename)
        console.print(table)

        return filename

# ========== MAIN ==========
if __name__ == '__main__':
    # Find existing brain
    brain_path = None
    for f in sorted(os.listdir('.'), key=lambda x: os.path.getmtime(x) if os.path.isfile(x) else 0, reverse=True):
        if 'brain' in f.lower() and f.endswith('.json'):
            brain_path = f
            break

    trainer = Trainer(brain_path=brain_path)
    output_file = trainer.train(EPISODES)

    console.print(f"\n[bold green]Done![/bold green] New brain: [cyan]{output_file}[/cyan]")
    console.print("[dim]Copy to index.html to deploy[/dim]")
