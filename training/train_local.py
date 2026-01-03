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
EPISODES = 10000  # Lots of training for new larger network
NEW_INPUT_SIZE = 133  # 8 rays × 15 features + 13 state
HIDDEN1_SIZE = 192  # Larger hidden layers for more inputs
HIDDEN2_SIZE = 96
OUTPUT_SIZE = 3
BATCH_SIZE = 128  # Larger batch for stability
LEARNING_RATE = 0.0005  # Lower LR for longer training
GAMMA = 0.99
EPSILON_START = 0.6  # More exploration at start
EPSILON_MIN = 0.02  # Lower final epsilon
EPSILON_DECAY = 0.9995  # Very slow decay for 10k episodes

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

    def __init__(self, num_agents=4, team_mode=None):
        self.num_agents = num_agents
        self.projectiles = []  # Simulated bullets
        # If team_mode is None, randomly choose per episode (50/50)
        self.team_mode = team_mode if team_mode is not None else random.choice([True, False])
        self.reset()

    def reset(self):
        self.agents = []
        self.health_pickups = set()
        self.gun_pickups = {}
        self.segment_map = {}
        self.step_count = 0
        self.projectiles = []

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
                'team': i % 2,  # Alternate teams for team mode simulation
                'last_turn_time': 0, 'last_turns': [0, 0],  # Momentum tracking
                'invincible_until': 0, 'base_speed': 8
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
        my_team = agent['team']

        vision = np.zeros(NEW_INPUT_SIZE, dtype=np.float32)

        # Create projectile position set for fast lookup
        projectile_positions = set((int(p['x']), int(p['y'])) for p in self.projectiles)

        for ray_idx in range(8):
            rotated_idx = (ray_idx + current_dir * 2) % 8
            dx, dy = self.RAY_OFFSETS[rotated_idx]

            health_dist = self_danger = wall_dist = 0.0
            enemy_smaller = enemy_bigger = enemy_body = nearest_enemy_gun = 0.0
            gun_dists = [0.0, 0.0, 0.0, 0.0]
            # New features
            projectile_danger = ally_head = ally_body = 0.0
            local_crowding = 0

            for dist in range(1, 16):
                cx, cy = head_x + dx * dist, head_y + dy * dist
                if cx < 0 or cx >= self.WORLD_COLS or cy < 0 or cy >= self.WORLD_ROWS:
                    if wall_dist == 0: wall_dist = 1.0 / dist
                    break

                # Self collision
                if self_danger == 0 and (cx, cy) in my_segments:
                    self_danger = 1.0 / dist
                    local_crowding += 1

                # Projectile danger
                if projectile_danger == 0 and (cx, cy) in projectile_positions:
                    projectile_danger = 1.0 / dist

                # Health pickups
                if health_dist == 0 and (cx, cy) in self.health_pickups:
                    health_dist = 1.0 / dist

                # Gun pickups
                pos = (cx, cy)
                if pos in self.gun_pickups and gun_dists[self.gun_pickups[pos]] == 0:
                    gun_dists[self.gun_pickups[pos]] = 1.0 / dist

                # Other agents
                if pos in self.segment_map:
                    other_idx, seg_idx = self.segment_map[pos]
                    if other_idx != agent_idx and self.agents[other_idx]['alive']:
                        other = self.agents[other_idx]
                        is_ally = other['team'] == my_team
                        local_crowding += 1

                        if is_ally:
                            # Ally detection
                            if seg_idx == 0 and ally_head == 0:
                                ally_head = 1.0 / dist
                            elif seg_idx > 0 and ally_body == 0:
                                ally_body = 1.0 / dist
                        else:
                            # Enemy detection
                            if seg_idx == 0:
                                if len(other['segments']) < my_length and enemy_smaller == 0:
                                    enemy_smaller = 1.0 / dist
                                elif enemy_bigger == 0:
                                    enemy_bigger = 1.0 / dist
                            elif enemy_body == 0:
                                enemy_body = 1.0 / dist

            # Normalize crowding
            local_crowding = min(local_crowding / 15, 1.0)

            # Store 15 features per direction
            base = ray_idx * 15
            vision[base:base+5] = [health_dist] + gun_dists
            vision[base+5:base+11] = [self_danger, wall_dist, enemy_smaller, enemy_bigger, enemy_body, nearest_enemy_gun]
            vision[base+11:base+15] = [projectile_danger, ally_head, ally_body, local_crowding]

        # State inputs (indices 120-132)
        total_hp = sum(s['hp'] for s in agent['segments'][1:] if s['hp'] != float('inf'))
        total_max = sum(s['maxHp'] for s in agent['segments'][1:] if s['maxHp'] != float('inf'))
        health_ratio = total_hp / total_max if total_max > 0 else 1.0

        dist_to_edge = min(head_x, head_y, self.WORLD_COLS - 1 - head_x, self.WORLD_ROWS - 1 - head_y)
        arena_pos = min(dist_to_edge / (min(self.WORLD_COLS, self.WORLD_ROWS) / 2), 1.0)

        # Threat density
        threat = 0.0
        for i, other in enumerate(self.agents):
            if i == agent_idx or not other['alive']: continue
            oh = other['segments'][0]
            dist = abs(oh['x'] - head_x) + abs(oh['y'] - head_y)
            if dist < 15:
                size_factor = 2.0 if len(other['segments']) > my_length else 0.5
                threat += size_factor / max(dist, 1)
        threat = min(threat / 3, 1.0)

        # Momentum features
        time_since_turn = min((self.step_count - agent['last_turn_time']) / 20, 1.0)
        last_turn_1 = (agent['last_turns'][0] + 1) / 2  # -1,0,1 -> 0,0.5,1
        last_turn_2 = (agent['last_turns'][1] + 1) / 2

        # Invincibility (normalized to ~30 steps max)
        invincibility = max(0, agent['invincible_until'] - self.step_count) / 30

        # Relative speed (all agents have same base speed in training, so ~0.5)
        relative_speed = 0.5

        vision[120:133] = [
            health_ratio, arena_pos, threat, min(my_length/20, 1), 0.25,  # 120-124: health, arena, threat, length, gun
            0, 0, 1,  # 125-127: is_mvp, mvp_time, mvp_dist
            time_since_turn, last_turn_1, last_turn_2, invincibility, relative_speed  # 128-132: new features
        ]
        return vision

    def step(self, actions):
        self.step_count += 1
        rewards = np.zeros(self.num_agents, dtype=np.float32)

        # Apply turns and track momentum
        for i, agent in enumerate(self.agents):
            if not agent['alive']: continue
            old_dir = agent['direction']
            if actions[i] == 0:
                agent['direction'] = (agent['direction'] - 1) % 4
                turn_dir = -1
            elif actions[i] == 2:
                agent['direction'] = (agent['direction'] + 1) % 4
                turn_dir = 1
            else:
                turn_dir = 0

            # Track turns for momentum context
            if turn_dir != 0:
                agent['last_turn_time'] = self.step_count
                agent['last_turns'][1] = agent['last_turns'][0]
                agent['last_turns'][0] = turn_dir

        # Simulate some projectiles (random spawn for training variety)
        if random.random() < 0.1 and len(self.projectiles) < 20:
            # Spawn a projectile from a random alive agent
            alive_agents = [a for a in self.agents if a['alive'] and len(a['segments']) > 1]
            if alive_agents:
                shooter = random.choice(alive_agents)
                head = shooter['segments'][0]
                dx, dy = self.DIRECTIONS[shooter['direction']]
                self.projectiles.append({'x': head['x'] + dx*2, 'y': head['y'] + dy*2, 'vx': dx, 'vy': dy, 'life': 10})

        # Move projectiles
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
                is_teammate = self.team_mode and agent['team'] == other['team']

                if hx == oh['x'] and hy == oh['y']:
                    if is_teammate:
                        # Teammates collide but don't hurt - both bounce back (blocked)
                        # Just a small penalty for bumping into teammate
                        rewards[i] -= 0.1
                        rewards[j] -= 0.1
                    else:
                        # Opponents - normal FFA rules
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
                    if is_teammate:
                        # Teammates block but don't kill
                        rewards[i] -= 0.1
                    else:
                        # Opponent body collision - death
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

        if self.team_mode:
            # Team mode: game ends when one team is eliminated
            team0_alive = sum(1 for a in self.agents if a['alive'] and a['team'] == 0)
            team1_alive = sum(1 for a in self.agents if a['alive'] and a['team'] == 1)
            game_over = team0_alive == 0 or team1_alive == 0 or self.step_count >= 500

            if game_over and (team0_alive == 0 or team1_alive == 0):
                winning_team = 0 if team0_alive > 0 else 1
                for i, a in enumerate(self.agents):
                    if a['team'] == winning_team:
                        rewards[i] += 5.0  # Winning team bonus
                    else:
                        rewards[i] -= 2.0  # Losing team penalty
        else:
            # FFA mode: last one standing wins
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

    def play_episode(self, team_mode=False):
        env = LocomotEnv(num_agents=4, team_mode=team_mode)
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

        # Determine win based on mode
        if team_mode:
            my_team = env.agents[0]['team']
            team_alive = sum(1 for a in env.agents if a['alive'] and a['team'] == my_team)
            enemy_alive = sum(1 for a in env.agents if a['alive'] and a['team'] != my_team)
            won = team_alive > 0 and enemy_alive == 0
        else:
            won = env.agents[0]['alive']

        if won:
            self.wins += 1

        return total_reward, won, len(env.agents[0]['segments'])

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

    def train(self, episodes=EPISODES, team_mode=False, mode_name="FFA"):
        console.print(Panel.fit(
            f"[bold cyan]LOCOMOT.IO Neural Network Training - {mode_name}[/bold cyan]\n"
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

            task = progress.add_task(f"[cyan]{mode_name}...", total=episodes)

            for ep in range(episodes):
                reward, won, length = self.play_episode(team_mode=team_mode)
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
                    self.save_brain(f'brain_{mode_name.lower()}_checkpoint_{ep}.json')
                    self.wins = 0
                    self.total_games = 0

        elapsed = time.time() - start_time

        # Save final brain with mode in filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'brain_{mode_name.lower()}_{timestamp}.json'
        self.save_brain(filename)

        # Results table
        table = Table(title=f"{mode_name} Training Complete!", border_style="green")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Mode", mode_name)
        table.add_row("Time", f"{elapsed:.1f}s")
        table.add_row("Speed", f"{episodes/elapsed:.1f} ep/s")
        table.add_row("Final ε", f"{self.epsilon:.3f}")
        table.add_row("Saved to", filename)
        console.print(table)

        return filename

# ========== MAIN ==========
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train LOCOMOT.IO neural networks')
    parser.add_argument('--mode', choices=['ffa', 'team', 'both'], default='both',
                        help='Training mode: ffa, team, or both (default: both)')
    parser.add_argument('--episodes', type=int, default=EPISODES,
                        help=f'Episodes per mode (default: {EPISODES})')
    args = parser.parse_args()

    console.print("[yellow]LOCOMOT.IO Neural Network Training[/yellow]")
    console.print("[dim]133-input architecture[/dim]\n")

    if args.mode in ['ffa', 'both']:
        console.print("[bold blue]Training FFA Brain...[/bold blue]\n")
        ffa_trainer = Trainer(brain_path=None)
        ffa_file = ffa_trainer.train(args.episodes, team_mode=False, mode_name="FFA")
        console.print(f"[green]FFA brain saved:[/green] {ffa_file}\n")

    if args.mode in ['team', 'both']:
        console.print("[bold red]Training Team Brain...[/bold red]\n")
        team_trainer = Trainer(brain_path=None)
        team_file = team_trainer.train(args.episodes, team_mode=True, mode_name="Team")
        console.print(f"[green]Team brain saved:[/green] {team_file}\n")

    console.print("[bold green]Done![/bold green]")
    console.print("[dim]Copy brain files to index.html to deploy[/dim]")
