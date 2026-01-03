#!/usr/bin/env python3
"""
LOCOMOT.IO Continuous Behavioral Training Daemon
Trains on human player data, updating as new data arrives.

Run: python3 train_behavioral.py [--once] [--deploy]
  --once: Train once on all available data, then exit
  --deploy: Auto-deploy trained model to server after each training cycle
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import requests
import argparse
import time
import signal
import sys
from datetime import datetime
from pathlib import Path
from collections import deque
import random

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'rich', '-q'])
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel

console = Console()

# ========== CONFIG ==========
SERVER_URL = 'https://locomot-io.savecharlie.partykit.dev/party/collective'
INPUT_SIZE = 133
HIDDEN1_SIZE = 192
HIDDEN2_SIZE = 96
OUTPUT_SIZE = 4  # 0=left, 1=straight, 2=right, 3=shoot

# Training hyperparams
BATCH_SIZE = 64
LEARNING_RATE = 0.001
REPLAY_BUFFER_SIZE = 100000  # Max frames to keep in memory
MIN_BUFFER_SIZE = 500  # Need at least this many frames to start training
EPOCHS_PER_CYCLE = 10  # Epochs per training cycle on new data
POLL_INTERVAL = 30  # Seconds between polling for new data
CHECKPOINT_INTERVAL = 300  # Save checkpoint every 5 minutes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global flag for graceful shutdown
running = True


def signal_handler(sig, frame):
    global running
    console.print("\n[yellow]Shutting down gracefully...[/yellow]")
    running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ========== NETWORK ==========
class BehavioralNet(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE):
        super().__init__()
        self.input_size = input_size
        self.net = nn.Sequential(
            nn.Linear(input_size, HIDDEN1_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN1_SIZE, HIDDEN2_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN2_SIZE, output_size)
        )

    def forward(self, x):
        return self.net(x)


# ========== EXPERIENCE REPLAY BUFFER ==========
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, max_size=REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=max_size)
        self.seen_timestamps = set()  # Track which sessions we've already added

    def add_session(self, session):
        """Add frames from a session to the buffer."""
        timestamp = session.get('timestamp', 0)
        if timestamp in self.seen_timestamps:
            return 0  # Already have this session

        frames = session.get('frames', [])
        added = 0
        for frame in frames:
            state = frame.get('s', [])
            action = frame.get('a', 1)  # Default to straight

            if len(state) >= INPUT_SIZE:
                self.buffer.append((state[:INPUT_SIZE], action))
                added += 1

        if added > 0:
            self.seen_timestamps.add(timestamp)

        return added

    def sample(self, batch_size):
        """Sample a random batch from the buffer."""
        batch = random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
        states = torch.FloatTensor([s for s, a in batch])
        actions = torch.LongTensor([a for s, a in batch])
        return states, actions

    def get_all(self):
        """Get all data as tensors (for initial training)."""
        if len(self.buffer) == 0:
            return None, None
        states = torch.FloatTensor([s for s, a in self.buffer])
        actions = torch.LongTensor([a for s, a in self.buffer])
        return states, actions

    def __len__(self):
        return len(self.buffer)

    def get_action_distribution(self):
        """Get distribution of actions in buffer."""
        if len(self.buffer) == 0:
            return {}
        actions = [a for s, a in self.buffer]
        dist = {}
        for a in range(OUTPUT_SIZE):
            count = sum(1 for x in actions if x == a)
            dist[a] = count / len(actions)
        return dist


# ========== DATA FETCHING ==========
def fetch_all_behavioral_data():
    """Fetch all behavioral sessions from the server."""
    console.print("[yellow]Fetching all behavioral data...[/yellow]")

    try:
        response = requests.post(SERVER_URL, json={
            'type': 'get_behavioral_data',
            'limit': 1000
        }, timeout=60)

        if response.status_code != 200:
            console.print(f"[red]Server error: {response.status_code}[/red]")
            return []

        data = response.json()
        console.print(f"[green]Fetched {data['count']} sessions, {data['totalFrames']} total frames[/green]")
        return data.get('sessions', [])

    except Exception as e:
        console.print(f"[red]Fetch error: {e}[/red]")
        return []


def fetch_new_behavioral_data(since_timestamp):
    """Fetch only sessions newer than the given timestamp."""
    try:
        response = requests.post(SERVER_URL, json={
            'type': 'get_new_behavioral',
            'since': since_timestamp
        }, timeout=30)

        if response.status_code != 200:
            return []

        data = response.json()
        return data.get('sessions', [])

    except Exception as e:
        console.print(f"[red]Fetch error: {e}[/red]")
        return []


# ========== TRAINING ==========
def train_epoch(model, optimizer, criterion, buffer, batch_size=BATCH_SIZE):
    """Train for one epoch by sampling from replay buffer."""
    model.train()

    # Number of batches to train on = buffer_size / batch_size
    n_batches = max(1, len(buffer) // batch_size)
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for _ in range(n_batches):
        states, actions = buffer.sample(batch_size)
        states = states.to(device)
        actions = actions.to(device)

        optimizer.zero_grad()
        outputs = model(states)
        loss = criterion(outputs, actions)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        total_correct += (preds == actions).sum().item()
        total_samples += len(actions)

    avg_loss = total_loss / n_batches
    accuracy = total_correct / total_samples if total_samples > 0 else 0

    return avg_loss, accuracy


def train_on_buffer(model, optimizer, criterion, buffer, epochs=EPOCHS_PER_CYCLE):
    """Train for multiple epochs on the current buffer."""
    best_acc = 0

    for epoch in range(epochs):
        loss, acc = train_epoch(model, optimizer, criterion, buffer)
        best_acc = max(best_acc, acc)

        if (epoch + 1) % 2 == 0:
            console.print(f"  Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}, Acc: {acc*100:.1f}%")

    return best_acc


# ========== EXPORT / DEPLOY ==========
def export_to_json(model, filename):
    """Export model to JSON format compatible with the game."""
    state_dict = model.state_dict()

    brain = {
        'input_size': INPUT_SIZE,
        'type': 'behavioral',
        'trained_at': datetime.now().isoformat()
    }

    for key, tensor in state_dict.items():
        brain[key] = tensor.cpu().numpy().tolist()

    with open(filename, 'w') as f:
        json.dump(brain, f)

    return filename


def deploy_to_server(model):
    """Upload trained model weights to the server."""
    console.print("[yellow]Deploying model to server...[/yellow]")

    state_dict = model.state_dict()
    brain = {
        'input_size': INPUT_SIZE,
        'type': 'behavioral',
        'trained_at': datetime.now().isoformat()
    }

    for key, tensor in state_dict.items():
        brain[key] = tensor.cpu().numpy().tolist()

    try:
        response = requests.post(SERVER_URL, json={
            'type': 'set_brain',
            'brain': brain
        }, timeout=30)

        if response.status_code == 200:
            console.print("[green]Model deployed to server![/green]")
            return True
        else:
            console.print(f"[red]Deploy failed: {response.status_code}[/red]")
            return False
    except Exception as e:
        console.print(f"[red]Deploy error: {e}[/red]")
        return False


# ========== MAIN DAEMON ==========
def run_daemon(auto_deploy=False):
    """Run the continuous training daemon."""
    global running

    console.print("[bold magenta]LOCOMOT.IO Continuous Training Daemon[/bold magenta]")
    console.print(f"Device: [cyan]{device}[/cyan]")
    console.print(f"Auto-deploy: [cyan]{auto_deploy}[/cyan]")
    console.print("=" * 50)

    # Initialize
    buffer = ReplayBuffer()
    model = BehavioralNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    last_timestamp = 0
    last_checkpoint = time.time()
    training_cycles = 0

    # Initial data fetch
    console.print("\n[bold]Initial data load...[/bold]")
    sessions = fetch_all_behavioral_data()

    for session in sessions:
        added = buffer.add_session(session)
        ts = session.get('timestamp', 0)
        if ts > last_timestamp:
            last_timestamp = ts

    console.print(f"[green]Buffer: {len(buffer)} frames from {len(sessions)} sessions[/green]")

    # Show action distribution
    dist = buffer.get_action_distribution()
    if dist:
        console.print("\n[bold]Action distribution:[/bold]")
        action_names = ['Left', 'Straight', 'Right', 'Shoot']
        for a, pct in dist.items():
            name = action_names[a] if a < len(action_names) else f'Action {a}'
            console.print(f"  {name}: {pct*100:.1f}%")

    # Initial training if we have enough data
    if len(buffer) >= MIN_BUFFER_SIZE:
        console.print(f"\n[bold]Initial training ({EPOCHS_PER_CYCLE * 3} epochs)...[/bold]")
        best_acc = train_on_buffer(model, optimizer, criterion, buffer, epochs=EPOCHS_PER_CYCLE * 3)
        console.print(f"[green]Initial training complete. Best acc: {best_acc*100:.1f}%[/green]")
        training_cycles += 1

        # Save initial checkpoint
        export_to_json(model, 'brain_behavioral_latest.json')

        if auto_deploy:
            deploy_to_server(model)
    else:
        console.print(f"[yellow]Need {MIN_BUFFER_SIZE - len(buffer)} more frames before training...[/yellow]")

    # Main loop
    console.print(f"\n[bold]Polling for new data every {POLL_INTERVAL}s...[/bold]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    while running:
        try:
            time.sleep(POLL_INTERVAL)

            if not running:
                break

            # Fetch new sessions
            new_sessions = fetch_new_behavioral_data(last_timestamp)

            if new_sessions:
                new_frames = 0
                for session in new_sessions:
                    added = buffer.add_session(session)
                    new_frames += added
                    ts = session.get('timestamp', 0)
                    if ts > last_timestamp:
                        last_timestamp = ts

                if new_frames > 0:
                    console.print(f"[green]+{new_frames} new frames from {len(new_sessions)} sessions[/green]")

                    # Train on updated buffer
                    if len(buffer) >= MIN_BUFFER_SIZE:
                        console.print(f"[cyan]Training cycle {training_cycles + 1}...[/cyan]")
                        best_acc = train_on_buffer(model, optimizer, criterion, buffer)
                        training_cycles += 1
                        console.print(f"[green]Cycle complete. Acc: {best_acc*100:.1f}%, Buffer: {len(buffer)}[/green]")

                        # Deploy if enabled
                        if auto_deploy:
                            deploy_to_server(model)

            # Periodic checkpoint
            if time.time() - last_checkpoint > CHECKPOINT_INTERVAL and len(buffer) >= MIN_BUFFER_SIZE:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'brain_behavioral_{timestamp}.json'
                export_to_json(model, filename)
                export_to_json(model, 'brain_behavioral_latest.json')
                console.print(f"[dim]Checkpoint saved: {filename}[/dim]")
                last_checkpoint = time.time()

        except KeyboardInterrupt:
            break

    # Final save
    if len(buffer) >= MIN_BUFFER_SIZE:
        console.print("\n[yellow]Saving final checkpoint...[/yellow]")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_to_json(model, f'brain_behavioral_{timestamp}.json')
        export_to_json(model, 'brain_behavioral_latest.json')
        console.print("[green]Final checkpoint saved![/green]")

    console.print(f"\n[bold]Training stats:[/bold]")
    console.print(f"  Total frames: {len(buffer)}")
    console.print(f"  Training cycles: {training_cycles}")
    console.print("[bold green]Daemon stopped.[/bold green]")


def run_once(deploy=False):
    """Train once on all available data, then exit."""
    console.print("[bold magenta]LOCOMOT.IO Behavioral Training (One-shot)[/bold magenta]")
    console.print(f"Device: [cyan]{device}[/cyan]")
    console.print("=" * 50)

    # Fetch all data
    sessions = fetch_all_behavioral_data()

    if not sessions:
        console.print("[red]No sessions found![/red]")
        return

    # Build buffer
    buffer = ReplayBuffer()
    for session in sessions:
        buffer.add_session(session)

    console.print(f"[green]Loaded {len(buffer)} frames from {len(sessions)} sessions[/green]")

    if len(buffer) < MIN_BUFFER_SIZE:
        console.print(f"[red]Need at least {MIN_BUFFER_SIZE} frames to train[/red]")
        return

    # Show action distribution
    dist = buffer.get_action_distribution()
    table = Table(title="Action Distribution")
    table.add_column("Action", style="cyan")
    table.add_column("Percent", style="green")

    action_names = ['Left', 'Straight', 'Right', 'Shoot']
    for a in range(OUTPUT_SIZE):
        name = action_names[a] if a < len(action_names) else f'Action {a}'
        pct = dist.get(a, 0) * 100
        table.add_row(name, f"{pct:.1f}%")
    console.print(table)

    # Initialize model
    model = BehavioralNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Split for validation
    states, actions = buffer.get_all()
    n = len(states)
    n_val = int(n * 0.1)
    indices = torch.randperm(n)

    train_states = states[indices[n_val:]].to(device)
    train_actions = actions[indices[n_val:]].to(device)
    val_states = states[indices[:n_val]].to(device)
    val_actions = actions[indices[:n_val]].to(device)

    console.print(f"[cyan]Train: {len(train_states)}, Val: {len(val_states)}[/cyan]")

    # Train with progress bar
    epochs = 50
    best_val_acc = 0
    best_model_state = None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("[cyan]Training...", total=epochs)

        for epoch in range(epochs):
            model.train()

            # Shuffle training data
            perm = torch.randperm(len(train_states), device=device)
            train_states = train_states[perm]
            train_actions = train_actions[perm]

            epoch_loss = 0
            n_batches = 0

            for i in range(0, len(train_states), BATCH_SIZE):
                batch_states = train_states[i:i+BATCH_SIZE]
                batch_actions = train_actions[i:i+BATCH_SIZE]

                optimizer.zero_grad()
                outputs = model(batch_states)
                loss = criterion(outputs, batch_actions)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(val_states)
                val_preds = val_outputs.argmax(dim=1)
                val_acc = (val_preds == val_actions).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            progress.update(task, advance=1,
                          description=f"[cyan]Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/n_batches:.4f} | Val: {val_acc*100:.1f}%")

    # Load best model
    if best_model_state:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    console.print(f"\n[green]Best validation accuracy: {best_val_acc*100:.1f}%[/green]")

    # Export
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'brain_behavioral_{timestamp}.json'
    export_to_json(model, filename)
    export_to_json(model, 'brain_behavioral_latest.json')

    console.print(f"[green]Saved: {filename}[/green]")
    console.print(f"[green]Saved: brain_behavioral_latest.json[/green]")

    if deploy:
        deploy_to_server(model)

    console.print("\n[bold green]Training complete![/bold green]")


def main():
    parser = argparse.ArgumentParser(description='LOCOMOT.IO Behavioral Training')
    parser.add_argument('--once', action='store_true', help='Train once then exit')
    parser.add_argument('--deploy', action='store_true', help='Auto-deploy to server')
    args = parser.parse_args()

    if args.once:
        run_once(deploy=args.deploy)
    else:
        run_daemon(auto_deploy=args.deploy)


if __name__ == '__main__':
    main()
