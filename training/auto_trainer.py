#!/usr/bin/env python3
"""
LOCOMOT.IO Auto-Trainer Daemon
Watches for new player data and automatically creates/updates player bots.

Run: python3 auto_trainer.py
  - Polls every 5 minutes for new behavioral data
  - Creates bots for players with 3+ sessions
  - Updates existing bots when player has 2x the data

Can run alongside arena.py daemon for full automation.
"""

import requests
import time
import json
from datetime import datetime
from collections import defaultdict

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'rich', '-q'])
    from rich.console import Console
    from rich.table import Table
    console = Console()

SERVER_URL = 'https://locomot-io.savecharlie.partykit.dev/party/collective'

# Config
POLL_INTERVAL = 5 * 60  # 5 minutes
MIN_SESSIONS_FOR_BOT = 3  # Create bot after 3 sessions
TOP_SESSIONS = 5  # Train on top N sessions
QUALITY_CUTOFF = 0.80  # Also use sessions >= 80% of best quality


def fetch_manifest():
    """Get current genome manifest."""
    try:
        resp = requests.post(SERVER_URL, json={'type': 'get_genome_manifest'}, timeout=30)
        return resp.json()
    except:
        return None


def fetch_behavioral_data():
    """Fetch all behavioral sessions."""
    try:
        resp = requests.post(SERVER_URL, json={
            'type': 'get_behavioral_data',
            'limit': 2000
        }, timeout=60)
        return resp.json().get('sessions', [])
    except Exception as e:
        console.print(f"[red]Fetch error: {e}[/red]")
        return []


def get_player_stats(sessions):
    """Group sessions by player and count."""
    stats = defaultdict(lambda: {'sessions': 0, 'frames': 0, 'latest': 0, 'top5_hash': ''})

    # First pass: group sessions by player
    player_sessions = defaultdict(list)
    for s in sessions:
        pid = s.get('playerId', 'anon')
        if pid == 'anon':
            continue
        player_sessions[pid].append(s)

    # Second pass: compute stats including top 5 hash
    for pid, psessions in player_sessions.items():
        stats[pid]['sessions'] = len(psessions)
        stats[pid]['frames'] = sum(len(s.get('frames', [])) for s in psessions)
        stats[pid]['latest'] = max(s.get('timestamp', 0) for s in psessions)

        # Get top 5 qualities and create a hash to detect changes
        qualities = sorted([s.get('quality', 0) for s in psessions], reverse=True)[:TOP_SESSIONS]
        # Hash is the top 5 qualities rounded to 2 decimals
        stats[pid]['top5_hash'] = ','.join(f'{q:.2f}' for q in qualities)
        stats[pid]['top5_qualities'] = qualities

    return dict(stats)


def get_existing_bots(manifest):
    """Get dict of player_id -> genome info for behavioral bots."""
    bots = {}
    for g in manifest.get('genomes', []):
        source = g.get('source', '')
        if source.startswith('player:'):
            player_id = source.replace('player:', '')
            bots[player_id] = g
    return bots


def filter_best_sessions(sessions):
    """Filter to only use top sessions for training.

    Uses: top N sessions OR sessions >= 80% of best quality
    This ensures bots learn from your best gameplay, not mediocre runs.
    """
    if not sessions:
        return sessions

    # Sort by quality descending
    sorted_sessions = sorted(sessions, key=lambda x: x.get('quality', 0), reverse=True)
    best_quality = sorted_sessions[0].get('quality', 0)

    if best_quality == 0:
        return sorted_sessions[:TOP_SESSIONS]

    # Take top N
    best = sorted_sessions[:TOP_SESSIONS]

    # Also add any sessions >= 80% of best quality
    cutoff = best_quality * QUALITY_CUTOFF
    for s in sorted_sessions[TOP_SESSIONS:]:
        if s.get('quality', 0) >= cutoff:
            best.append(s)

    return best


def train_player_bot(player_id, sessions):
    """Train a bot for a specific player."""
    # Filter to only best sessions
    best_sessions = filter_best_sessions(sessions)
    console.print(f"[cyan]Training bot for {player_id} ({len(best_sessions)}/{len(sessions)} best sessions)...[/cyan]")

    # Import training module
    import sys
    sys.path.insert(0, '/home/ivy/locomot-io/training')
    from train_behavioral import BehavioralNet, ReplayBuffer, deploy_to_server
    import torch
    import torch.nn as nn
    import torch.optim as optim

    device = torch.device('cpu')

    # Build buffer from best sessions only
    buffer = ReplayBuffer()
    for session in best_sessions:
        buffer.add_session(session)

    if len(buffer) < 100:
        console.print(f"[yellow]Not enough frames for {player_id} ({len(buffer)})[/yellow]")
        return False

    console.print(f"[green]Training on {len(buffer)} frames...[/green]")

    # Train
    model = BehavioralNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    states, actions = buffer.get_all()
    states = states.to(device)
    actions = actions.to(device)

    batch_size = 64
    epochs = 30

    for epoch in range(epochs):
        indices = torch.randperm(len(states))
        total_loss = 0
        batches = 0

        for i in range(0, len(states), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_states = states[batch_idx]
            batch_actions = actions[batch_idx]

            optimizer.zero_grad()
            outputs = model(batch_states)
            loss = criterion(outputs, batch_actions)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

    console.print(f"[green]Training complete (loss: {total_loss/batches:.4f})[/green]")

    # Deploy to pool
    success = deploy_to_server(model, player_id=player_id)
    return success


def run_daemon():
    """Main daemon loop."""
    console.print("[bold magenta]LOCOMOT.IO Auto-Trainer Daemon[/bold magenta]")
    console.print(f"Poll interval: {POLL_INTERVAL}s")
    console.print(f"Min sessions for bot: {MIN_SESSIONS_FOR_BOT}")
    console.print(f"Retrain when: top {TOP_SESSIONS} changes")
    console.print("=" * 50)

    # Track what we've trained
    trained_top5 = {}  # player_id -> top5_hash when last trained

    while True:
        try:
            console.print(f"\n[dim]{datetime.now().strftime('%H:%M:%S')} Checking for new players...[/dim]")

            # Fetch data
            sessions = fetch_behavioral_data()
            manifest = fetch_manifest()

            if not sessions or not manifest:
                console.print("[yellow]Could not fetch data, retrying...[/yellow]")
                time.sleep(60)
                continue

            # Get player stats
            player_stats = get_player_stats(sessions)
            existing_bots = get_existing_bots(manifest)

            # Find players needing bots
            actions_taken = []

            for player_id, stats in player_stats.items():
                session_count = stats['sessions']
                top5_hash = stats['top5_hash']
                top5 = stats.get('top5_qualities', [])

                # Skip if not enough sessions
                if session_count < MIN_SESSIONS_FOR_BOT:
                    continue

                # Check if needs training
                last_top5 = trained_top5.get(player_id, '')

                if player_id not in existing_bots:
                    # New player, create bot
                    top5_str = ', '.join(f'{q:.2f}' for q in top5[:3])
                    console.print(f"\n[green]New player: {player_id} (top scores: {top5_str}...)[/green]")
                    player_sessions = [s for s in sessions if s.get('playerId') == player_id]

                    if train_player_bot(player_id, player_sessions):
                        trained_top5[player_id] = top5_hash
                        actions_taken.append(f"Created {player_id}Bot")

                elif top5_hash != last_top5:
                    # Top 5 changed - a new high score entered the top!
                    console.print(f"\n[magenta]🏆 {player_id}'s top 5 changed! Retraining on best runs...[/magenta]")
                    player_sessions = [s for s in sessions if s.get('playerId') == player_id]

                    if train_player_bot(player_id, player_sessions):
                        trained_top5[player_id] = top5_hash
                        actions_taken.append(f"🏆 {player_id}Bot improved!")

            # Summary
            if actions_taken:
                console.print(f"\n[bold green]Actions: {', '.join(actions_taken)}[/bold green]")
            else:
                # Show current state
                table = Table(title="Player Bot Status", show_header=True)
                table.add_column("Player", style="cyan")
                table.add_column("Sessions", style="green")
                table.add_column("Has Bot", style="yellow")

                for pid, stats in sorted(player_stats.items(), key=lambda x: -x[1]['sessions'])[:10]:
                    has_bot = "✓" if pid in existing_bots else "✗"
                    table.add_row(pid, str(stats['sessions']), has_bot)

                console.print(table)

            console.print(f"[dim]Next check in {POLL_INTERVAL//60} minutes...[/dim]")
            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping...[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            time.sleep(60)


if __name__ == '__main__':
    run_daemon()
