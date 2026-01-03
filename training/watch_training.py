#!/usr/bin/env python3
"""
Training Watcher - Shows live progress with smartness indicator
Run: python3 watch_training.py
"""

import time
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

console = Console()

def get_smartness(win_rate, reward):
    """Convert stats to a fun smartness level"""
    if win_rate >= 0.6:
        return "🧠🧠🧠 GENIUS", "bold green"
    elif win_rate >= 0.45:
        return "🧠🧠 CLEVER", "green"
    elif win_rate >= 0.3:
        return "🧠 LEARNING", "yellow"
    elif win_rate >= 0.15:
        return "🐣 BABY STEPS", "cyan"
    else:
        return "🥒 PICKLE BRAIN", "magenta"

def get_trend(history):
    """Show if rewards are improving"""
    if len(history) < 10:
        return "📊 gathering data...", "dim"
    recent = sum(history[-10:]) / 10
    older = sum(history[:10]) / 10
    diff = recent - older
    if diff > 10:
        return "🚀 SKYROCKETING", "bold green"
    elif diff > 3:
        return "📈 improving", "green"
    elif diff > -3:
        return "➡️  stable", "yellow"
    else:
        return "📉 struggling", "red"

def watch():
    progress_file = '/tmp/training_progress.txt'
    reward_history = []
    last_mode = None

    with Live(console=console, refresh_per_second=4) as live:
        while True:
            try:
                with open(progress_file, 'r') as f:
                    line = f.read().strip()

                if not line:
                    raise FileNotFoundError

                parts = line.split(',')
                mode = parts[0]
                ep = int(parts[1])
                total = int(parts[2])
                reward = float(parts[3])
                win_rate = float(parts[4])
                epsilon = float(parts[5])

                # Reset history on mode change
                if mode != last_mode:
                    reward_history = []
                    last_mode = mode

                reward_history.append(reward)
                if len(reward_history) > 100:
                    reward_history.pop(0)

                progress_pct = (ep / total) * 100
                mode_color = "red" if mode == "Team" else "cyan"

                # Build progress bar
                bar_width = 30
                filled = int(bar_width * ep / total)
                bar = "█" * filled + "░" * (bar_width - filled)

                smartness, smart_color = get_smartness(win_rate, reward)
                trend, trend_color = get_trend(reward_history)

                # ETA calculation
                if ep > 100:
                    # Rough estimate: ~4.5 ep/s on CPU
                    remaining_ep = total - ep
                    eta_sec = remaining_ep / 4.5
                    eta_min = int(eta_sec // 60)
                    eta_str = f"~{eta_min}m left"
                else:
                    eta_str = "calculating..."

                table = Table(show_header=False, box=None, padding=(0, 2))
                table.add_column("Label", style="dim", width=12)
                table.add_column("Value", style="bold")

                table.add_row("Mode", f"[{mode_color}]{'🎮 ' + mode.upper()}[/]")
                table.add_row("Progress", f"[{mode_color}]{bar}[/] {progress_pct:.1f}%")
                table.add_row("Episode", f"{ep:,} / {total:,}  [dim]({eta_str})[/]")
                table.add_row("", "")
                table.add_row("Win Rate", f"[{'green' if win_rate >= 0.4 else 'yellow' if win_rate >= 0.2 else 'red'}]{win_rate:.0%}[/]")
                table.add_row("Reward", f"[{'green' if reward > 20 else 'yellow' if reward > 0 else 'red'}]{reward:+.1f}[/]")
                table.add_row("Exploration", f"[dim]ε = {epsilon:.3f}[/]")
                table.add_row("", "")
                table.add_row("Intelligence", f"[{smart_color}]{smartness}[/]")
                table.add_row("Trend", f"[{trend_color}]{trend}[/]")

                panel = Panel(
                    table,
                    title="[bold]🚂 LOCOMOT.IO Brain Training[/]",
                    subtitle="[dim]Ctrl+C to exit[/]",
                    border_style=mode_color
                )

                live.update(panel)

            except FileNotFoundError:
                live.update(Panel(
                    "[yellow]⏳ Waiting for training to start...[/]\n\n[dim]Run: python3 train_local.py --mode both --episodes 10000[/]",
                    title="[bold]🚂 LOCOMOT.IO Brain Training[/]",
                    border_style="yellow"
                ))
            except Exception as e:
                live.update(Panel(f"[red]Error: {e}[/]"))

            time.sleep(0.25)

if __name__ == '__main__':
    console.print("\n[bold cyan]🔍 Watching training progress...[/]\n")
    try:
        watch()
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped watching.[/]")
