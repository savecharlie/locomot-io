#!/usr/bin/env python3
"""
LOCOMOT.IO Evolution Daemon
The complete automated training system.

Runs two parallel processes:
1. Auto-trainer: Creates bots from player behavioral data
2. Arena: Runs tournaments, fine-tunes winners, breeds champions, culls losers

Usage:
  python3 evolution_daemon.py                    # Run forever
  python3 evolution_daemon.py --hours 8          # Run for 8 hours (overnight)
  python3 evolution_daemon.py --arena-only       # Just arena training
  python3 evolution_daemon.py --behavioral-only  # Just player bot creation
"""

import subprocess
import sys
import time
import signal
import argparse
from datetime import datetime, timedelta

try:
    from rich.console import Console
    from rich.panel import Panel
    console = Console()
except ImportError:
    subprocess.check_call(['pip', 'install', 'rich', '-q'])
    from rich.console import Console
    from rich.panel import Panel
    console = Console()

# Global process handles
processes = []


def signal_handler(sig, frame):
    console.print("\n[yellow]Shutting down...[/yellow]")
    for p in processes:
        p.terminate()
    sys.exit(0)


def run_evolution(hours=None, arena_only=False, behavioral_only=False):
    """Run the evolution system."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    end_time = None
    if hours:
        end_time = datetime.now() + timedelta(hours=hours)

    console.print(Panel.fit(
        "[bold magenta]LOCOMOT.IO EVOLUTION DAEMON[/bold magenta]\n"
        "[dim]Bots evolve while you sleep...[/dim]",
        border_style="magenta"
    ))

    if hours:
        console.print(f"[cyan]Running for {hours} hours (until {end_time.strftime('%H:%M')})[/cyan]")
    else:
        console.print("[cyan]Running indefinitely (Ctrl+C to stop)[/cyan]")

    console.print()

    # Start processes
    if not arena_only:
        console.print("[green]Starting Auto-Trainer (creates player bots)...[/green]")
        p1 = subprocess.Popen(
            [sys.executable, 'auto_trainer.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        processes.append(p1)

    if not behavioral_only:
        console.print("[green]Starting Arena Daemon (tournaments + evolution)...[/green]")
        cmd = [sys.executable, 'arena.py', '--daemon', '--cycle-hours', '2']
        if hours:
            cycles = max(1, int(hours / 2))
            cmd.extend(['--cycles', str(cycles)])
        p2 = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        processes.append(p2)

    console.print()
    console.print("[bold green]Evolution running![/bold green]")
    console.print("[dim]Check genome pool for new bots appearing...[/dim]")
    console.print()

    # Monitor
    try:
        while True:
            # Check if we should stop
            if end_time and datetime.now() >= end_time:
                console.print("\n[yellow]Time limit reached, stopping...[/yellow]")
                break

            # Check process status
            for p in processes:
                if p.poll() is not None:
                    console.print(f"[yellow]Process exited with code {p.returncode}[/yellow]")

            # Print any output (non-blocking)
            for p in processes:
                if p.stdout:
                    import select
                    if select.select([p.stdout], [], [], 0)[0]:
                        line = p.stdout.readline()
                        if line:
                            console.print(f"[dim]{line.rstrip()}[/dim]")

            time.sleep(1)

    except KeyboardInterrupt:
        pass

    # Cleanup
    console.print("\n[yellow]Stopping processes...[/yellow]")
    for p in processes:
        p.terminate()
        p.wait()

    console.print("[green]Evolution daemon stopped.[/green]")


def main():
    parser = argparse.ArgumentParser(description='LOCOMOT.IO Evolution Daemon')
    parser.add_argument('--hours', type=float, help='Run for N hours then stop')
    parser.add_argument('--arena-only', action='store_true', help='Only run arena training')
    parser.add_argument('--behavioral-only', action='store_true', help='Only run behavioral bot creation')
    args = parser.parse_args()

    run_evolution(
        hours=args.hours,
        arena_only=args.arena_only,
        behavioral_only=args.behavioral_only
    )


if __name__ == '__main__':
    main()
