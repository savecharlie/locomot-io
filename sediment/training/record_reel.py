#!/usr/bin/env python3
"""Record the bot playing Sediment and extract highlight clips for an ad reel.

Usage:
    python3 record_reel.py [--duration 180] [--output reel.mp4]

Requires: pip install playwright && playwright install chromium
Also requires: ffmpeg
"""
import json, os, sys, subprocess, time, argparse

def main():
    parser = argparse.ArgumentParser(description='Record Sediment bot for ad reel')
    parser.add_argument('--duration', type=int, default=180, help='Recording duration in seconds')
    parser.add_argument('--output', default='reel.mp4', help='Output reel filename')
    parser.add_argument('--url', default='https://locomot.io/sediment/', help='Game URL')
    parser.add_argument('--raw-only', action='store_true', help='Only record raw video, skip cutting')
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    video_dir = '/tmp/sediment_recording'
    os.makedirs(video_dir, exist_ok=True)
    highlights = []
    record_start = [0]  # mutable ref for closure

    print(f"Recording bot for {args.duration}s from {args.url}")
    print("Highlights will be logged to console...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # visible so Ivy can watch
        context = browser.new_context(
            record_video_dir=video_dir,
            record_video_size={"width": 1280, "height": 720},
            viewport={"width": 1280, "height": 720}
        )
        page = context.new_page()

        # Capture console highlight events
        def on_console(msg):
            text = msg.text
            if text.startswith('HIGHLIGHT:'):
                parts = text.split(':')
                event_type = parts[1]
                ts_ms = float(parts[2])
                extra = parts[3] if len(parts) > 3 else ''
                # Convert to seconds since recording start
                elapsed = (ts_ms - record_start[0]) / 1000.0
                highlights.append({
                    'type': event_type,
                    'time': round(elapsed, 2),
                    'extra': extra
                })
                print(f"  [{elapsed:6.1f}s] {event_type} {extra}")

        page.on('console', on_console)

        # Load the game
        page.goto(args.url, wait_until='networkidle')
        time.sleep(2)

        # Get the recording start time by injecting a console marker
        page.evaluate('console.log("HIGHLIGHT:START:" + performance.now().toFixed(0))')
        time.sleep(0.5)
        # Extract start time from the last highlight
        if highlights and highlights[-1]['type'] == 'START':
            # We need the raw performance.now value, re-parse
            pass
        # Simpler: just record performance.now at start
        start_perf = float(page.evaluate('performance.now()'))
        record_start[0] = start_perf
        highlights.clear()  # remove the START marker

        # Start the game with a tap/click, then enable bot
        page.click('canvas', position={"x": 640, "y": 360})
        time.sleep(1)

        # Click the brain button to enable the bot
        page.click('#botBtn')
        print("Bot enabled! Recording...")
        time.sleep(0.5)

        # Click canvas to start the game (space/tap to begin)
        page.click('canvas', position={"x": 640, "y": 360})

        # Wait for the duration
        start_wall = time.time()
        while time.time() - start_wall < args.duration:
            time.sleep(1)
            elapsed = time.time() - start_wall
            if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                print(f"  ... {int(elapsed)}s / {args.duration}s")

        print(f"\nRecording complete. {len(highlights)} highlight events captured.")

        # Close to finalize video
        page.close()
        context.close()
        browser.close()

    # Find the recorded video file
    videos = sorted(
        [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.webm')],
        key=os.path.getmtime, reverse=True
    )
    if not videos:
        print("ERROR: No video file found!")
        return

    raw_video = videos[0]
    print(f"Raw video: {raw_video}")

    # Save highlights JSON
    highlights_file = os.path.join(video_dir, 'highlights.json')
    with open(highlights_file, 'w') as f:
        json.dump(highlights, f, indent=2)
    print(f"Highlights saved: {highlights_file}")

    if args.raw_only:
        print("Raw-only mode — skipping reel cutting.")
        return

    # ── Cut highlight reel ──
    # Priority: RESURRECT > COMBO > LEVEL_COMPLETE > QUAKE > DEATH
    PRIORITY = {'LEVEL_COMPLETE': 5, 'RESURRECT': 4, 'COMBO': 3, 'QUAKE': 2, 'DEATH': 1}

    # Sort highlights by priority (desc), then time
    scored = sorted(highlights, key=lambda h: (-PRIORITY.get(h['type'], 0), h['time']))

    # Pick best highlights, spaced at least 5s apart
    selected = []
    for h in scored:
        if len(selected) >= 8:
            break
        if any(abs(h['time'] - s['time']) < 5 for s in selected):
            continue
        selected.append(h)

    # Sort selected by time for chronological order
    selected.sort(key=lambda h: h['time'])

    if not selected:
        print("No highlights to cut! Try a longer recording.")
        return

    print(f"\nCutting {len(selected)} highlight clips:")

    # Create individual clips
    clips = []
    for i, h in enumerate(selected):
        # 3 seconds before, 2 seconds after the event
        start = max(0, h['time'] - 3)
        duration = 5
        clip_file = os.path.join(video_dir, f'clip_{i:02d}.mp4')
        clips.append(clip_file)

        print(f"  Clip {i}: {h['type']} at {h['time']:.1f}s [{start:.1f}s → {start+duration:.1f}s]")

        subprocess.run([
            'ffmpeg', '-y', '-ss', str(start), '-i', raw_video,
            '-t', str(duration), '-c:v', 'libx264', '-preset', 'fast',
            '-crf', '18', '-an', clip_file
        ], capture_output=True)

    # Create concat file
    concat_file = os.path.join(video_dir, 'concat.txt')
    with open(concat_file, 'w') as f:
        for clip in clips:
            f.write(f"file '{clip}'\n")

    # Concatenate clips with crossfade
    output_path = os.path.join(os.path.dirname(__file__), '..', args.output)
    subprocess.run([
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_file,
        '-c:v', 'libx264', '-preset', 'slow', '-crf', '16',
        '-vf', 'fps=60', '-an', output_path
    ], capture_output=True)

    print(f"\nReel saved: {output_path}")
    print(f"Duration: ~{len(selected) * 5}s")
    print("Selected highlights:")
    for h in selected:
        print(f"  {h['time']:6.1f}s  {h['type']} {h['extra']}")


if __name__ == '__main__':
    main()
