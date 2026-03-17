#!/usr/bin/env python3
"""
Sediment Store Listing Capture Rig
===================================
Bot plays the game. We record video and screenshot key moments.
Then Haiku picks the best screenshots for the Play Store listing.

Usage:
    python3 capture_gameplay.py              # capture ~5 minutes of gameplay
    python3 capture_gameplay.py --duration 600  # 10 minutes
    python3 capture_gameplay.py --rank        # just rank existing screenshots with Haiku

Output:
    ~/sediment-twa/captures/videos/   — gameplay recordings (.webm)
    ~/sediment-twa/captures/shots/    — all screenshots
    ~/sediment-twa/captures/best/     — Haiku's top picks (copied + renamed)
"""

import asyncio
import json
import os
import sys
import time
import shutil
import urllib.request
from pathlib import Path
from datetime import datetime

# Dirs
CAPTURE_DIR = Path.home() / "sediment-twa" / "captures"
SHOTS_DIR = CAPTURE_DIR / "shots"
BEST_DIR = CAPTURE_DIR / "best"
VIDEO_DIR = CAPTURE_DIR / "videos"

for d in [SHOTS_DIR, BEST_DIR, VIDEO_DIR]:
    d.mkdir(parents=True, exist_ok=True)

GAME_URL = "https://locomot.io/sediment/?autoplay=1&dev=1"
VIEWPORT = {"width": 1920, "height": 1080}
DURATION = 300  # seconds default
START_LEVEL = 1  # can override with --level N

# JS to inject — hooks into game state and exposes it
HOOK_JS = """
window._capture = {
    events: [],
    lastLevel: 0,
    lastDeaths: 0,
    lastState: '',
    shotCounter: 0
};

// Poll game state every 200ms
setInterval(() => {
    const c = window._capture;
    try {
        const level = typeof currentLevel !== 'undefined' ? currentLevel : 0;
        const deaths = typeof player !== 'undefined' ? (player.deathCount || 0) : 0;
        const dead = typeof player !== 'undefined' ? player.dead : false;
        const isStarted = typeof started !== 'undefined' ? started : false;
        const isPaused = typeof paused !== 'undefined' ? paused : false;
        const isLevelComplete = typeof levelComplete !== 'undefined' ? levelComplete : false;
        const isGameWon = typeof gameWon !== 'undefined' ? gameWon : false;
        const quakes = typeof player !== 'undefined' ? (player.quakeCount || 0) : 0;

        let state = 'playing';
        if (!isStarted) state = 'start';
        else if (isGameWon) state = 'won';
        else if (isLevelComplete) state = 'level_complete';
        else if (isPaused) state = 'paused';
        else if (dead) state = 'dead';

        // Detect transitions
        if (state !== c.lastState) {
            c.events.push({
                type: 'state_change',
                from: c.lastState,
                to: state,
                level: level,
                deaths: deaths,
                quakes: quakes,
                time: Date.now()
            });
            c.lastState = state;
        }

        if (level !== c.lastLevel && level > 0) {
            c.events.push({
                type: 'level_change',
                level: level,
                deaths: deaths,
                time: Date.now()
            });
            c.lastLevel = level;
        }

        // Store current state for polling
        c.current = { state, level, deaths, dead, quakes, isStarted };
    } catch(e) {}
}, 200);
"""


async def capture_gameplay(duration=DURATION, start_level=START_LEVEL, headless=True):
    from playwright.async_api import async_playwright

    print(f"Starting capture rig — {duration}s, L{start_level}+, headless={headless}")
    print(f"Screenshots → {SHOTS_DIR}")
    print(f"Videos → {VIDEO_DIR}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_L{start_level}"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(
            viewport=VIEWPORT,
            record_video_dir=str(VIDEO_DIR),
            record_video_size=VIEWPORT,
        )
        page = await context.new_page()

        print(f"Loading {GAME_URL}")
        await page.goto(GAME_URL, wait_until="networkidle")
        await asyncio.sleep(1)

        # Wait for game canvas to be visible (landscape check)
        await page.wait_for_selector("#c", state="visible", timeout=10000)
        await asyncio.sleep(1)

        # Screenshot the start screen
        start_path = SHOTS_DIR / f"{timestamp}_000_start_screen.png"
        await page.screenshot(path=str(start_path))
        print(f"  [shot] start screen")

        # Inject hooks
        await page.evaluate(HOOK_JS)
        await asyncio.sleep(0.5)

        # Set starting level via JS injection
        if start_level > 1:
            await page.evaluate(f"""
                player._highestLevel = 24;
                startSelectedLevel = {start_level};
            """)
            await asyncio.sleep(0.3)

        # Press Space to start the game
        await page.keyboard.press("Space")
        await asyncio.sleep(2)

        # If bot didn't auto-start, press again
        is_started = await page.evaluate("typeof started !== 'undefined' && started")
        if not is_started:
            await page.keyboard.press("Space")
            await asyncio.sleep(1)

        shot_count = 1
        last_periodic = time.time()
        last_level = 0
        start_time = time.time()
        seen_events = 0

        print(f"Bot is playing! Capturing for {duration}s...")

        while time.time() - start_time < duration:
            await asyncio.sleep(0.5)

            # Get game state
            try:
                state = await page.evaluate("window._capture.current")
                events = await page.evaluate("window._capture.events")
            except Exception:
                continue

            if not state:
                continue

            # Process new events
            new_events = events[seen_events:]
            seen_events = len(events)

            for ev in new_events:
                ev_type = ev.get("type", "")
                ev_to = ev.get("to", "")
                level = ev.get("level", 0)

                # Screenshot on interesting transitions
                should_shot = False
                label = ""

                if ev_type == "state_change":
                    if ev_to == "dead":
                        # Screenshot every death for maximum coverage
                        should_shot = True
                        label = f"death_L{level}_d{state.get('deaths', 0)}"
                    elif ev_to == "level_complete":
                        should_shot = True
                        label = f"level_complete_L{level}"
                    elif ev_to == "won":
                        should_shot = True
                        label = f"win_screen"
                    elif ev_to == "playing" and ev.get("from") == "dead":
                        # Resurrection / respawn — sometimes interesting
                        if state.get("quakes", 0) > 0:
                            should_shot = True
                            label = f"resurrect_L{level}"

                elif ev_type == "level_change":
                    if level != last_level:
                        should_shot = True
                        label = f"level_start_L{level}"
                        last_level = level

                if should_shot:
                    shot_count += 1
                    path = SHOTS_DIR / f"{timestamp}_{shot_count:03d}_{label}.png"
                    try:
                        await page.screenshot(path=str(path))
                        print(f"  [shot] {label}")
                    except Exception as e:
                        print(f"  [shot failed] {label}: {e}")

            # Periodic screenshot every 4 seconds for variety
            if time.time() - last_periodic > 4:
                last_periodic = time.time()
                shot_count += 1
                level = state.get("level", "?")
                deaths = state.get("deaths", "?")
                label = f"periodic_L{level}_d{deaths}"
                path = SHOTS_DIR / f"{timestamp}_{shot_count:03d}_{label}.png"
                try:
                    await page.screenshot(path=str(path))
                    elapsed = int(time.time() - start_time)
                    print(f"  [shot] {label} ({elapsed}s / {duration}s)")
                except Exception:
                    pass

        print(f"\nCapture complete! {shot_count} screenshots taken")

        # Close — video saves on context close
        await context.close()
        await browser.close()

    # Rename the video file
    vids = list(VIDEO_DIR.glob("*.webm"))
    if vids:
        latest = max(vids, key=lambda f: f.stat().st_mtime)
        new_name = VIDEO_DIR / f"sediment_{timestamp}.webm"
        latest.rename(new_name)
        print(f"Video saved: {new_name}")

    print(f"\nTotal screenshots: {shot_count}")
    print(f"Run with --rank to have Haiku pick the best ones")
    return shot_count


def rank_screenshots():
    """Use Haiku to rank screenshots for Play Store quality."""
    import base64

    shots = sorted(SHOTS_DIR.glob("*.png"))
    if not shots:
        print("No screenshots found. Run capture first.")
        return

    print(f"Found {len(shots)} screenshots. Asking Haiku to rank them...")

    # Get OAuth token for Haiku
    creds_path = Path.home() / ".claude" / ".credentials.json"
    creds = json.load(open(creds_path))
    token = creds["claudeAiOauth"]["accessToken"]

    # Process in batches of 8 (vision + text per image)
    results = []
    batch_size = 5

    for i in range(0, len(shots), batch_size):
        batch = shots[i:i + batch_size]
        print(f"  Rating batch {i // batch_size + 1}/{(len(shots) + batch_size - 1) // batch_size}...")

        content = []
        for shot in batch:
            # Resize to save tokens
            try:
                from PIL import Image
                img = Image.open(shot)
                img.thumbnail((960, 540))
                import io
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
            except ImportError:
                b64 = base64.b64encode(shot.read_bytes()).decode()

            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": b64}
            })
            content.append({
                "type": "text",
                "text": f"Image: {shot.name}"
            })

        content.append({
            "type": "text",
            "text": """Rate each screenshot above for Google Play Store listing quality on a scale of 1-10.

Consider:
- Visual excitement and action (particles, effects, chaos = good)
- Shows the core mechanic clearly (death-to-terrain, row clearing)
- Good composition (player visible, interesting terrain)
- NOT too cluttered or confusing
- Would make someone want to download the game

For each image, respond with EXACTLY this format (one per line):
FILENAME | SCORE | REASON (brief)

Example:
screenshot_001_start.png | 7 | Clean title screen with animated character
screenshot_005_death.png | 9 | Great action shot with particles and corpse placement"""
        })

        body = json.dumps({
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 1500,
            "messages": [{"role": "user", "content": content}]
        }).encode()

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "oauth-2025-04-20"
            }
        )

        try:
            resp = json.loads(urllib.request.urlopen(req, timeout=60).read())
            text = resp["candidates"][0]["content"]["parts"][0]["text"] if "candidates" in resp else resp["content"][0]["text"]

            for line in text.strip().split("\n"):
                line = line.strip()
                if "|" in line:
                    parts = line.split("|")
                    if len(parts) >= 2:
                        fname = parts[0].strip()
                        try:
                            score = int(parts[1].strip())
                        except ValueError:
                            continue
                        reason = parts[2].strip() if len(parts) > 2 else ""
                        results.append((score, fname, reason))
        except Exception as e:
            print(f"  Haiku error: {e}")
            continue

    if not results:
        print("No ratings received. Check API token.")
        return

    # Sort by score descending
    results.sort(key=lambda x: -x[0])

    print(f"\n{'='*60}")
    print(f"HAIKU'S RANKINGS ({len(results)} screenshots rated)")
    print(f"{'='*60}")

    for score, fname, reason in results:
        marker = " ***" if score >= 8 else ""
        print(f"  {score}/10  {fname}  — {reason}{marker}")

    # Copy top 8 to best/ directory
    top_shots = results[:8]
    for d in BEST_DIR.iterdir():
        d.unlink()  # clear old picks

    print(f"\nCopying top {len(top_shots)} to {BEST_DIR}/")
    for i, (score, fname, reason) in enumerate(top_shots):
        # Find the actual file
        matches = list(SHOTS_DIR.glob(f"*{fname}*")) if not (SHOTS_DIR / fname).exists() else [SHOTS_DIR / fname]
        if not matches:
            # Try fuzzy match
            for shot in SHOTS_DIR.glob("*.png"):
                if fname.replace(".png", "") in shot.name:
                    matches = [shot]
                    break
        if matches:
            src = matches[0]
            dst = BEST_DIR / f"store_{i + 1:02d}_score{score}_{src.stem}.png"
            shutil.copy2(src, dst)
            print(f"  {dst.name}")

    print(f"\nDone! Best screenshots in {BEST_DIR}/")
    print(f"Videos in {VIDEO_DIR}/")


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--rank" in args:
        rank_screenshots()
    else:
        duration = DURATION
        start_level = START_LEVEL
        headless = "--headless" in args or "--fast" in args

        if "--duration" in args:
            idx = args.index("--duration")
            if idx + 1 < len(args):
                duration = int(args[idx + 1])

        if "--level" in args:
            idx = args.index("--level")
            if idx + 1 < len(args):
                start_level = int(args[idx + 1])

        asyncio.run(capture_gameplay(duration, start_level, headless))
        print("\nNow run: python3 capture_gameplay.py --rank")
