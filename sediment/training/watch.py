"""Live viewer for Sediment NN training — watch the current best agent play."""

import json
import os
import sys
import time

import numpy as np
import pygame

from sediment_sim import SedimentSim, SmallNN, get_cells, TILE, LEVEL_H_TILES, SEGMENT_W, get_segments_for_level

SCALE = 4  # pixels per tile-pixel
FPS = 60
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_weights.json')
RELOAD_INTERVAL = 5.0  # check for new weights every 5s

# Colors
BG = (18, 18, 24)
FLOOR = (60, 60, 70)
SPIKE = (200, 50, 50)
CORPSE_SOLID = (100, 130, 160)
CORPSE_SPRING = (80, 200, 120)
CORPSE_BOOSTER = (200, 160, 60)
CORPSE_SPIKE = (180, 60, 60)
PLAYER = (220, 220, 240)
PLAYER_SPRING = (100, 240, 140)
PLAYER_BOOSTER = (240, 200, 80)
PLAYER_SPIKE_MAT = (240, 80, 80)
GOAL = (80, 220, 80)
SAW = (255, 80, 80)
PLAT = (90, 90, 110)
TEXT_COL = (200, 200, 210)
INVINCIBLE = (255, 220, 100)


def load_weights():
    try:
        mtime = os.path.getmtime(WEIGHTS_PATH)
        with open(WEIGHTS_PATH) as f:
            data = json.load(f)
        # Reconstruct flat weight vector from layer matrices
        flat = []
        for key in ['w1', 'b1', 'w2', 'b2', 'w3', 'b3']:
            arr = np.array(data[key], dtype=np.float32)
            flat.extend(arr.flatten().tolist())
        return np.array(flat, dtype=np.float32), mtime
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return None, 0


def get_player_color(player):
    if player['crunch_timer'] > 0:
        return INVINCIBLE
    mat = player['material']
    if mat == 'spring':
        return PLAYER_SPRING
    elif mat == 'booster':
        return PLAYER_BOOSTER
    elif mat == 'spike':
        return PLAYER_SPIKE_MAT
    return PLAYER


def get_corpse_color(mat):
    if mat == 'spring':
        return CORPSE_SPRING
    elif mat == 'booster':
        return CORPSE_BOOSTER
    elif mat == 'spike':
        return CORPSE_SPIKE
    return CORPSE_SOLID


def main():
    level_num = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    pygame.init()

    # Calculate window size based on level
    num_segs = get_segments_for_level(level_num)
    level_w_tiles = num_segs * SEGMENT_W
    win_w = level_w_tiles * TILE * SCALE
    win_h = LEVEL_H_TILES * TILE * SCALE

    # Cap window width, use camera scrolling
    MAX_WIN_W = 1200
    actual_win_w = min(win_w, MAX_WIN_W)

    screen = pygame.display.set_mode((actual_win_w, win_h + 40))
    pygame.display.set_caption('Sediment NN Training Viewer')
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('monospace', 14)

    # Load weights and create sim
    weights, last_mtime = load_weights()
    if weights is None:
        print("No weights found, waiting...")
        while weights is None:
            time.sleep(1)
            weights, last_mtime = load_weights()

    nn = SmallNN(weights)
    sim = SedimentSim(level_num, corpses=True)
    last_reload = time.time()
    gen_text = "?"
    action_cooldown = 0.0

    # Read gen + level from training log
    def read_gen():
        try:
            with open('/tmp/sediment_train.log') as f:
                lines = f.readlines()
            for line in reversed(lines):
                if 'Gen' in line:
                    return line.strip()
            return "Training..."
        except:
            return "No log"

    def read_training_level():
        """Parse current training level from log."""
        try:
            with open('/tmp/sediment_train.log') as f:
                lines = f.readlines()
            for line in reversed(lines):
                if '| L' in line:
                    # "Gen   17 | L2 |..."
                    part = line.split('| L')[1]
                    return int(part.split()[0].split('|')[0])
            return level_num
        except:
            return level_num

    running = True
    speed = 1.0  # playback speed multiplier

    while running:
        dt = 1.0 / FPS * speed

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    speed = min(10.0, speed + 0.5)
                elif event.key == pygame.K_MINUS:
                    speed = max(0.5, speed - 0.5)
                elif event.key == pygame.K_r:
                    # Force reload + reset
                    weights, last_mtime = load_weights()
                    if weights is not None:
                        nn = SmallNN(weights)
                    sim = SedimentSim(level_num, corpses=True)

        # Reload weights periodically and track training level
        now = time.time()
        if now - last_reload > RELOAD_INTERVAL:
            last_reload = now
            new_weights, new_mtime = load_weights()
            if new_weights is not None and new_mtime > last_mtime:
                nn = SmallNN(new_weights)
                last_mtime = new_mtime
                # Check if training advanced to a new level
                train_level = read_training_level()
                if train_level != level_num:
                    level_num = train_level
                    # Recalc window for new level width
                    num_segs = get_segments_for_level(level_num)
                    win_w = num_segs * SEGMENT_W * TILE * SCALE
                    actual_win_w = min(win_w, MAX_WIN_W)
                    screen = pygame.display.set_mode((actual_win_w, win_h + 40))
                # Reset sim to show fresh run with new weights
                sim = SedimentSim(level_num, corpses=True)
            gen_text = read_gen()

        # Step sim with NN actions (rate-limited like browser)
        if not sim.player['dead']:
            action_cooldown -= dt
            if action_cooldown <= 0:
                inputs = sim.get_nn_inputs()
                action = nn.forward(inputs)
                action_cooldown = 0.12
            else:
                action = 0
        else:
            action = 0

        sim.step(dt, action)

        # If level completed, restart on same level to keep watching
        if sim.levels_completed > 0:
            sim = SedimentSim(level_num, corpses=True)
            action_cooldown = 0.0

        # Camera follows player
        cam_x = max(0, sim.player['x'] * SCALE - actual_win_w // 3)
        cam_x = min(cam_x, win_w - actual_win_w)

        # Draw
        screen.fill(BG)

        # Tiles
        p = sim.player
        # Only draw visible columns
        start_col = max(0, int(cam_x / (TILE * SCALE)))
        end_col = min(level_w_tiles, start_col + actual_win_w // (TILE * SCALE) + 2)

        for y in range(LEVEL_H_TILES):
            for x in range(start_col, end_col):
                sx = x * TILE * SCALE - cam_x
                sy = y * TILE * SCALE

                tile = sim.level[y][x] if y < len(sim.level) and x < len(sim.level[y]) else 0
                corpse = sim.corpse_grid[y][x] if y < len(sim.corpse_grid) and x < len(sim.corpse_grid[y]) else None

                if tile == 1:
                    pygame.draw.rect(screen, FLOOR, (sx, sy, TILE * SCALE, TILE * SCALE))
                elif tile == 2:
                    pygame.draw.rect(screen, SPIKE, (sx, sy, TILE * SCALE, TILE * SCALE))
                    # Spike triangle
                    pygame.draw.polygon(screen, (255, 80, 80), [
                        (sx + TILE * SCALE // 2, sy + 2),
                        (sx + 2, sy + TILE * SCALE - 2),
                        (sx + TILE * SCALE - 2, sy + TILE * SCALE - 2),
                    ])
                elif corpse:
                    color = get_corpse_color(corpse['mat'])
                    pygame.draw.rect(screen, color, (sx + 1, sy + 1, TILE * SCALE - 2, TILE * SCALE - 2))

        # Goal marker
        goal_sx = sim.goal_x * SCALE - cam_x
        pygame.draw.rect(screen, GOAL, (goal_sx, 0, 4 * SCALE, LEVEL_H_TILES * TILE * SCALE), 2)

        # Buzzsaws
        for saw in sim.buzzsaws:
            sx = saw['x'] * SCALE - cam_x
            sy = saw['y'] * SCALE
            r = int(saw['r'] * SCALE)
            if -r < sx < actual_win_w + r:
                pygame.draw.circle(screen, SAW, (int(sx), int(sy)), r)
                pygame.draw.circle(screen, (255, 200, 200), (int(sx), int(sy)), r, 2)

        # Moving platforms
        for mp in sim.moving_platforms:
            sx = mp['x'] * SCALE - cam_x
            sy = mp['y'] * SCALE
            pw = mp['w'] * TILE * SCALE
            if -pw < sx < actual_win_w + pw:
                pygame.draw.rect(screen, PLAT, (int(sx), int(sy), pw, TILE * SCALE))

        # Player
        if not p['dead']:
            cells = get_cells(p['shape'], p['rotation'])
            pcol = get_player_color(p)
            for cx, cy in cells:
                sx = (p['x'] + cx * TILE) * SCALE - cam_x
                sy = (p['y'] + cy * TILE) * SCALE
                pygame.draw.rect(screen, pcol, (int(sx) + 1, int(sy) + 1, TILE * SCALE - 2, TILE * SCALE - 2))

        # HUD
        hud_y = LEVEL_H_TILES * TILE * SCALE + 4
        dist_pct = min(100, p['best_distance'] / max(1, sim.goal_x) * 100)
        hud = (
            f"L{sim.level_num} | Dist: {dist_pct:.0f}% | "
            f"Deaths: {p['death_count']} | Crunches: {p['crunch_count']} | "
            f"Speed: {speed:.1f}x | {gen_text[:80]}"
        )
        txt = font.render(hud, True, TEXT_COL)
        screen.blit(txt, (8, hud_y))

        # Progress bar at bottom
        bar_w = int(actual_win_w * dist_pct / 100)
        pygame.draw.rect(screen, (40, 120, 40), (0, hud_y + 20, bar_w, 8))
        pygame.draw.rect(screen, (60, 60, 70), (bar_w, hud_y + 20, actual_win_w - bar_w, 8))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == '__main__':
    main()
