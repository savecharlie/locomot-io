"""Sediment game simulator — headless Python port for NN training.
Ports the core physics, collision, level gen from sediment/index.html.
No rendering, audio, or visual effects."""

import math
import random
import numpy as np

# ── Constants ──
TILE = 12
LEVEL_H_TILES = 14
SEGMENT_W = 40
GRAVITY = 700
JUMP_FORCE = -280
RUN_SPEED = 100
WALL_SLIDE_SPEED = 40
COLUMN_CLEAR_THRESHOLD = 6

# ── Shape definitions (exact copy from JS) ──
SHAPES = {
    'I': [
        [[0,1],[1,1],[2,1],[3,1]],
        [[2,0],[2,1],[2,2],[2,3]],
        [[0,2],[1,2],[2,2],[3,2]],
        [[1,0],[1,1],[1,2],[1,3]]
    ],
    'O': [
        [[0,0],[1,0],[0,1],[1,1]],
        [[0,0],[1,0],[0,1],[1,1]],
        [[0,0],[1,0],[0,1],[1,1]],
        [[0,0],[1,0],[0,1],[1,1]]
    ],
    'T': [
        [[0,0],[1,0],[2,0],[1,1]],
        [[1,0],[1,1],[1,2],[0,1]],
        [[1,1],[0,2],[1,2],[2,2]],
        [[1,0],[1,1],[1,2],[2,1]]
    ],
    'S': [
        [[1,0],[2,0],[0,1],[1,1]],
        [[0,0],[0,1],[1,1],[1,2]],
        [[1,1],[2,1],[0,2],[1,2]],
        [[1,0],[1,1],[2,1],[2,2]]
    ],
    'Z': [
        [[0,0],[1,0],[1,1],[2,1]],
        [[2,0],[1,1],[2,1],[1,2]],
        [[0,1],[1,1],[1,2],[2,2]],
        [[1,0],[0,1],[1,1],[0,2]]
    ],
    'L': [
        [[0,0],[1,0],[2,0],[0,1]],
        [[0,0],[1,0],[1,1],[1,2]],
        [[2,1],[0,2],[1,2],[2,2]],
        [[1,0],[1,1],[1,2],[2,2]]
    ],
    'J': [
        [[0,0],[1,0],[2,0],[2,1]],
        [[1,0],[1,1],[0,2],[1,2]],
        [[0,1],[0,2],[1,2],[2,2]],
        [[1,0],[2,0],[1,1],[1,2]]
    ]
}

SHAPE_NAMES = list(SHAPES.keys())
MATERIAL_TYPES = ['solid', 'spring', 'booster']

# SRS wall kick data
KICK_DATA = [
    [[0,0],[-1,0],[-1,-1],[0,2],[-1,2]],
    [[0,0],[1,0],[1,1],[0,-2],[1,-2]],
    [[0,0],[1,0],[1,-1],[0,2],[1,2]],
    [[0,0],[-1,0],[-1,1],[0,-2],[-1,-2]]
]
KICK_DATA_I = [
    [[0,0],[-2,0],[1,0],[-2,1],[1,-2]],
    [[0,0],[-1,0],[2,0],[-1,-2],[2,1]],
    [[0,0],[2,0],[-1,0],[2,-1],[-1,2]],
    [[0,0],[1,0],[-2,0],[1,2],[-2,-1]]
]


def get_cells(shape_name, rotation):
    return SHAPES[shape_name][rotation]


def shape_bounds(shape_name, rotation):
    cells = get_cells(shape_name, rotation)
    min_c = min(c for c, r in cells)
    max_c = max(c for c, r in cells)
    min_r = min(r for c, r in cells)
    max_r = max(r for c, r in cells)
    return min_c, max_c, min_r, max_r, max_c - min_c + 1, max_r - min_r + 1


def pick_material(lvl):
    if lvl >= 4 and random.random() < 0.25:
        return 'spike'
    return random.choice(MATERIAL_TYPES)


def get_level_difficulty(lvl):
    return min(10, 1 + (lvl - 1) * 0.82)


def get_segments_for_level(lvl):
    return min(7, 3 + int((lvl - 1) * 0.4))


class SedimentSim:
    """Headless Sediment game simulation."""

    def __init__(self, level_num=1, corpses=True):
        self.level_num = level_num
        self.corpses = corpses
        self.level = []       # level[y][x] = 0|1|2
        self.corpse_grid = [] # corpse_grid[y][x] = None or {'mat':str}
        self.level_h = LEVEL_H_TILES
        self.level_w = 0
        self.buzzsaws = []
        self.moving_platforms = []
        self.segments = []
        self.pending_clears = []
        self.goal_x = 0
        self.time = 0.0

        # Shape bag (7-bag randomizer)
        self.shape_bag = []

        # Player
        self.player = {
            'x': 2 * TILE, 'y': 0,
            'vx': RUN_SPEED, 'vy': 0,
            'shape': 'T', 'rotation': 0,
            'material': 'solid',
            'on_ground': False, 'on_wall': 0,
            'dead': False, 'facing': 1,
            'boost_timer': 0.0,
            'death_count': 0, 'crunch_count': 0,
            'crunch_timer': 0.0,
            'distance': 0.0, 'best_distance': 0.0,
            'spawn_x': 2 * TILE,
            'jumps_left': 2, 'coyote_timer': 0.0,
            'death_timer': 0.0, 'stuck_timer': 0.0,
        }
        self.levels_completed = 0

        self._init_level(level_num)

    def _pick_shape(self):
        if not self.shape_bag:
            self.shape_bag = list(SHAPE_NAMES)
            random.shuffle(self.shape_bag)
        return self.shape_bag.pop()

    def _init_level(self, lvl):
        self.level = []
        self.corpse_grid = []
        self.buzzsaws = []
        self.moving_platforms = []
        self.segments = []
        self.pending_clears = []
        self.level_h = LEVEL_H_TILES
        self.level_num = lvl

        num_segs = get_segments_for_level(lvl)
        for i in range(num_segs):
            self._generate_segment(i, lvl)
        self.level_w = num_segs * SEGMENT_W
        self.goal_x = (num_segs * SEGMENT_W - 4) * TILE

        p = self.player
        p['x'] = 2 * TILE
        p['y'] = (self.level_h - 3) * TILE
        p['vx'] = RUN_SPEED
        p['vy'] = 0
        p['shape'] = self._pick_shape()
        p['rotation'] = 0
        p['material'] = pick_material(lvl)
        p['on_ground'] = False
        p['on_wall'] = 0
        p['dead'] = False
        p['boost_timer'] = 0.0
        p['crunch_timer'] = 0.0
        p['distance'] = 0.0
        p['spawn_x'] = 2 * TILE
        p['jumps_left'] = 2
        p['coyote_timer'] = 0.0
        p['death_timer'] = 0.0
        p['stuck_timer'] = 0.0

    def _generate_segment(self, seg_index, lvl):
        h = self.level_h
        start_x = seg_index * SEGMENT_W
        d_base = get_level_difficulty(lvl)
        d = min(10, d_base + seg_index * 0.3)

        # Ensure grid is large enough
        while len(self.level) < h:
            self.level.append([])
            self.corpse_grid.append([])

        for y in range(h):
            while len(self.level[y]) < start_x + SEGMENT_W:
                self.level[y].append(0)
                self.corpse_grid[y].append(None)
            for x in range(start_x, start_x + SEGMENT_W):
                self.level[y][x] = 1 if (y == 0 or y == h - 1) else 0
                self.corpse_grid[y][x] = None

        # Chasms
        num_chasms = 2 + int(d * 0.8)
        chasm_min = 10 if seg_index == 0 else 3
        for _ in range(num_chasms):
            cx = start_x + chasm_min + random.randint(0, max(0, SEGMENT_W - chasm_min - 5))
            cw = 2 + int(random.random() * (1 + d * 0.4))
            for dx in range(cw):
                xx = cx + dx
                if xx < start_x + SEGMENT_W:
                    if lvl >= 8 and random.random() < min(0.8, d * 0.1):
                        if h - 2 >= 0:
                            self.level[h - 2][xx] = 2
                    else:
                        self.level[h - 1][xx] = 0

        # Floor spikes (L2+)
        if lvl >= 2:
            num_spikes = 1 + int(d * 0.6)
            spike_min = 10 if seg_index == 0 else 3
            for _ in range(num_spikes):
                sx = start_x + spike_min + random.randint(0, max(0, SEGMENT_W - spike_min - 4))
                sw = 2 + random.randint(0, 2)
                for dx in range(sw):
                    xx = sx + dx
                    if xx < start_x + SEGMENT_W and self.level[h - 1][xx] == 1:
                        self.level[h - 2][xx] = 2

        # Ceiling spikes (L3+)
        if lvl >= 3:
            num_cs = max(1, int(d * 0.5))
            for _ in range(num_cs):
                sx = start_x + 2 + random.randint(0, max(0, SEGMENT_W - 5))
                if sx < start_x + SEGMENT_W:
                    self.level[1][sx] = 2
                    if d > 6 and random.random() < 0.4:
                        self.level[2][sx] = 2

        # Static platforms (L5+)
        if lvl >= 5:
            num_plats = 1 + int(d * 0.3)
            plat_min = 14 if seg_index == 0 else 4
            for _ in range(num_plats):
                px = start_x + plat_min + random.randint(0, max(0, SEGMENT_W - plat_min - 7))
                pw = 2 + random.randint(0, 2)
                py = 3 + random.randint(0, max(0, h - 7))
                for dx in range(pw):
                    xx = px + dx
                    if xx < start_x + SEGMENT_W and py < h:
                        self.level[py][xx] = 1

        # Buzzsaws (L6+)
        if lvl >= 6:
            num_saws = max(1, int((d - 4) * 0.6))
            for _ in range(num_saws):
                saw_x = (start_x + 8 + random.randint(0, max(0, SEGMENT_W - 17))) * TILE
                saw_y = (2 + random.randint(0, max(0, h - 6))) * TILE
                vertical = random.random() < 0.5
                rng = (2 + random.randint(0, 2)) * TILE
                self.buzzsaws.append({
                    'x': saw_x, 'y': saw_y,
                    'base_x': saw_x, 'base_y': saw_y,
                    'r': 6, 'vertical': vertical, 'range': rng,
                    'speed': 30 + random.random() * 40,
                    'phase': random.random() * math.pi * 2,
                })

        # Moving platforms H (L7+)
        if lvl >= 7:
            num_mov = 1 + int((d - 5) * 0.4)
            mov_min = 16 if seg_index == 0 else 5
            for _ in range(num_mov):
                mx = (start_x + mov_min + random.randint(0, max(0, SEGMENT_W - mov_min - 9))) * TILE
                my = (3 + random.randint(0, max(0, h - 7))) * TILE
                vert = lvl >= 10 and random.random() < 0.4
                self.moving_platforms.append({
                    'x': mx, 'y': my, 'w': 3,
                    'base_x': mx, 'base_y': my,
                    'vertical': vert,
                    'range': (2 + random.randint(0, 1)) * TILE,
                    'speed': 20 + random.random() * 30,
                    'phase': random.random() * math.pi * 2,
                })

        # Spike walls (L9+)
        if lvl >= 9:
            num_walls = max(1, int((d - 6) * 0.5))
            for _ in range(num_walls):
                wx = start_x + 6 + random.randint(0, max(0, SEGMENT_W - 13))
                wall_h = 2 + random.randint(0, 2)
                start_y = h - 2 - wall_h
                for dy in range(wall_h):
                    yy = start_y + dy
                    if 0 <= yy < h and wx < start_x + SEGMENT_W:
                        self.level[yy][wx] = 2

        # Safe spawn
        if seg_index == 0:
            for y in range(1, h - 1):
                for x in range(10):
                    if self.level[y][x] == 2:
                        self.level[y][x] = 0
            for x in range(10):
                self.level[h - 1][x] = 1

        self.segments.append(seg_index)

    def _tile_is_solid(self, tx, ty):
        if ty < 0:
            return True
        if ty >= self.level_h:
            return False
        if tx < 0 or tx >= self.level_w:
            return tx < 0  # left wall solid, right void
        if self.level[ty][tx] == 1:
            return True
        if self.corpse_grid[ty][tx] is not None:
            return True
        for p in self.moving_platforms:
            px1 = int(p['x'] / TILE)
            py = int(p['y'] / TILE)
            if ty == py and px1 <= tx < px1 + p['w']:
                return True
        return False

    def _tile_is_spike(self, tx, ty):
        if ty < 0 or ty >= self.level_h or tx < 0 or tx >= self.level_w:
            return False
        if self.level[ty][tx] == 2:
            return True
        c = self.corpse_grid[ty][tx]
        if c and c['mat'] == 'spike':
            return True
        return False

    def _tet_solid(self, px, py, shape, rot):
        cells = get_cells(shape, rot)
        for cx, cy in cells:
            wx = px + cx * TILE
            wy = py + cy * TILE
            l = int(wx // TILE)
            r = int((wx + TILE - 1) // TILE)
            t = int(wy // TILE)
            b = int((wy + TILE - 1) // TILE)
            for ty in range(t, b + 1):
                for tx in range(l, r + 1):
                    if self._tile_is_solid(tx, ty):
                        return True
        return False

    def _tet_spike(self, px, py, shape, rot):
        cells = get_cells(shape, rot)
        for cx, cy in cells:
            wx = px + cx * TILE + 1
            wy = py + cy * TILE + 1
            l = int(wx // TILE)
            r = int((wx + TILE - 3) // TILE)
            t = int(wy // TILE)
            b = int((wy + TILE - 3) // TILE)
            for ty in range(t, b + 1):
                for tx in range(l, r + 1):
                    if self._tile_is_spike(tx, ty):
                        return True
        return False

    def _corpse_mat_at(self, tx, ty):
        tx, ty = int(tx), int(ty)
        if 0 <= ty < self.level_h and 0 <= tx < self.level_w:
            c = self.corpse_grid[ty][tx]
            if c:
                return c['mat']
        return None

    def _check_adjacent_corpse(self, px, py, shape, rot, direction, mat):
        cells = get_cells(shape, rot)
        for cx, cy in cells:
            if direction == 'below':
                check_tx = int((px + cx * TILE + TILE / 2) / TILE)
                check_ty = int((py + (cy + 1) * TILE) / TILE)
            elif direction == 'above':
                check_tx = int((px + cx * TILE + TILE / 2) / TILE)
                check_ty = int((py + cy * TILE - 1) / TILE)
            elif direction == 'right':
                check_tx = int((px + (cx + 1) * TILE) / TILE)
                check_ty = int((py + cy * TILE + TILE / 2) / TILE)
            elif direction == 'left':
                check_tx = int((px + cx * TILE - 1) / TILE)
                check_ty = int((py + cy * TILE + TILE / 2) / TILE)
            else:
                continue
            if self._corpse_mat_at(check_tx, check_ty) == mat:
                return (check_tx, check_ty)
        return None

    def _can_escape_right(self):
        p = self.player
        shape, rot = p['shape'], p['rotation']
        max_up = 8 * TILE

        for dy in range(1, max_up + 1):
            test_y = p['y'] - dy
            if self._tet_solid(p['x'], test_y, shape, rot):
                break
            if not self._tet_solid(p['x'] + 1, test_y, shape, rot):
                return True

        for d in [1, -1]:
            new_rot = (rot + d + 4) % 4
            kicks = KICK_DATA_I if shape == 'I' else KICK_DATA
            kick_idx = rot if d == 1 else new_rot
            for kx, ky in kicks[kick_idx]:
                tx = p['x'] + (kx if d == 1 else -kx) * TILE
                ty = p['y'] + (-ky if d == 1 else ky) * TILE
                if not self._tet_solid(tx, ty, shape, new_rot):
                    if not self._tet_solid(tx + 1, ty, shape, new_rot):
                        return True
                    for dy2 in range(1, max_up + 1):
                        jy = ty - dy2
                        if self._tet_solid(tx, jy, shape, new_rot):
                            break
                        if not self._tet_solid(tx + 1, jy, shape, new_rot):
                            return True
        return False

    def _try_rotate(self, direction):
        p = self.player
        old_rot = p['rotation']
        new_rot = (old_rot + direction + 4) % 4
        kicks = KICK_DATA_I if p['shape'] == 'I' else KICK_DATA
        kick_idx = old_rot if direction == 1 else new_rot

        for kx, ky in kicks[kick_idx]:
            tx = p['x'] + (kx if direction == 1 else -kx) * TILE
            ty = p['y'] + (-ky if direction == 1 else ky) * TILE
            if not self._tet_solid(tx, ty, p['shape'], new_rot):
                p['x'] = tx
                p['y'] = ty
                p['rotation'] = new_rot
                return True
        return False

    def _die(self):
        p = self.player
        if p['dead']:
            return
        p['dead'] = True
        p['death_timer'] = 0.5
        p['death_count'] += 1

        # Place corpse blocks (skip in no-corpse training mode)
        if self.corpses:
            cells = get_cells(p['shape'], p['rotation'])
            for cx, cy in cells:
                tx = round((p['x'] + cx * TILE) / TILE)
                ty = round((p['y'] + cy * TILE) / TILE)
                if 0 <= ty < self.level_h and 0 <= tx < self.level_w:
                    if p['material'] == 'spike':
                        self.level[ty][tx] = 2
                    elif self.corpse_grid[ty][tx] is None and self.level[ty][tx] == 0:
                        # Neutralize terrain spikes underneath
                        if self.level[ty][tx] == 2:
                            self.level[ty][tx] = 0
                        self.corpse_grid[ty][tx] = {'mat': p['material']}

            # Apply corpse gravity
            self._apply_corpse_gravity()

            # Check for clears
            self._check_clears()

    def _apply_corpse_gravity(self):
        changed = True
        while changed:
            changed = False
            for y in range(self.level_h - 2, 0, -1):
                for x in range(self.level_w):
                    c = self.corpse_grid[y][x]
                    if c and not self._tile_is_solid_no_corpse(x, y + 1) and self.corpse_grid[y + 1][x] is None:
                        self.corpse_grid[y + 1][x] = c
                        self.corpse_grid[y][x] = None
                        changed = True

    def _tile_is_solid_no_corpse(self, tx, ty):
        if ty < 0:
            return True
        if ty >= self.level_h or tx < 0 or tx >= self.level_w:
            return False
        return self.level[ty][tx] == 1

    def _check_clears(self):
        max_x = len(self.segments) * SEGMENT_W
        cleared = False

        # Vertical column clearing
        for x in range(max_x):
            run = 0
            found = False
            for y in range(self.level_h):
                c = self.corpse_grid[y][x] if y < len(self.corpse_grid) and x < len(self.corpse_grid[y]) else None
                if c and c['mat'] != 'spike':
                    run += 1
                else:
                    if run >= COLUMN_CLEAR_THRESHOLD:
                        found = True
                        break
                    run = 0
            if run >= COLUMN_CLEAR_THRESHOLD:
                found = True
            if found:
                for y in range(self.level_h):
                    if y < len(self.corpse_grid) and x < len(self.corpse_grid[y]):
                        self.corpse_grid[y][x] = None
                self.player['crunch_count'] += 1
                self.player['crunch_timer'] += 3.0
                cleared = True

        # Horizontal row clearing
        for y in range(1, self.level_h - 1):
            run = 0
            run_start = 0
            for x in range(max_x):
                c = self.corpse_grid[y][x] if x < len(self.corpse_grid[y]) else None
                if c and c['mat'] != 'spike':
                    if run == 0:
                        run_start = x
                    run += 1
                else:
                    if run >= COLUMN_CLEAR_THRESHOLD:
                        for xx in range(run_start, run_start + run):
                            if xx < len(self.corpse_grid[y]):
                                self.corpse_grid[y][xx] = None
                        self.player['crunch_count'] += 1
                        self.player['crunch_timer'] += 3.0
                        cleared = True
                    run = 0
            if run >= COLUMN_CLEAR_THRESHOLD:
                for xx in range(run_start, run_start + run):
                    if xx < len(self.corpse_grid[y]):
                        self.corpse_grid[y][xx] = None
                self.player['crunch_count'] += 1
                self.player['crunch_timer'] += 3.0
                cleared = True

    def _respawn(self):
        p = self.player
        p['dead'] = False
        p['x'] = p['spawn_x']
        p['y'] = (self.level_h - 3) * TILE
        p['vx'] = RUN_SPEED
        p['vy'] = 0
        p['shape'] = self._pick_shape()
        p['rotation'] = 0
        p['material'] = pick_material(self.level_num)
        p['on_ground'] = False
        p['on_wall'] = 0
        p['jumps_left'] = 2
        p['coyote_timer'] = 0
        p['stuck_timer'] = 0
        p['boost_timer'] = 0

        # Find valid spawn (not in solid)
        if self._tet_solid(p['x'], p['y'], p['shape'], p['rotation']):
            p['y'] = (self.level_h - 4) * TILE

        # Update spawn point to best distance
        best_tile_x = int(p['best_distance'] / TILE)
        if best_tile_x > p['spawn_x'] // TILE + 5:
            p['spawn_x'] = max(p['spawn_x'], (best_tile_x - 5) * TILE)

    def step(self, dt, action=0):
        """Advance one frame. action: 0=nothing, 1=jump, 2=rotate_cw, 3=rotate_ccw, 4=voluntary_death"""
        p = self.player
        self.time += dt

        # Update moving hazards
        for saw in self.buzzsaws:
            t = self.time * saw['speed'] / saw['range'] + saw['phase']
            if saw['vertical']:
                saw['y'] = saw['base_y'] + math.sin(t) * saw['range']
            else:
                saw['x'] = saw['base_x'] + math.sin(t) * saw['range']

        for mp in self.moving_platforms:
            t = self.time * mp['speed'] / mp['range'] + mp['phase']
            if mp['vertical']:
                mp['y'] = mp['base_y'] + math.sin(t) * mp['range']
            else:
                mp['x'] = mp['base_x'] + math.sin(t) * mp['range']

        # Handle death timer
        if p['dead']:
            p['death_timer'] -= dt
            if p['death_timer'] <= 0:
                self._respawn()
            return

        # Process pending clears (simplified — instant in sim)

        # Apply action
        jump_buffered = False
        if action == 1:
            jump_buffered = True
        elif action == 2:
            self._try_rotate(1)
        elif action == 3:
            self._try_rotate(-1)
        elif action == 4:
            self._die()
            return

        # Auto-run
        if p['boost_timer'] > 0:
            p['boost_timer'] -= dt
        else:
            p['vx'] = RUN_SPEED

        # Gravity
        p['vy'] += GRAVITY * dt
        if p['vy'] > 400:
            p['vy'] = 400

        # Wall detection (only when airborne, per-cell like game)
        p['on_wall'] = 0
        if not p['on_ground']:
            cells = get_cells(p['shape'], p['rotation'])
            for cx, cy in cells:
                check_tx = int((p['x'] + (cx + 1) * TILE) / TILE)
                check_ty = int((p['y'] + cy * TILE + TILE / 2) / TILE)
                if self._tile_is_solid(check_tx, check_ty):
                    p['on_wall'] = 1
                    break

        # Wall slide
        if p['on_wall'] and p['vy'] > WALL_SLIDE_SPEED:
            p['vy'] = WALL_SLIDE_SPEED

        # Coyote timer
        if p['on_ground']:
            p['coyote_timer'] = 0.06
        elif p['coyote_timer'] > 0:
            p['coyote_timer'] -= dt

        # Jump
        if jump_buffered:
            if p['on_wall']:
                p['vy'] = JUMP_FORCE * 0.9
                p['vx'] = -180 if p['on_wall'] == 1 else 180
                p['jumps_left'] = 1
                p['boost_timer'] = 0.15
                p['on_wall'] = 0
            elif p['on_ground'] or p['coyote_timer'] > 0:
                p['vy'] = JUMP_FORCE
                p['jumps_left'] = 1
                p['coyote_timer'] = 0
            elif p['jumps_left'] > 0:
                p['vy'] = JUMP_FORCE * 0.7
                p['jumps_left'] -= 1

        # X movement (sub-pixel stepping, matches game exactly)
        dx = p['vx'] * dt
        hit_wall_x = False
        if dx != 0:
            sign = 1 if dx > 0 else -1
            steps = math.ceil(abs(dx))
            for i in range(steps):
                step = min(1.0, abs(dx) - i) * sign
                if not self._tet_solid(p['x'] + step, p['y'], p['shape'], p['rotation']):
                    p['x'] += step
                else:
                    p['vx'] = 0
                    hit_wall_x = True
                    break

        # Booster on wall hit
        if hit_wall_x:
            adj = self._check_adjacent_corpse(p['x'], p['y'], p['shape'], p['rotation'], 'right', 'booster')
            if adj:
                p['vx'] = 350
                p['vy'] = -140
                p['boost_timer'] = 0.25

        # Y movement (sub-pixel stepping, matches game exactly)
        dy = p['vy'] * dt
        p['on_ground'] = False
        if dy != 0:
            sign = 1 if dy > 0 else -1
            steps = math.ceil(abs(dy))
            for i in range(steps):
                step = min(1.0, abs(dy) - i) * sign
                if not self._tet_solid(p['x'], p['y'] + step, p['shape'], p['rotation']):
                    p['y'] += step
                else:
                    if p['vy'] > 0:
                        p['on_ground'] = True
                        p['jumps_left'] = 2
                    p['vy'] = 0
                    break
        # Extra ground check (matches game)
        if self._tet_solid(p['x'], p['y'] + 1, p['shape'], p['rotation']):
            p['on_ground'] = True

        # Spring/booster on landing
        if p['on_ground']:
            spring = self._check_adjacent_corpse(p['x'], p['y'], p['shape'], p['rotation'], 'below', 'spring')
            if spring:
                p['vy'] = JUMP_FORCE * 1.5
                p['on_ground'] = False
            booster = self._check_adjacent_corpse(p['x'], p['y'], p['shape'], p['rotation'], 'below', 'booster')
            if booster:
                p['vx'] = p['facing'] * 350
                p['vy'] = JUMP_FORCE * 0.3
                p['on_ground'] = False
                p['boost_timer'] = 0.25

        # Spring from below (ceiling bounce)
        if p['vy'] == 0 and dy < 0:
            spring_above = self._check_adjacent_corpse(p['x'], p['y'], p['shape'], p['rotation'], 'above', 'spring')
            if spring_above:
                p['vy'] = abs(JUMP_FORCE) * 1.2

        # Void death
        if p['y'] > self.level_h * TILE + 20:
            if p['crunch_timer'] > 0:
                p['vy'] = JUMP_FORCE
                p['y'] = (self.level_h - 2) * TILE
            else:
                self._die()
                return

        # Stuck detection
        if hit_wall_x and p['vx'] >= 0:
            if not self._can_escape_right():
                p['stuck_timer'] += dt
                if p['stuck_timer'] >= 0.5:
                    self._die()
                    return
            else:
                p['stuck_timer'] = 0
        else:
            p['stuck_timer'] = 0

        # Hazard checks (skip if invincible)
        if p['crunch_timer'] > 0:
            p['crunch_timer'] -= dt
        else:
            if self._tet_spike(p['x'], p['y'], p['shape'], p['rotation']):
                self._die()
                return
            # Buzzsaw collision
            cells = get_cells(p['shape'], p['rotation'])
            for cx, cy in cells:
                cell_cx = p['x'] + cx * TILE + TILE / 2
                cell_cy = p['y'] + cy * TILE + TILE / 2
                for saw in self.buzzsaws:
                    dist = math.sqrt((cell_cx - saw['x'])**2 + (cell_cy - saw['y'])**2)
                    if dist < saw['r'] + TILE / 2:
                        self._die()
                        return

        # Track distance
        cells = get_cells(p['shape'], p['rotation'])
        max_cell_x = max(p['x'] + cx * TILE for cx, cy in cells)
        p['distance'] = max(p['distance'], max_cell_x)
        p['best_distance'] = max(p['best_distance'], p['distance'])

        # Level complete
        if max_cell_x >= self.goal_x:
            self.levels_completed += 1
            self._init_level(self.level_num + 1)

    # ── NN Input Extraction ──

    def get_nn_inputs(self):
        """Extract the 72-dimensional input vector for the NN."""
        p = self.player
        if p['dead']:
            return np.zeros(72, dtype=np.float32)

        inputs = []

        # A. Player state (9)
        inputs.append(p['vy'] / 400.0)
        inputs.append(p['jumps_left'] / 2.0)
        inputs.append(1.0 if p['on_ground'] else 0.0)
        inputs.append((p['on_wall'] + 1) / 2.0)  # -1→0, 0→0.5, 1→1
        inputs.append(p['y'] / (self.level_h * TILE))
        inputs.append(1.0 if p['boost_timer'] > 0 else 0.0)
        inputs.append(1.0 if p['crunch_timer'] > 0 else 0.0)
        inputs.append(min(1.0, p['stuck_timer'] / 0.5))
        inputs.append(min(1.0, p['distance'] / max(1, self.goal_x)))

        # B. Shape geometry 4x4 (16)
        cells = get_cells(p['shape'], p['rotation'])
        min_c = min(c for c, r in cells)
        min_r = min(r for c, r in cells)
        grid_4x4 = [0.0] * 16
        for c, r in cells:
            gc = c - min_c
            gr = r - min_r
            if 0 <= gc < 4 and 0 <= gr < 4:
                grid_4x4[gr * 4 + gc] = 1.0
        inputs.extend(grid_4x4)

        # C. Material (3)
        inputs.append(1.0 if p['material'] == 'spring' else 0.0)
        inputs.append(1.0 if p['material'] == 'booster' else 0.0)
        inputs.append(1.0 if p['material'] == 'spike' else 0.0)

        # D. Terrain lookahead: 10 columns × 4 features (40)
        # Log-spaced: dense near, sparse far — see as far as the player can
        _col_offsets = [1, 2, 3, 5, 7, 10, 15, 20, 28, 38]
        player_col = int(p['x'] / TILE)
        for col_offset in _col_offsets:
            col = player_col + col_offset
            ground_h = 0.0
            ceiling_h = 0.0
            has_spike = 0.0
            has_gap = 1.0  # assume gap until we find ground

            # Scan column top to bottom
            for y in range(self.level_h):
                solid = False
                spike = False
                if 0 <= col < self.level_w:
                    if self.level[y][col] == 1:
                        solid = True
                    elif self.level[y][col] == 2:
                        spike = True
                        has_spike = 1.0
                    c = self.corpse_grid[y][col] if col < len(self.corpse_grid[y]) else None
                    if c:
                        solid = True
                        if c['mat'] == 'spike':
                            has_spike = 1.0

                if solid and ceiling_h == 0.0 and y > 0:
                    ceiling_h = y / self.level_h

            # Scan bottom to top for ground
            for y in range(self.level_h - 1, -1, -1):
                if 0 <= col < self.level_w:
                    if self.level[y][col] == 1 or self.corpse_grid[y][col] is not None:
                        ground_h = (self.level_h - y) / self.level_h
                        has_gap = 0.0
                        break

            inputs.extend([ground_h, ceiling_h, has_spike, has_gap])

        # E. Buzzsaw proximity (4) — 2 nearest
        player_cx = p['x'] + TILE * 1.5
        player_cy = p['y'] + TILE * 1.5
        saw_dists = []
        for saw in self.buzzsaws:
            dx_s = (saw['x'] - player_cx) / (10 * TILE)
            dy_s = (saw['y'] - player_cy) / (self.level_h * TILE)
            dist = dx_s**2 + dy_s**2
            saw_dists.append((dist, dx_s, dy_s))
        saw_dists.sort()

        for i in range(2):
            if i < len(saw_dists):
                inputs.append(max(-1, min(1, saw_dists[i][1])))
                inputs.append(max(-1, min(1, saw_dists[i][2])))
            else:
                inputs.extend([0.0, 0.0])

        return np.array(inputs, dtype=np.float32)


def evaluate_agent(weights, level_num=1, max_time=60.0, dt=1/60, corpses=False):
    """Run a single agent evaluation. Returns fitness."""
    sim = SedimentSim(level_num, corpses=corpses)
    nn = SmallNN(weights)

    while sim.time < max_time:
        if sim.player['dead']:
            sim.step(dt, action=0)
            continue

        inputs = sim.get_nn_inputs()
        action = nn.forward(inputs)
        sim.step(dt, action)

    p = sim.player
    fitness = p['best_distance'] + 500 * sim.levels_completed + 50 * p['crunch_count']
    return fitness, sim.levels_completed > 0


class SmallNN:
    """Tiny feedforward NN: 72 → 48 → 32 → 5"""

    LAYER_SIZES = [72, 48, 32, 5]

    def __init__(self, weights=None):
        if weights is None:
            weights = self.random_weights()
        self.w1, self.b1, self.w2, self.b2, self.w3, self.b3 = self._unpack(weights)

    @staticmethod
    def num_params():
        sizes = SmallNN.LAYER_SIZES
        total = 0
        for i in range(len(sizes) - 1):
            total += sizes[i] * sizes[i+1] + sizes[i+1]
        return total

    @staticmethod
    def random_weights():
        return np.random.randn(SmallNN.num_params()).astype(np.float32) * 0.5

    def _unpack(self, flat):
        sizes = self.LAYER_SIZES
        idx = 0
        params = []
        for i in range(len(sizes) - 1):
            n_in, n_out = sizes[i], sizes[i+1]
            w = flat[idx:idx + n_in * n_out].reshape(n_out, n_in)
            idx += n_in * n_out
            b = flat[idx:idx + n_out]
            idx += n_out
            params.extend([w, b])
        return params

    def forward(self, x):
        # Layer 1: ReLU
        h = np.maximum(0, self.w1 @ x + self.b1)
        # Layer 2: ReLU
        h = np.maximum(0, self.w2 @ h + self.b2)
        # Layer 3: softmax → argmax
        logits = self.w3 @ h + self.b3
        return int(np.argmax(logits))

    def to_json(self):
        """Export weights as JSON-serializable list for browser."""
        import json
        return json.dumps({
            'w1': self.w1.tolist(), 'b1': self.b1.tolist(),
            'w2': self.w2.tolist(), 'b2': self.b2.tolist(),
            'w3': self.w3.tolist(), 'b3': self.b3.tolist(),
        })


if __name__ == '__main__':
    # Quick test
    sim = SedimentSim(level_num=1)
    print(f"Level 1: {len(sim.segments)} segments, goal at {sim.goal_x}px")
    print(f"NN inputs shape: {sim.get_nn_inputs().shape}")
    print(f"NN params: {SmallNN.num_params()}")

    # Test one evaluation
    weights = SmallNN.random_weights()
    fitness = evaluate_agent(weights, level_num=1, max_time=10.0)
    print(f"Random agent fitness (10s): {fitness:.0f}")
