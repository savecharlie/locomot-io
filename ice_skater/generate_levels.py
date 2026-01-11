#!/usr/bin/env python3
"""
Ice Skater Level Generator
Generates puzzles with: sliding, blocks, ramps, corners, holes, keys, sticky tiles
"""

import json
import random
import copy
from collections import deque
from typing import List, Tuple, Optional, Set, Dict

# Tile types
WALL = '#'
ICE = '.'
START = 'S'
GOAL = 'P'
BLOCK = 'B'
HOLE = 'H'
KEY = 'K'
LOCKED = 'L'
STICKY = 'T'
RAMP_N = '^'  # High end north
RAMP_S = 'v'  # High end south
RAMP_E = '>'  # High end east
RAMP_W = '<'  # High end west
CORNER_UR = '7'  # Up->Right, Left->Down
CORNER_UL = '8'  # Up->Left, Right->Down
CORNER_DR = '9'  # Down->Right, Left->Up
CORNER_DL = '0'  # Down->Left, Right->Up

DIRECTIONS = {
    'up': (0, -1),
    'down': (0, 1),
    'left': (-1, 0),
    'right': (1, 0)
}

RAMPS = {RAMP_N, RAMP_S, RAMP_E, RAMP_W}
CORNERS = {CORNER_UR, CORNER_UL, CORNER_DR, CORNER_DL}


def get_corner_redirect(tile: str, dx: int, dy: int) -> Optional[Tuple[int, int]]:
    """Get new direction after hitting a corner tile."""
    if tile == CORNER_UR:  # Up->Right, Left->Down
        if dy == -1: return (1, 0)   # up -> right
        if dx == -1: return (0, 1)   # left -> down
    elif tile == CORNER_UL:  # Up->Left, Right->Down
        if dy == -1: return (-1, 0)  # up -> left
        if dx == 1: return (0, 1)    # right -> down
    elif tile == CORNER_DR:  # Down->Right, Left->Up
        if dy == 1: return (1, 0)    # down -> right
        if dx == -1: return (0, -1)  # left -> up
    elif tile == CORNER_DL:  # Down->Left, Right->Up
        if dy == 1: return (-1, 0)   # down -> left
        if dx == 1: return (0, -1)   # right -> up
    return None


class GameState:
    def __init__(self, grid: List[List[str]], player_x: int, player_y: int,
                 blocks: Set[Tuple[int, int]], has_key: bool = False,
                 unlocked: Set[Tuple[int, int]] = None,
                 filled_holes: Set[Tuple[int, int]] = None,
                 ramps: Dict[Tuple[int, int], str] = None):
        self.grid = grid
        self.width = len(grid[0])
        self.height = len(grid)
        self.player_x = player_x
        self.player_y = player_y
        self.blocks = blocks
        self.has_key = has_key
        self.unlocked = unlocked or set()
        self.filled_holes = filled_holes or set()
        self.ramps = ramps or {}
        self.dead = False
        self.won = False

    def copy(self):
        return GameState(
            [row[:] for row in self.grid],
            self.player_x, self.player_y,
            set(self.blocks), self.has_key,
            set(self.unlocked), set(self.filled_holes),
            dict(self.ramps)
        )

    def get_tile(self, x: int, y: int) -> str:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        return WALL

    def to_key(self) -> tuple:
        """Hashable state for BFS."""
        return (
            self.player_x, self.player_y,
            frozenset(self.blocks),
            self.has_key,
            frozenset(self.unlocked),
            frozenset(self.filled_holes),
            frozenset(self.ramps.items())
        )


def simulate_move(state: GameState, direction: str) -> Optional[GameState]:
    """Simulate a move and return new state, or None if invalid."""
    dx, dy = DIRECTIONS[direction]
    new_state = state.copy()

    x, y = new_state.player_x, new_state.player_y

    # Slide until hitting something
    moves = 0
    max_moves = 50  # Safety limit

    while moves < max_moves:
        nx, ny = x + dx, y + dy
        tile = new_state.get_tile(nx, ny)

        # Wall or locked wall stops movement
        if tile == WALL:
            break
        if tile == LOCKED and (nx, ny) not in new_state.unlocked:
            # Try to unlock if we have key
            if new_state.has_key:
                new_state.unlocked.add((nx, ny))
                new_state.has_key = False
            break

        # Block - try to push
        if (nx, ny) in new_state.blocks:
            pushed = push_block(new_state, nx, ny, dx, dy)
            if not pushed:
                break
            # Player ends up on block's original position
            x, y = nx, ny
            break

        # Ramp - check if pushable
        if (nx, ny) in new_state.ramps:
            pushed = push_ramp(new_state, nx, ny, dx, dy)
            if not pushed:
                break
            # Player ends up on ramp's original position
            x, y = nx, ny
            break

        # Move to new position
        x, y = nx, ny

        # Check what we landed on
        if tile == HOLE and (x, y) not in new_state.filled_holes:
            new_state.dead = True
            break

        if tile == KEY:
            new_state.has_key = True
            new_state.grid[y][x] = ICE

        if tile == GOAL:
            new_state.won = True
            break

        if tile == STICKY:
            break  # Stop on sticky

        # Corner redirect
        if tile in CORNERS:
            redirect = get_corner_redirect(tile, dx, dy)
            if redirect:
                dx, dy = redirect
            else:
                break  # Invalid approach angle

        moves += 1

    if x == state.player_x and y == state.player_y:
        return None  # No movement

    new_state.player_x = x
    new_state.player_y = y
    return new_state


def push_block(state: GameState, bx: int, by: int, dx: int, dy: int) -> bool:
    """Push a block. Returns True if successful."""
    state.blocks.remove((bx, by))

    x, y = bx, by
    max_moves = 50
    moves = 0

    while moves < max_moves:
        nx, ny = x + dx, y + dy
        tile = state.get_tile(nx, ny)

        if tile == WALL:
            break
        if tile == LOCKED and (nx, ny) not in state.unlocked:
            break
        if (nx, ny) in state.blocks:
            break  # Can't push into another block
        if (nx, ny) in state.ramps:
            break  # Can't push into ramp

        x, y = nx, ny

        # Block falls into hole
        if tile == HOLE and (x, y) not in state.filled_holes:
            state.filled_holes.add((x, y))
            return True  # Block is gone

        # Corner redirect for block
        if tile in CORNERS:
            redirect = get_corner_redirect(tile, dx, dy)
            if redirect:
                dx, dy = redirect
            else:
                break

        moves += 1

    state.blocks.add((x, y))
    return x != bx or y != by


def push_ramp(state: GameState, rx: int, ry: int, dx: int, dy: int) -> bool:
    """Push a ramp. Returns True if successful."""
    ramp_type = state.ramps.pop((rx, ry))

    x, y = rx, ry
    max_moves = 50
    moves = 0

    while moves < max_moves:
        nx, ny = x + dx, y + dy
        tile = state.get_tile(nx, ny)

        if tile == WALL:
            break
        if tile == LOCKED and (nx, ny) not in state.unlocked:
            break
        if (nx, ny) in state.blocks:
            break
        if (nx, ny) in state.ramps:
            break

        x, y = nx, ny

        # Ramp falls into hole
        if tile == HOLE and (x, y) not in state.filled_holes:
            state.filled_holes.add((x, y))
            return True

        # Corner redirect for ramp
        if tile in CORNERS:
            redirect = get_corner_redirect(tile, dx, dy)
            if redirect:
                dx, dy = redirect
            else:
                break

        moves += 1

    state.ramps[(x, y)] = ramp_type
    return x != rx or y != ry


def solve_bfs(initial_state: GameState, max_depth: int = 30) -> Optional[List[str]]:
    """BFS solver. Returns solution or None."""
    queue = deque([(initial_state, [])])
    visited = {initial_state.to_key()}

    while queue:
        state, path = queue.popleft()

        if len(path) > max_depth:
            continue

        for direction in ['up', 'down', 'left', 'right']:
            new_state = simulate_move(state, direction)
            if new_state is None:
                continue

            if new_state.won:
                return path + [direction]

            if new_state.dead:
                continue

            key = new_state.to_key()
            if key not in visited:
                visited.add(key)
                queue.append((new_state, path + [direction]))

    return None


def create_level(width: int, height: int, mechanics: List[str], difficulty: int) -> Optional[dict]:
    """Generate a single level with specified mechanics."""
    # Create empty grid with walls
    grid = [[WALL if x == 0 or x == width-1 or y == 0 or y == height-1 else ICE
             for x in range(width)] for y in range(height)]

    # Available interior positions
    interior = [(x, y) for x in range(2, width-2) for y in range(2, height-2)]
    random.shuffle(interior)

    if len(interior) < 4:
        return None

    # Place start and goal
    start_x, start_y = interior.pop()
    goal_x, goal_y = interior.pop()
    grid[start_y][start_x] = START
    grid[goal_y][goal_x] = GOAL

    blocks = set()
    ramps = {}

    # Add mechanics based on requested types
    if 'BLOCK' in mechanics and interior:
        num_blocks = random.randint(1, min(3, len(interior)))
        for _ in range(num_blocks):
            if interior:
                bx, by = interior.pop()
                blocks.add((bx, by))

    if 'RAMP' in mechanics and interior:
        num_ramps = random.randint(1, min(2, len(interior)))
        ramp_types = [RAMP_N, RAMP_S, RAMP_E, RAMP_W]
        for _ in range(num_ramps):
            if interior:
                rx, ry = interior.pop()
                ramps[(rx, ry)] = random.choice(ramp_types)
                grid[ry][rx] = ramps[(rx, ry)]

    if 'CORNER' in mechanics:
        # Place corners in appropriate positions
        # Top-left corner uses CORNER_UR (7) - sends away from walls
        # Top-right uses CORNER_UL (8)
        # Bottom-left uses CORNER_DR (9)
        # Bottom-right uses CORNER_DL (0)
        corner_positions = [
            (1, 1, CORNER_UR),
            (width-2, 1, CORNER_UL),
            (1, height-2, CORNER_DR),
            (width-2, height-2, CORNER_DL)
        ]
        num_corners = random.randint(1, min(4, difficulty))
        random.shuffle(corner_positions)
        for i in range(num_corners):
            cx, cy, ctype = corner_positions[i]
            if grid[cy][cx] == ICE:
                grid[cy][cx] = ctype

    if 'HOLE' in mechanics and interior:
        num_holes = random.randint(1, min(2, len(interior)))
        for _ in range(num_holes):
            if interior:
                hx, hy = interior.pop()
                grid[hy][hx] = HOLE

    if 'STICKY' in mechanics and interior:
        num_sticky = random.randint(1, min(2, len(interior)))
        for _ in range(num_sticky):
            if interior:
                sx, sy = interior.pop()
                grid[sy][sx] = STICKY

    if 'KEY' in mechanics and interior:
        # Add key and locked wall
        kx, ky = interior.pop()
        grid[ky][kx] = KEY
        # Place locked wall somewhere
        for x in range(1, width-1):
            for y in range(1, height-1):
                if grid[y][x] == ICE and random.random() < 0.1:
                    grid[y][x] = LOCKED
                    break

    # Add some random internal walls for complexity
    num_walls = random.randint(0, difficulty * 2)
    for _ in range(num_walls):
        if interior:
            wx, wy = interior.pop()
            if grid[wy][wx] == ICE:
                grid[wy][wx] = WALL

    # Create initial state and solve
    initial_state = GameState(
        grid, start_x, start_y, blocks, False, set(), set(), ramps
    )

    solution = solve_bfs(initial_state, max_depth=min(15, 5 + difficulty * 2))

    if solution and len(solution) >= max(2, difficulty):
        # Build level dict
        grid_strings = []
        for y in range(height):
            row = ''
            for x in range(width):
                if (x, y) in blocks:
                    row += BLOCK
                elif (x, y) in ramps:
                    row += ramps[(x, y)]
                else:
                    row += grid[y][x]
            grid_strings.append(row)

        # Detect mechanics used
        used_mechanics = []
        if blocks:
            used_mechanics.append('BLOCK_PUSHED')
        if ramps:
            used_mechanics.append('RAMP_USED')
        if any(c in ''.join(grid_strings) for c in CORNERS):
            used_mechanics.append('CORNER_USED')
        if HOLE in ''.join(grid_strings):
            used_mechanics.append('HOLE_PRESENT')
        if STICKY in ''.join(grid_strings):
            used_mechanics.append('STICKY_USED')
        if KEY in ''.join(grid_strings):
            used_mechanics.append('KEY_USED')

        return {
            'grid': grid_strings,
            'width': width,
            'height': height,
            'par': len(solution),
            'solution': solution,
            'quality': 100 + len(solution) * 10,
            'mechanics': used_mechanics,
            'penguinOnWall': False
        }

    return None


def generate_levels(count: int, progress_callback=None) -> List[dict]:
    """Generate specified number of levels with progressive difficulty."""
    levels = []
    attempts = 0
    max_attempts = count * 100  # More attempts to ensure we hit target

    # Difficulty progression
    difficulty_brackets = [
        (0.1, 1, ['BASIC']),           # Simple sliding
        (0.2, 2, ['BLOCK']),           # Introduce blocks
        (0.3, 2, ['CORNER']),          # Introduce corners
        (0.4, 3, ['BLOCK', 'CORNER']), # Blocks + corners
        (0.5, 3, ['RAMP']),            # Ramps
        (0.6, 3, ['BLOCK', 'RAMP']),   # Blocks + ramps
        (0.7, 4, ['CORNER', 'RAMP']),  # Corners + ramps
        (0.8, 4, ['HOLE']),            # Holes
        (0.9, 5, ['BLOCK', 'HOLE']),   # Blocks + holes
        (1.0, 5, ['BLOCK', 'CORNER', 'RAMP', 'HOLE'])  # Everything
    ]

    while len(levels) < count and attempts < max_attempts:
        attempts += 1

        # Determine difficulty based on progress
        progress = len(levels) / count
        difficulty = 1
        mechanics = ['BASIC']

        for threshold, diff, mechs in difficulty_brackets:
            if progress < threshold:
                difficulty = diff
                mechanics = mechs
                break

        # Phone shape: taller than wide (portrait orientation)
        min_w = 6 + difficulty // 2
        max_w = 8 + difficulty
        min_h = 10 + difficulty
        max_h = 14 + difficulty * 2
        width = random.randint(min_w, max_w)
        height = random.randint(min_h, max_h)

        level = create_level(width, height, mechanics, difficulty)

        if level:
            levels.append(level)
            if progress_callback:
                progress_callback(len(levels), count)

    return levels


def save_chunked(levels: List[dict], output_dir: str, chunk_size: int = 500):
    """Save levels in chunked format."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    chunks = []
    for i in range(0, len(levels), chunk_size):
        chunk_levels = levels[i:i+chunk_size]
        chunk_id = i // chunk_size
        chunk_file = f'chunk_{chunk_id}.json'
        chunk_path = os.path.join(output_dir, chunk_file)

        with open(chunk_path, 'w') as f:
            json.dump({'levels': chunk_levels}, f)

        file_size = os.path.getsize(chunk_path) / 1024
        chunks.append({
            'id': chunk_id,
            'file': chunk_file,
            'start': i,
            'end': i + len(chunk_levels),
            'count': len(chunk_levels),
            'size_kb': round(file_size, 1)
        })

    # Write index
    index = {
        'total_levels': len(levels),
        'chunk_size': chunk_size,
        'chunks': chunks
    }
    with open(os.path.join(output_dir, 'index.json'), 'w') as f:
        json.dump(index, f, indent=2)

    return index


if __name__ == '__main__':
    import sys

    count = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    output_dir = sys.argv[2] if len(sys.argv) > 2 else '/home/ivy/locomot-io/ice_skater/level_chunks'

    print(f"Generating {count} levels...")

    def progress(current, total):
        if current % 100 == 0 or current == total:
            print(f"  {current}/{total} ({100*current//total}%)")

    levels = generate_levels(count, progress)

    print(f"\nGenerated {len(levels)} levels")
    print("Saving to chunks...")

    index = save_chunked(levels, output_dir)

    print(f"Done! {index['total_levels']} levels in {len(index['chunks'])} chunks")
    print(f"Output: {output_dir}")
