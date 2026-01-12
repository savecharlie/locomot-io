#!/usr/bin/env python3
"""
Ice Skater Level Generator v2
Generates puzzles with mechanic validation - every element serves a purpose.
Includes tutorial generation and enhanced quality scoring.
"""

import json
import random
import copy
import os
from collections import deque
from typing import List, Tuple, Optional, Set, Dict, Union

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

# Ramp climb directions - direction you must go to climb UP the ramp
RAMP_CLIMB_DIR = {
    RAMP_N: (0, -1),   # ^ high end north, climb by going up
    RAMP_S: (0, 1),    # v high end south, climb by going down
    RAMP_E: (1, 0),    # > high end east, climb by going right
    RAMP_W: (-1, 0),   # < high end west, climb by going left
}

# Mechanics that can be validated
ALL_MECHANICS = {
    'BLOCK_PUSH',       # Pushed a block
    'HOLE_FILL',        # Block/ramp fell in hole
    'CORNER_REDIRECT',  # Used corner to change direction
    'STICKY_STOP',      # Stopped on sticky tile
    'RAMP_USE',         # Any ramp interaction (climb/descend)
    'KEY_LOCK',         # Used key to open lock
}

# Tutorial curriculum - teaches one mechanic at a time
TUTORIAL_CURRICULUM = [
    # Stage 1: Pure sliding (3 levels)
    {'name': 'slide', 'count': 3, 'elements': [], 'teach': None,
     'max_par': 2, 'grid_size': (4, 6), 'text': 'Swipe to slide'},

    # Stage 2: Blocks exist as obstacles (3 levels)
    {'name': 'blocks_obstacle', 'count': 3, 'elements': ['BLOCK'], 'teach': None,
     'forbidden': ['BLOCK_PUSH'], 'max_par': 3, 'grid_size': (5, 7), 'text': 'Blocks stop you'},

    # Stage 3: Must push blocks (3 levels)
    {'name': 'block_push', 'count': 3, 'elements': ['BLOCK'], 'teach': 'BLOCK_PUSH',
     'max_par': 4, 'grid_size': (5, 8), 'text': 'Push blocks'},

    # Stage 4: Corners redirect (3 levels)
    {'name': 'corners', 'count': 3, 'elements': ['CORNER'], 'teach': 'CORNER_REDIRECT',
     'max_par': 4, 'grid_size': (5, 7), 'text': 'Corners redirect'},

    # Stage 5: Holes + blocks (3 levels)
    {'name': 'hole_fill', 'count': 3, 'elements': ['BLOCK', 'HOLE'], 'teach': 'HOLE_FILL',
     'max_par': 5, 'grid_size': (6, 8), 'text': 'Fill holes with blocks'},

    # Stage 6: Sticky tiles (3 levels)
    {'name': 'sticky', 'count': 3, 'elements': ['STICKY'], 'teach': 'STICKY_STOP',
     'max_par': 5, 'grid_size': (5, 8), 'text': 'Sticky tiles stop you'},

    # Stage 7: Ramps (3 levels)
    {'name': 'ramps', 'count': 3, 'elements': ['RAMP'], 'teach': 'RAMP_USE',
     'max_par': 5, 'grid_size': (6, 9), 'text': 'Ramps change height'},

    # Stage 8: Keys and locks (3 levels)
    {'name': 'keys', 'count': 3, 'elements': ['KEY'], 'teach': 'KEY_LOCK',
     'max_par': 6, 'grid_size': (6, 9), 'text': 'Keys open locks'},
]


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
                 ramps: Dict[Tuple[int, int], str] = None,
                 player_height: int = 0):
        self.grid = grid
        self.width = len(grid[0])
        self.height = len(grid)
        self.player_x = player_x
        self.player_y = player_y
        self.player_height = player_height  # 0 = ground, 1 = elevated
        self.blocks = blocks
        self.has_key = has_key
        self.unlocked = unlocked or set()
        self.filled_holes = filled_holes or set()
        self.ramps = ramps or {}
        self.dead = False
        self.won = False

    def copy(self):
        new_state = GameState(
            [row[:] for row in self.grid],
            self.player_x, self.player_y,
            set(self.blocks), self.has_key,
            set(self.unlocked), set(self.filled_holes),
            dict(self.ramps), self.player_height
        )
        return new_state

    def get_tile(self, x: int, y: int) -> str:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        return WALL

    def to_key(self) -> tuple:
        """Hashable state for BFS."""
        return (
            self.player_x, self.player_y, self.player_height,
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
    height = new_state.player_height

    # Slide until hitting something
    moves = 0
    max_moves = 50  # Safety limit

    while moves < max_moves:
        nx, ny = x + dx, y + dy
        tile = new_state.get_tile(nx, ny)

        # Wall always stops movement
        if tile == WALL:
            break

        # Locked wall
        if tile == LOCKED and (nx, ny) not in new_state.unlocked:
            if height == 0:
                # Try to unlock if we have key
                if new_state.has_key:
                    new_state.unlocked.add((nx, ny))
                    new_state.has_key = False
                break
            else:
                break  # Can't pass over locked walls even when elevated

        # Handle based on elevation
        if height == 0:
            # GROUND LEVEL

            # Block - try to push
            if (nx, ny) in new_state.blocks:
                pushed = push_block(new_state, nx, ny, dx, dy)
                if not pushed:
                    break
                # Player ends up on block's original position
                x, y = nx, ny
                break

            # Ramp - climb or push
            if (nx, ny) in new_state.ramps:
                ramp_type = new_state.ramps[(nx, ny)]
                climb_dir = RAMP_CLIMB_DIR[ramp_type]

                if (dx, dy) == climb_dir:
                    # Climbing UP the ramp
                    x, y = nx, ny
                    height = 1
                    # Continue sliding at height 1
                    moves += 1
                    continue
                else:
                    # Try to push the ramp
                    pushed = push_ramp(new_state, nx, ny, dx, dy)
                    if not pushed:
                        break
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

        else:
            # ELEVATED (height == 1)

            # Pass over ground-level blocks
            if (nx, ny) in new_state.blocks:
                x, y = nx, ny
                moves += 1
                continue

            # Ramp - descend or pass over
            if (nx, ny) in new_state.ramps:
                ramp_type = new_state.ramps[(nx, ny)]
                climb_dir = RAMP_CLIMB_DIR[ramp_type]
                descent_dir = (-climb_dir[0], -climb_dir[1])

                if (dx, dy) == descent_dir:
                    # Going DOWN the ramp
                    x, y = nx, ny
                    height = 0
                    # Continue sliding at height 0
                    moves += 1
                    continue
                else:
                    # Pass over ramp (we're elevated)
                    x, y = nx, ny
                    moves += 1
                    continue

            # Move to new position
            x, y = nx, ny

            # At height 1, pass over holes
            # At height 1, pass over corners (no redirect)
            # At height 1, can still pick up keys and reach goal

            if tile == KEY:
                new_state.has_key = True
                new_state.grid[y][x] = ICE

            if tile == GOAL:
                new_state.won = True
                break

            if tile == STICKY:
                height = 0  # Land on sticky
                break

        moves += 1

    if x == state.player_x and y == state.player_y and height == state.player_height:
        return None  # No movement

    new_state.player_x = x
    new_state.player_y = y
    new_state.player_height = height
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


# ============================================================================
# MECHANIC VALIDATION SYSTEM
# ============================================================================

def trace_mechanics(state: GameState, solution: List[str]) -> Set[str]:
    """Trace a solution and return all mechanics USED."""
    used = set()
    current = state.copy()

    for direction in solution:
        old_blocks = set(current.blocks)
        old_height = current.player_height
        old_key = current.has_key
        old_unlocked = set(current.unlocked)
        old_filled = set(current.filled_holes)
        old_ramps = dict(current.ramps)

        new_state = simulate_move(current, direction)
        if new_state is None:
            continue

        # Detect block push
        if new_state.blocks != old_blocks:
            used.add('BLOCK_PUSH')

        # Detect hole fill (block or ramp disappeared)
        if len(new_state.blocks) < len(old_blocks) or len(new_state.ramps) < len(old_ramps):
            used.add('HOLE_FILL')

        # Detect ramp use (height changed)
        if new_state.player_height != old_height:
            used.add('RAMP_USE')

        # Detect key collection
        if new_state.has_key and not old_key:
            used.add('KEY_LOCK')

        # Detect lock opening
        if new_state.unlocked != old_unlocked:
            used.add('KEY_LOCK')

        # Detect sticky stop - check end tile
        end_tile = current.grid[new_state.player_y][new_state.player_x]
        if end_tile == STICKY:
            used.add('STICKY_STOP')

        current = new_state

    return used


def is_mechanic_required(state: GameState, mechanic: str, max_depth: int = 30) -> bool:
    """Check if a mechanic is REQUIRED (level unsolvable without it)."""
    modified = state.copy()

    if mechanic == 'BLOCK_PUSH':
        # Make blocks into walls (immovable)
        for bx, by in list(modified.blocks):
            modified.grid[by][bx] = WALL
        modified.blocks = set()

    elif mechanic == 'CORNER_REDIRECT':
        # Replace corners with walls
        for y in range(modified.height):
            for x in range(modified.width):
                if modified.grid[y][x] in CORNERS:
                    modified.grid[y][x] = WALL

    elif mechanic == 'STICKY_STOP':
        # Replace sticky with ice
        for y in range(modified.height):
            for x in range(modified.width):
                if modified.grid[y][x] == STICKY:
                    modified.grid[y][x] = ICE

    elif mechanic == 'RAMP_USE':
        # Replace ramps with walls
        for pos in list(modified.ramps.keys()):
            rx, ry = pos
            modified.grid[ry][rx] = WALL
        modified.ramps = {}

    elif mechanic == 'KEY_LOCK':
        # Remove keys and open all locks
        for y in range(modified.height):
            for x in range(modified.width):
                if modified.grid[y][x] == KEY:
                    modified.grid[y][x] = ICE
                if modified.grid[y][x] == LOCKED:
                    modified.grid[y][x] = ICE

    elif mechanic == 'HOLE_FILL':
        # Make holes into walls (can't be filled)
        for y in range(modified.height):
            for x in range(modified.width):
                if modified.grid[y][x] == HOLE:
                    modified.grid[y][x] = WALL

    # Try to solve without the mechanic
    return solve_bfs(modified, max_depth=max_depth) is None


def has_element(state: GameState, element: str) -> bool:
    """Check if a level has a particular element type."""
    if element == 'BLOCK':
        return len(state.blocks) > 0
    elif element == 'CORNER':
        for row in state.grid:
            for tile in row:
                if tile in CORNERS:
                    return True
        return False
    elif element == 'STICKY':
        return any(STICKY in row for row in state.grid)
    elif element == 'RAMP':
        return len(state.ramps) > 0
    elif element == 'KEY':
        return any(KEY in row for row in state.grid)
    elif element == 'HOLE':
        return any(HOLE in row for row in state.grid)
    return False


def analyze_level(state: GameState, solution: List[str]) -> dict:
    """Full mechanic analysis of a level."""
    used = trace_mechanics(state, solution)

    required = set()
    for mechanic in used:
        if is_mechanic_required(state, mechanic):
            required.add(mechanic)

    # Also check corner redirect specially (might not show in trace)
    if has_element(state, 'CORNER') and is_mechanic_required(state, 'CORNER_REDIRECT'):
        required.add('CORNER_REDIRECT')

    return {
        'mechanics_used': list(used),
        'mechanics_required': list(required),
        'num_required': len(required)
    }


def calculate_quality(state: GameState, solution: List[str], analysis: dict = None) -> int:
    """Calculate quality score based on mechanic engagement."""
    if analysis is None:
        analysis = analyze_level(state, solution)

    par = len(solution)
    base = 100 + par * 10

    # Bonus for mechanics used
    mechanic_bonus = len(analysis['mechanics_used']) * 15

    # Bigger bonus for mechanics REQUIRED
    required_bonus = analysis['num_required'] * 30

    # Clean design bonus (all elements serve a purpose)
    # Simplified: if we have blocks and BLOCK_PUSH is required, that's clean
    clean_bonus = 0
    if state.blocks and 'BLOCK_PUSH' in analysis['mechanics_required']:
        clean_bonus += 20
    if has_element(state, 'CORNER') and 'CORNER_REDIRECT' in analysis['mechanics_required']:
        clean_bonus += 20

    return base + mechanic_bonus + required_bonus + clean_bonus


# ============================================================================
# TUTORIAL GENERATION
# ============================================================================

def create_tutorial_level(width: int, height: int, elements: List[str]) -> Optional[GameState]:
    """Create a level with only specified elements for tutorials."""
    grid = [[WALL if x == 0 or x == width-1 or y == 0 or y == height-1 else ICE
             for x in range(width)] for y in range(height)]

    interior = [(x, y) for x in range(1, width-1) for y in range(1, height-1)]
    random.shuffle(interior)

    if len(interior) < 2:
        return None

    start_x, start_y = interior.pop()
    goal_x, goal_y = interior.pop()
    grid[start_y][start_x] = START
    grid[goal_y][goal_x] = GOAL

    blocks = set()
    ramps = {}

    # Place only allowed elements
    if 'BLOCK' in elements:
        num = random.randint(1, min(2, len(interior)))
        for _ in range(num):
            if interior:
                bx, by = interior.pop()
                blocks.add((bx, by))

    if 'HOLE' in elements:
        num = random.randint(1, 1)
        for _ in range(num):
            if interior:
                hx, hy = interior.pop()
                grid[hy][hx] = HOLE

    if 'STICKY' in elements:
        num = random.randint(1, 2)
        for _ in range(num):
            if interior:
                tx, ty = interior.pop()
                grid[ty][tx] = STICKY

    if 'CORNER' in elements:
        # Place 1 corner in a useful position
        corner_spots = [(1, 1, CORNER_UR), (width-2, 1, CORNER_UL),
                        (1, height-2, CORNER_DR), (width-2, height-2, CORNER_DL)]
        random.shuffle(corner_spots)
        cx, cy, ctype = corner_spots[0]
        if grid[cy][cx] == ICE:
            grid[cy][cx] = ctype

    if 'RAMP' in elements:
        if interior:
            rx, ry = interior.pop()
            ramp_type = random.choice([RAMP_N, RAMP_S, RAMP_E, RAMP_W])
            ramps[(rx, ry)] = ramp_type
            grid[ry][rx] = ramp_type

    if 'KEY' in elements:
        if interior:
            kx, ky = interior.pop()
            grid[ky][kx] = KEY
            # Place locked wall
            for x in range(1, width-1):
                for y in range(1, height-1):
                    if grid[y][x] == ICE and random.random() < 0.15:
                        grid[y][x] = LOCKED
                        break
                else:
                    continue
                break

    # Add some internal walls
    num_walls = random.randint(0, 2)
    for _ in range(num_walls):
        if interior:
            wx, wy = interior.pop()
            if grid[wy][wx] == ICE:
                grid[wy][wx] = WALL

    return GameState(grid, start_x, start_y, blocks, False, set(), set(), ramps)


def generate_tutorial(params: dict, max_attempts: int = 500) -> Optional[dict]:
    """Generate a single tutorial level meeting the params."""
    width, height = params['grid_size']
    elements = params.get('elements', [])
    teach = params.get('teach')
    forbidden = set(params.get('forbidden', []))
    max_par = params.get('max_par', 10)

    for _ in range(max_attempts):
        state = create_tutorial_level(width, height, elements)
        if state is None:
            continue

        solution = solve_bfs(state, max_depth=max_par)
        if solution is None:
            continue

        if len(solution) > max_par:
            continue

        if len(solution) < 1:
            continue

        # Check teaches required mechanic
        if teach:
            if not is_mechanic_required(state, teach):
                continue

        # Check no forbidden mechanics required
        skip = False
        for fm in forbidden:
            if is_mechanic_required(state, fm):
                skip = True
                break
        if skip:
            continue

        # Build level dict
        grid_strings = []
        for y in range(state.height):
            row = ''
            for x in range(state.width):
                if (x, y) in state.blocks:
                    row += BLOCK
                elif (x, y) in state.ramps:
                    row += state.ramps[(x, y)]
                else:
                    row += state.grid[y][x]
            grid_strings.append(row)

        return {
            'grid': grid_strings,
            'width': state.width,
            'height': state.height,
            'par': len(solution),
            'solution': solution,
            'tutorial_text': params.get('text', ''),
            'stage': params.get('name', ''),
            'penguinOnWall': False
        }

    return None


def generate_all_tutorials(progress_callback=None, candidates_per_stage: int = 100) -> List[dict]:
    """Generate all tutorial levels following the curriculum.

    Args:
        candidates_per_stage: Generate this many candidates per stage, keep best 3
    """
    tutorials = []
    learned = set()
    total_stages = len(TUTORIAL_CURRICULUM)

    for stage_idx, stage in enumerate(TUTORIAL_CURRICULUM):
        print(f"  Generating stage: {stage['name']} ({candidates_per_stage} candidates)...")

        # Forbidden = everything not yet learned (except what we're teaching)
        forbidden = set(ALL_MECHANICS) - learned
        if stage.get('teach'):
            forbidden.discard(stage['teach'])

        # Add any explicitly forbidden mechanics
        forbidden.update(stage.get('forbidden', []))

        params = {
            'elements': stage['elements'],
            'teach': stage.get('teach'),
            'forbidden': forbidden,
            'max_par': stage['max_par'],
            'grid_size': stage['grid_size'],
            'text': stage['text'],
            'name': stage['name']
        }

        # Generate many candidates
        candidates = []
        for attempt in range(candidates_per_stage * 10):  # More attempts to hit target
            level = generate_tutorial(params, max_attempts=50)
            if level:
                # Score the tutorial (shorter = better for tutorials, but must work)
                level['tutorial_quality'] = 100 - level['par'] * 5  # Prefer shorter
                candidates.append(level)
                if len(candidates) >= candidates_per_stage:
                    break

        if candidates:
            # Sort by quality and keep best 3
            candidates.sort(key=lambda x: x['tutorial_quality'], reverse=True)
            best = candidates[:stage['count']]
            for level in best:
                level['index'] = len(tutorials)
                tutorials.append(level)
            print(f"    Kept {len(best)} best (quality range: {best[-1]['tutorial_quality']}-{best[0]['tutorial_quality']})")
        else:
            print(f"    WARNING: No valid tutorials for {stage['name']}")

        if progress_callback:
            progress_callback(stage_idx + 1, total_stages)

        # After this stage, player has learned the mechanic
        if stage.get('teach'):
            learned.add(stage['teach'])

    return tutorials


# ============================================================================
# LEVEL GENERATION (ORIGINAL + ENHANCED)
# ============================================================================

def create_level(width: int, height: int, mechanics: List[str], difficulty: int, narrow: bool = False) -> Optional[dict]:
    """Generate a single level with specified mechanics."""
    # Create empty grid with walls
    grid = [[WALL if x == 0 or x == width-1 or y == 0 or y == height-1 else ICE
             for x in range(width)] for y in range(height)]

    # Available interior positions - for narrow puzzles use all non-wall tiles
    if narrow:
        interior = [(x, y) for x in range(1, width-1) for y in range(1, height-1)]
    else:
        interior = [(x, y) for x in range(2, width-2) for y in range(2, height-2)]
    random.shuffle(interior)

    if len(interior) < 2:  # Need at least start and goal
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

        # Enhanced mechanic analysis
        analysis = analyze_level(initial_state, solution)
        quality = calculate_quality(initial_state, solution, analysis)

        return {
            'grid': grid_strings,
            'width': width,
            'height': height,
            'par': len(solution),
            'solution': solution,
            'quality': quality,
            'mechanics_used': analysis['mechanics_used'],
            'mechanics_required': analysis['mechanics_required'],
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


def generate_narrow_levels(count: int, interior_w: int, interior_h: int, progress_callback=None) -> List[dict]:
    """Generate narrow puzzles with fixed interior dimensions."""
    levels = []
    attempts = 0
    max_attempts = count * 200  # More attempts needed for constrained puzzles

    # Total dimensions = interior + 2 walls
    width = interior_w + 2
    height = interior_h + 2

    # Mechanics that work in narrow spaces
    narrow_mechanics = [
        ['BASIC'],
        ['BLOCK'],
        ['STICKY'],
        ['BLOCK', 'STICKY'],
    ]

    while len(levels) < count and attempts < max_attempts:
        attempts += 1

        mechanics = random.choice(narrow_mechanics)
        difficulty = random.randint(1, 3)

        level = create_level(width, height, mechanics, difficulty, narrow=True)

        if level:
            levels.append(level)
            if progress_callback and (len(levels) % 100 == 0 or len(levels) == count):
                progress_callback(len(levels), count)

    return levels


def generate_ratio_levels(count: int, ratio_w: int, ratio_h: int, progress_callback=None) -> List[dict]:
    """Generate puzzles with a specific width:height ratio."""
    levels = []
    attempts = 0
    max_attempts = count * 100

    # Generate various sizes maintaining the ratio
    # For 1:3 ratio, use widths 5-8 giving heights 15-24
    base_widths = [5, 6, 7, 8]

    while len(levels) < count and attempts < max_attempts:
        attempts += 1

        # Pick a base width and calculate height from ratio
        base_w = random.choice(base_widths)
        width = base_w
        height = base_w * ratio_h // ratio_w

        # Determine difficulty based on progress
        progress = len(levels) / count if count > 0 else 0
        if progress < 0.3:
            difficulty = random.randint(1, 2)
            mechanics = random.choice([['BASIC'], ['BLOCK'], ['CORNER']])
        elif progress < 0.6:
            difficulty = random.randint(2, 3)
            mechanics = random.choice([['BLOCK'], ['CORNER'], ['BLOCK', 'CORNER'], ['RAMP']])
        else:
            difficulty = random.randint(3, 5)
            mechanics = random.choice([['BLOCK', 'CORNER'], ['RAMP'], ['BLOCK', 'RAMP'], ['HOLE'], ['BLOCK', 'HOLE']])

        level = create_level(width, height, mechanics, difficulty)

        if level:
            levels.append(level)
            if progress_callback and (len(levels) % 100 == 0 or len(levels) == count):
                progress_callback(len(levels), count)

    return levels


if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Ice Skater Level Generator v2 - with mechanic validation')
    parser.add_argument('--tutorials', action='store_true', help='Generate tutorial levels only')
    parser.add_argument('--with-tutorials', action='store_true', help='Generate tutorials + regular levels')
    parser.add_argument('--tutorial-candidates', type=int, default=100, help='Candidates per tutorial stage (default: 100)')
    parser.add_argument('--narrow', type=str, help='Generate narrow puzzles with WxH interior (e.g., 1x3)')
    parser.add_argument('--ratio', type=str, help='Generate puzzles with W:H ratio (e.g., 1:3 for 3x tall as wide)')
    parser.add_argument('--count', type=int, default=50000, help='Number to generate (default: 50000)')
    parser.add_argument('--top', type=int, default=500, help='Top N to keep (default: 500)')
    parser.add_argument('--output', type=str, default='/home/ivy/locomot-io/ice_skater/level_chunks')
    args = parser.parse_args()

    def progress(current, total):
        if current % 100 == 0 or current == total:
            pct = 100 * current // total if total > 0 else 0
            print(f"  {current}/{total} ({pct}%)")

    # Tutorial-only mode
    if args.tutorials:
        print(f"Generating tutorial levels ({args.tutorial_candidates} candidates per stage)...")
        tutorials = generate_all_tutorials(progress, candidates_per_stage=args.tutorial_candidates)
        print(f"\nGenerated {len(tutorials)} tutorial levels")

        # Save tutorials
        import os
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, 'tutorials.json'), 'w') as f:
            json.dump({'tutorials': tutorials}, f)
        print(f"Saved to {args.output}/tutorials.json")
        sys.exit(0)

    # Combined mode: tutorials + regular levels
    if args.with_tutorials:
        print("=" * 60)
        print(f"PHASE 1: Generating tutorial levels ({args.tutorial_candidates} candidates per stage)...")
        print("=" * 60)
        tutorials = generate_all_tutorials(progress, candidates_per_stage=args.tutorial_candidates)
        print(f"Generated {len(tutorials)} tutorial levels\n")

        print("=" * 60)
        print(f"PHASE 2: Generating {args.count} regular levels...")
        print("=" * 60)
        levels = generate_levels(args.count, progress)
        print(f"Generated {len(levels)} regular levels")

        if len(levels) > 0:
            # Sort by quality and keep top N
            levels.sort(key=lambda x: x['quality'], reverse=True)
            levels = levels[:args.top]
            print(f"Kept top {len(levels)} by quality (range: {levels[-1]['quality']}-{levels[0]['quality']})")

            # Sort by difficulty
            levels.sort(key=lambda x: x['par'])
            print(f"Sorted by difficulty (par range: {levels[0]['par']}-{levels[-1]['par']})")

        # Combine: tutorials first, then regular levels
        all_levels = tutorials + levels
        print(f"\nTotal: {len(all_levels)} levels ({len(tutorials)} tutorials + {len(levels)} regular)")

        print("\nSaving to chunks...")
        index = save_chunked(all_levels, args.output)
        print(f"Done! {index['total_levels']} levels in {len(index['chunks'])} chunks")

        # Also save tutorials separately
        with open(os.path.join(args.output, 'tutorials.json'), 'w') as f:
            json.dump({'tutorials': tutorials, 'count': len(tutorials)}, f)
        print(f"Tutorials also saved to {args.output}/tutorials.json")
        sys.exit(0)

    if args.ratio:
        # Parse W:H ratio format
        ratio_w, ratio_h = map(int, args.ratio.split(':'))
        print(f"Generating {args.count} puzzles with {ratio_w}:{ratio_h} ratio, keeping top {args.top}...")
        levels = generate_ratio_levels(args.count, ratio_w, ratio_h, progress)
    elif args.narrow:
        # Parse WxH format
        interior_w, interior_h = map(int, args.narrow.lower().split('x'))
        print(f"Generating {args.count} narrow puzzles ({interior_w}x{interior_h} interior), keeping top {args.top}...")
        levels = generate_narrow_levels(args.count, interior_w, interior_h, progress)
    else:
        print(f"Generating {args.count} levels, keeping top {args.top} by quality, sorted by difficulty...")
        levels = generate_levels(args.count, progress)

    print(f"\nGenerated {len(levels)} levels")

    if len(levels) == 0:
        print("No valid levels generated!")
        sys.exit(1)

    # Sort by quality (descending) and take top N
    levels.sort(key=lambda x: x['quality'], reverse=True)
    levels = levels[:args.top]
    print(f"Kept top {len(levels)} by quality (range: {levels[-1]['quality']}-{levels[0]['quality']})")

    # Sort by difficulty (par)
    levels.sort(key=lambda x: x['par'])
    print(f"Sorted by difficulty (par range: {levels[0]['par']}-{levels[-1]['par']})")

    print("Saving to chunks...")
    index = save_chunked(levels, args.output)

    print(f"Done! {index['total_levels']} levels in {len(index['chunks'])} chunks")
    print(f"Output: {args.output}")
