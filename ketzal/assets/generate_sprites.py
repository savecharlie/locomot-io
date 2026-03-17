#!/usr/bin/env python3
"""
Ketzal sprite generator — all tiles from consistent rules.

RULES:
  Light source: TOP-LEFT
  Palette: D=#081820  K=#346856  M=#88c070  L=#e0f8d0

  Raised surfaces: L on top/left edges, K/D on bottom/right edges, M face
  Recessed areas: D (mortar, crevices, gaps)

  Snake is a ROUND TUBE:
    Top highlight strip = L
    Upper body = M
    Lower body = K
    Bottom shadow = D
    Left edge = M (lit side)
    Right edge = K (shadow side)

  Snake tapers: head→tail gets thinner
  Width tiers: full=12px, med=10px, thin=6px, tip=4px

  Turns (L-pieces): combine tube halves with proper corner fill
"""

from PIL import Image
import base64, io, os, math

D = (0x08, 0x18, 0x20, 255)
K = (0x34, 0x68, 0x56, 255)
M = (0x88, 0xc0, 0x70, 255)
L = (0xe0, 0xf8, 0xd0, 255)
T = (0, 0, 0, 0)  # transparent

OUT = os.path.dirname(os.path.abspath(__file__)) + '/tiles/custom'
os.makedirs(OUT, exist_ok=True)

def new_tile():
    return Image.new('RGBA', (16, 16), T)

def save(img, name):
    img.save(f'{OUT}/{name}.png')
    # Also save 8x preview
    preview = img.resize((128, 128), Image.NEAREST)
    preview.save(f'{OUT}/{name}_preview.png')

# ═══════════════════════════════════════
# WALL BRICKS — proper per-brick lighting
# ═══════════════════════════════════════

def make_brick_tile(name, top_edge=False, bot_edge=False, left_edge=False, right_edge=False,
                    top_left=False, top_right=False):
    """Generate a wall tile with offset brick pattern.

    Each brick: top-left highlight (L), top/left face (M), body (K), bottom-right shadow (D)
    Mortar between bricks: D

    Brick layout in 16x16:
      Row 0: mortar (D)
      Rows 1-4: brick row A (two bricks: 7px + 1px mortar + 7px + 1px mortar)
      Row 5: mortar (D)  [only if not doing 3-row bricks]
      ...

    Actually let's do 3px tall bricks with 1px mortar:
      Rows 0: mortar
      Rows 1-3: brick row A
      Row 4: mortar
      Rows 5-7: brick row B (offset)
      Row 8: mortar
      Rows 9-11: brick row C
      Row 12: mortar
      Rows 13-15: brick row D (offset)
    """
    img = new_tile()
    px = img.load()

    # Brick dimensions
    BRICK_W = 7  # brick width
    MORTAR = 1   # mortar width
    BRICK_H = 3  # brick height

    # Row A bricks start at x=0: [brick 7px][mortar 1px][brick 7px][mortar 1px] = 16
    # Row B bricks offset by 4: [half 3px][mortar 1px][brick 7px][mortar 1px][half 4px] = 16

    row_a_starts = [0, 8]        # x positions where bricks start in row A
    row_b_starts = [-4, 4, 12]   # offset row (wraps: -4 means brick starts at 12 prev, visible 0-3)

    def draw_brick(bx, by, bw, bh):
        """Draw one brick with per-brick lighting."""
        for dy in range(bh):
            for dx in range(bw):
                x, y = bx + dx, by + dy
                if x < 0 or x >= 16 or y < 0 or y >= 16:
                    continue

                # Determine pixel color based on position within brick
                at_top = (dy == 0)
                at_bot = (dy == bh - 1)
                at_left = (dx == 0)
                at_right = (dx == bw - 1)

                if at_top and at_left:
                    color = L  # top-left corner highlight
                elif at_top:
                    color = M  # top edge lit
                elif at_left:
                    color = M  # left edge lit
                elif at_bot and at_right:
                    color = D  # bottom-right deepest shadow
                elif at_bot:
                    color = K  # bottom edge shadow
                elif at_right:
                    color = K  # right edge shadow
                else:
                    color = K  # brick face (shadowed body)
                    # Add subtle variation: second row from top gets M
                    if dy == 1 and dx > 0 and dx < bw - 1:
                        color = M

                px[x, y] = color

    def fill_mortar():
        """Fill entire tile with mortar (D) first."""
        for y in range(16):
            for x in range(16):
                px[x, y] = D

    fill_mortar()

    # Draw brick rows
    for row_idx in range(4):
        by = row_idx * (BRICK_H + MORTAR) + MORTAR  # skip first mortar line
        # Alternate between row A and row B patterns
        if row_idx % 2 == 0:
            # Row A: bricks at 0 and 8
            draw_brick(0, by, BRICK_W, BRICK_H)
            draw_brick(8, by, BRICK_W, BRICK_H)
        else:
            # Row B: offset - bricks at 4 and 12 (wrapping)
            draw_brick(4, by, BRICK_W, BRICK_H)
            # Left partial brick (wraps from right)
            draw_brick(-4, by, BRICK_W, BRICK_H)  # only 0-2 visible
            # Right partial brick
            draw_brick(12, by, BRICK_W, BRICK_H)  # only 12-15 visible

    # Apply edge overrides for wall position
    if top_edge:
        # Top row gets L highlight strip
        for x in range(16):
            px[x, 0] = L
            if px[x, 1] == D:  # mortar stays D
                pass
            else:
                px[x, 1] = M

    if bot_edge:
        # Bottom gets D shadow strip
        for x in range(16):
            px[x, 15] = D
            px[x, 14] = K

    if left_edge:
        # Left column gets M highlight
        for y in range(16):
            px[0, y] = M if y < 8 else K
            if y == 0 and top_edge:
                px[0, y] = L

    if right_edge:
        # Right column gets D shadow
        for y in range(16):
            px[15, y] = D
            if y < 4:
                px[14, y] = K

    if top_left:
        px[0, 0] = L
        px[1, 0] = L
        px[0, 1] = L

    if top_right:
        px[15, 0] = D
        px[14, 0] = K
        px[15, 1] = D

    save(img, name)
    return img


# ═══════════════════════════════════════
# SNAKE BODY — round tube with tapering
# ═══════════════════════════════════════

def tube_colors_at(px_pos, tube_center, tube_radius):
    """Get color for a pixel based on its position relative to tube center.

    Models a cylinder lit from top-left.
    Returns color or None if outside tube.
    """
    dist = abs(px_pos - tube_center)
    if dist > tube_radius:
        return None

    # Normalize: 0 = center, 1 = edge
    norm = dist / tube_radius if tube_radius > 0 else 0
    # Side: negative = left (toward light), positive = right (away from light)
    side = px_pos - tube_center

    if norm > 0.85:
        # Edge pixels
        return D if side > 0 else K
    elif norm > 0.6:
        return K if side > 0 else M
    elif norm > 0.3:
        return K if side > 0 else M
    else:
        return M  # center-ish


def tube_vcolors_at(py, tube_top, tube_bot):
    """Vertical shading for the tube (top=highlight, bottom=shadow)."""
    h = tube_bot - tube_top
    if h <= 0:
        return M
    rel = (py - tube_top) / h
    if rel < 0.15:
        return L  # top highlight
    elif rel < 0.45:
        return M  # upper lit
    elif rel < 0.75:
        return K  # lower shadow
    else:
        return D  # bottom shadow


def blend_tube(hcolor, vcolor):
    """Blend horizontal and vertical tube shading — take the darker of the two."""
    order = [L, M, K, D]
    hi = order.index(hcolor) if hcolor in order else 1
    vi = order.index(vcolor) if vcolor in order else 1
    return order[max(hi, vi)]


def make_snake_straight_v(name, width):
    """Vertical straight body segment. Width = tube diameter in pixels."""
    img = new_tile()
    px = img.load()

    center = 7.5  # center of 16px tile
    radius = width / 2

    for y in range(16):
        for x in range(16):
            hc = tube_colors_at(x, center, radius)
            if hc is None:
                continue
            # For straight vertical, vertical shading is uniform (tube continues)
            # Just use horizontal shading + subtle scale marks
            px[x, y] = hc

    # Add scale texture — subtle horizontal bands every 4px
    for band_y in [3, 7, 11]:
        for x in range(16):
            if px[x, band_y] == M:
                px[x, band_y] = K
            elif px[x, band_y] == L:
                px[x, band_y] = M

    save(img, name)
    return img


def make_snake_straight_h(name, width):
    """Horizontal straight body segment."""
    img = new_tile()
    px = img.load()

    center = 7.5
    radius = width / 2

    for y in range(16):
        for x in range(16):
            vc = tube_colors_at(y, center, radius)
            if vc is None:
                continue
            # For horizontal tube, the "top" is light, "bottom" is shadow
            # But light is from top-left, so top of horizontal tube = highlight
            px[x, y] = vc

    # Scale bands — vertical every 4px
    for band_x in [3, 7, 11]:
        for y in range(16):
            if px[band_x, y] == M:
                px[band_x, y] = K
            elif px[band_x, y] == L:
                px[band_x, y] = M

    save(img, name)
    return img


def make_snake_turn(name, width, from_dir, to_dir):
    """L-shaped turn piece. from_dir/to_dir are 'up','down','left','right'.

    The turn connects a segment coming FROM from_dir going TO to_dir.
    E.g., from_dir='down', to_dir='right' means the snake was going up,
    and now turns right (the body comes from below, exits to the right).
    """
    img = new_tile()
    px_data = img.load()

    center = 7.5
    radius = width / 2

    # Determine which quadrants get filled
    # The turn creates an L shape. We fill pixels that are within radius of
    # either the vertical or horizontal center line, within the correct quadrants.

    # Map directions to which half of tile they occupy
    # from_dir = where the body is coming from (the tail side)
    # to_dir = where it's going (the head side)

    # Vertical half: from_dir in (up, down)
    # Horizontal half: to_dir in (left, right) or vice versa

    for y in range(16):
        for x in range(16):
            in_v = abs(x - center) <= radius  # within vertical tube
            in_h = abs(y - center) <= radius  # within horizontal tube

            # Determine which halves to draw based on turn direction
            draw = False

            if from_dir == 'down' and to_dir == 'right':
                # Body comes from below, exits right — fill bottom half of V + right half of H
                draw = (in_v and y >= 8) or (in_h and x >= 8)
            elif from_dir == 'down' and to_dir == 'left':
                draw = (in_v and y >= 8) or (in_h and x <= 7)
            elif from_dir == 'up' and to_dir == 'right':
                draw = (in_v and y <= 7) or (in_h and x >= 8)
            elif from_dir == 'up' and to_dir == 'left':
                draw = (in_v and y <= 7) or (in_h and x <= 7)
            elif from_dir == 'right' and to_dir == 'up':
                draw = (in_h and x >= 8) or (in_v and y <= 7)
            elif from_dir == 'right' and to_dir == 'down':
                draw = (in_h and x >= 8) or (in_v and y >= 8)
            elif from_dir == 'left' and to_dir == 'up':
                draw = (in_h and x <= 7) or (in_v and y <= 7)
            elif from_dir == 'left' and to_dir == 'down':
                draw = (in_h and x <= 7) or (in_v and y >= 8)

            if not draw:
                continue

            # Color based on distance from tube surface
            # Use distance from the nearest tube center line
            dx = abs(x - center)
            dy = abs(y - center)

            # For the L-turn, use the minimum distance to either axis
            if in_v and in_h:
                # In the corner — use the outer edge distance
                dist_to_edge = max(dx, dy)
                norm = dist_to_edge / radius if radius > 0 else 0
            elif in_v:
                norm = dx / radius if radius > 0 else 0
            else:
                norm = dy / radius if radius > 0 else 0

            # Light from top-left: top and left = bright, bottom and right = dark
            light_bias = (-x + center) * 0.3 + (-y + center) * 0.3

            if norm > 0.85:
                color = D if light_bias < 0 else K
            elif norm > 0.5:
                color = K if light_bias < -1 else M
            else:
                color = M if light_bias < 0 else L if light_bias > 2 else M

            px_data[x, y] = color

    save(img, name)
    return img


def make_snake_head(name, width, facing='up'):
    """Snake head — rounded front, eyes, slightly wider than body."""
    img = new_tile()
    px = img.load()

    center = 7.5
    radius = width / 2

    # Head shape: rounded at top (front), connects to body at bottom
    for y in range(16):
        for x in range(16):
            dx = abs(x - center)

            # Taper toward the front (top when facing up)
            if facing == 'up':
                # Top 4 rows: round off
                if y < 3:
                    effective_radius = radius * (0.4 + 0.2 * y)
                elif y < 6:
                    effective_radius = radius * (0.7 + 0.1 * (y - 3))
                else:
                    effective_radius = radius
            else:
                effective_radius = radius

            if dx > effective_radius:
                continue

            norm = dx / effective_radius if effective_radius > 0 else 0

            # Vertical shading
            rel_y = y / 15.0

            if norm > 0.85:
                color = D if x > center else K
            elif norm > 0.5:
                color = K if x > center else M
            else:
                if rel_y < 0.2:
                    color = L
                elif rel_y < 0.5:
                    color = M
                elif rel_y < 0.8:
                    color = K
                else:
                    color = K

            px[x, y] = color

    # Eyes — at about 35% down the head
    eye_y = int(16 * 0.35)
    eye_left_x = int(center - radius * 0.4)
    eye_right_x = int(center + radius * 0.4)

    if width >= 10:  # only draw eyes on medium+ heads
        # Eye sockets (D)
        for ey in range(eye_y - 1, eye_y + 2):
            for ex_off in [eye_left_x, eye_right_x]:
                for ex in range(ex_off - 1, ex_off + 1):
                    if 0 <= ex < 16 and 0 <= ey < 16:
                        px[ex, ey] = D
        # Pupils (L)
        px[eye_left_x, eye_y] = L
        px[eye_right_x, eye_y] = L
    elif width >= 6:
        # Small eyes — just 1px each
        px[int(center - 2), eye_y] = D
        px[int(center + 2), eye_y] = D
        px[int(center - 2), eye_y] = L
        px[int(center + 2), eye_y] = L

    save(img, name)
    return img


def make_snake_tail(name, width, facing='down'):
    """Snake tail — tapers to a point."""
    img = new_tile()
    px = img.load()

    center = 7.5
    radius = width / 2

    for y in range(16):
        for x in range(16):
            # Taper: full width at top (connects to body), narrows to a point
            if facing == 'down':
                taper = 1.0 - (y / 15.0) * 0.93  # goes from 100% to ~7%
            else:
                taper = (y / 15.0) * 0.93 + 0.07

            effective_radius = radius * taper
            dx = abs(x - center)

            if dx > effective_radius:
                continue

            norm = dx / effective_radius if effective_radius > 0 else 0

            if norm > 0.8:
                color = D if x > center else K
            elif norm > 0.4:
                color = K if x > center else M
            else:
                color = M

            # Top highlight on upper rows
            if y < 3 and norm < 0.5:
                color = L

            px[x, y] = color

    save(img, name)
    return img


def make_snake_headtail(name, width):
    """Combined head+tail for length-1 snake. Rounded top with eyes, tapers to point at bottom."""
    img = new_tile()
    px = img.load()

    center = 7.5
    radius = width / 2

    for y in range(16):
        for x in range(16):
            dx = abs(x - center)

            # Top: round head (rows 0-7), bottom: taper to tail (rows 8-15)
            if y < 3:
                effective_radius = radius * (0.3 + 0.23 * y)
            elif y < 8:
                effective_radius = radius
            else:
                # Taper: from full width at y=8 to ~1px at y=15
                t = (y - 8) / 7.0
                effective_radius = radius * (1.0 - t * 0.92)

            if dx > effective_radius:
                continue

            norm = dx / effective_radius if effective_radius > 0 else 0
            rel_y = y / 15.0

            if norm > 0.85:
                color = D if x > center else K
            elif norm > 0.5:
                color = K if x > center else M
            else:
                if rel_y < 0.15:
                    color = L
                elif rel_y < 0.4:
                    color = M
                elif rel_y < 0.7:
                    color = K
                else:
                    color = K

            px[x, y] = color

    # Eyes at ~30% down
    eye_y = 4
    if width >= 6:
        el = int(center - 2)
        er = int(center + 2)
        px[el, eye_y] = D
        px[er, eye_y] = D
        px[el, eye_y] = L
        px[er, eye_y] = L

    save(img, name)
    return img


def make_mine(name):
    """Mine/spike — dark sphere with spikes, GB palette."""
    img = new_tile()
    px = img.load()

    center = 7.5

    # Main body — dark sphere
    for y in range(16):
        for x in range(16):
            dist = math.sqrt((x - center)**2 + (y - center)**2)
            if dist > 6:
                continue

            # Dark body with subtle shading
            if dist < 3:
                if x < center and y < center:
                    px[x, y] = K  # highlight
                else:
                    px[x, y] = D
            elif dist < 5:
                px[x, y] = D
            else:
                px[x, y] = D

    # Highlight on top-left
    px[5, 5] = K
    px[6, 5] = K
    px[5, 6] = K

    # Spikes — 4 cardinal + 4 diagonal
    spike_positions = [
        (7, 0), (8, 0), (7, 1), (8, 1),     # top
        (7, 14), (8, 14), (7, 15), (8, 15),  # bottom
        (0, 7), (0, 8), (1, 7), (1, 8),      # left
        (14, 7), (14, 8), (15, 7), (15, 8),  # right
        (2, 2), (3, 3),                        # top-left
        (13, 2), (12, 3),                      # top-right
        (2, 13), (3, 12),                      # bottom-left
        (13, 13), (12, 12),                    # bottom-right
    ]
    for sx, sy in spike_positions:
        px[sx, sy] = D

    save(img, name)
    return img


# ═══════════════════════════════════════
# GENERATE ALL SPRITES
# ═══════════════════════════════════════

print("Generating wall tiles...")
make_brick_tile('wall_center')
make_brick_tile('wall_top', top_edge=True)
make_brick_tile('wall_bottom', bot_edge=True)
make_brick_tile('wall_left', left_edge=True)
make_brick_tile('wall_right', right_edge=True)
make_brick_tile('wall_top_left', top_edge=True, left_edge=True, top_left=True)
make_brick_tile('wall_top_right', top_edge=True, right_edge=True, top_right=True)

# Snake pieces at 3 width tiers
WIDTHS = {'full': 12, 'med': 10, 'thin': 6}

print("Generating snake body pieces...")
for tier, w in WIDTHS.items():
    make_snake_straight_v(f'body_v_{tier}', w)
    make_snake_straight_h(f'body_h_{tier}', w)

    # L-turn pieces — 4 orientations
    # Named by which corner the turn occupies
    make_snake_turn(f'turn_bl_{tier}', w, 'up', 'right')     # └ shape
    make_snake_turn(f'turn_br_{tier}', w, 'up', 'left')      # ┘ shape
    make_snake_turn(f'turn_tl_{tier}', w, 'down', 'right')   # ┌ shape
    make_snake_turn(f'turn_tr_{tier}', w, 'down', 'left')    # ┐ shape

print("Generating snake head/tail (4 rotations each)...")
# Head faces UP at rot=0. Generate all 4 directions to avoid canvas rotation artifacts.
# PIL: ROTATE_90 = CCW, ROTATE_270 = CW
ROT_NAMES = ['up', 'right', 'down', 'left']
ROT_TRANSFORMS = [None, Image.Transpose.ROTATE_270, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_90]

for tier, w in WIDTHS.items():
    base_head = make_snake_head(f'head_{tier}_up', w)
    for r in range(1, 4):
        rotated = base_head.transpose(ROT_TRANSFORMS[r])
        save(rotated, f'head_{tier}_{ROT_NAMES[r]}')

TAIL_ROT_NAMES = ['down', 'right', 'up', 'left']  # base faces DOWN, so rotations map differently
for tier, w in WIDTHS.items():
    base_tail = make_snake_tail(f'tail_{tier}_down', w)
    for r in range(1, 4):
        rotated = base_tail.transpose(ROT_TRANSFORMS[r])
        save(rotated, f'tail_{tier}_{TAIL_ROT_NAMES[r]}')

print("Generating head+tail combo (4 rotations)...")
base_ht = make_snake_headtail('headtail_up', 8)
for r in range(1, 4):
    rotated = base_ht.transpose(ROT_TRANSFORMS[r])
    save(rotated, f'headtail_{ROT_NAMES[r]}')

print("Generating mine...")
make_mine('mine')

# ═══════════════════════════════════════
# GENERATE BASE64 JS LOADER
# ═══════════════════════════════════════

print("\nGenerating JS sprite loader...")

all_sprites = []
for f in sorted(os.listdir(OUT)):
    if f.endswith('.png') and not f.endswith('_preview.png'):
        name = f.replace('.png', '')
        with open(f'{OUT}/{f}', 'rb') as fh:
            b64 = base64.b64encode(fh.read()).decode()
        all_sprites.append((name, b64))

with open(os.path.dirname(os.path.abspath(__file__)) + '/../ketzal_sprites.js', 'w') as f:
    f.write('// ── Ketzal Tile Sprites (generated, base64 embedded) ──\n')
    f.write('// Rules: GB palette, light top-left, per-brick shading, round tube snake\n')
    f.write('const SPRITE = {};\n')
    f.write('let spritesLoaded = 0;\n')
    f.write(f'const SPRITES_TOTAL = {len(all_sprites)};\n')
    f.write('let gameReady = false;\n')
    f.write('function loadSprite(name, b64) {\n')
    f.write("    const img = new Image();\n")
    f.write("    img.src = 'data:image/png;base64,' + b64;\n")
    f.write("    img.onload = () => { SPRITE[name] = img; if (++spritesLoaded >= SPRITES_TOTAL) gameReady = true; };\n")
    f.write('}\n\n')

    for name, b64 in all_sprites:
        f.write(f"loadSprite('{name}', '{b64}');\n")

    f.write('''
function drawSprite(name, x, y) {
    if (!SPRITE[name]) return;
    ctx.drawImage(SPRITE[name], 0, 0, 16, 16, x, y, TILE, TILE);
}

// Draw sprite rotated (0=up, 1=right, 2=down, 3=left)
function drawSpriteRot(name, x, y, rot) {
    if (!SPRITE[name]) return;
    if (rot === 0) { drawSprite(name, x, y); return; }
    ctx.save();
    ctx.translate(x + TILE/2, y + TILE/2);
    ctx.rotate(rot * Math.PI / 2);
    ctx.drawImage(SPRITE[name], 0, 0, 16, 16, -TILE/2, -TILE/2, TILE, TILE);
    ctx.restore();
}
''')

print(f"Generated {len(all_sprites)} sprites")
print("Output: ketzal_sprites.js")

# Also generate a labeled composite preview
COLS_PREVIEW = 8
rows_needed = math.ceil(len(all_sprites) / COLS_PREVIEW)
composite = Image.new('RGBA', (COLS_PREVIEW * 136, rows_needed * 152), (8, 24, 32, 255))

from PIL import ImageDraw, ImageFont
draw = ImageDraw.Draw(composite)

for idx, (name, _) in enumerate(all_sprites):
    col = idx % COLS_PREVIEW
    row = idx // COLS_PREVIEW
    tile_img = Image.open(f'{OUT}/{name}.png')
    big = tile_img.resize((128, 128), Image.NEAREST)
    x = col * 136 + 4
    y = row * 152 + 20
    composite.paste(big, (x, y))
    draw.text((x, y - 16), name, fill=(224, 248, 208, 255))

composite.save(os.path.dirname(os.path.abspath(__file__)) + '/sprite_preview.png')
print("Preview: assets/sprite_preview.png")
