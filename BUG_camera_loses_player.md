# Bug: Camera Loses Player on Game Start

## Symptom
Player disappears from view sometimes right when the game starts, especially when wrapping around world edges.

## Root Cause Found
In `init()` function (line ~5498-5501), the camera wrap tracking variables `lastHeadX` and `lastHeadY` are NOT initialized to the spawn position.

They're initialized globally to 0 at line 2257:
```js
let lastHeadX = 0, lastHeadY = 0;
```

But in `init()`, only these are set:
```js
cameraZoom = 2; targetZoom = 2;
cameraX = spawnX * GRID - canvas.width / 2 / cameraZoom;
cameraY = spawnY * GRID - canvas.height / 2 / cameraZoom;
cameraWrapOffsetX = 0; cameraWrapOffsetY = 0;
```

Compare to `doRespawn()` (line 5701) which correctly sets:
```js
lastHeadX = spawnX; lastHeadY = spawnY; // Reset wrap tracking
```

## Why This Causes the Bug
On first frame after init():
1. `lastHeadX = 0` (from global init)
2. `head.x = spawnX` (random 5 to ~295)
3. Camera update calculates: `dx = head.x - lastHeadX = spawnX - 0`
4. If `spawnX > 150` (WORLD_COLS/2), this triggers wrap detection:
   - `dx > WORLD_COLS / 2` → `cameraWrapOffsetX -= WORLD_COLS` → becomes -300
5. Camera targets `virtualHeadX = head.x + cameraWrapOffsetX = spawnX + (-300)`
6. Player is now off-screen!

## Fix
Add to `init()` after line 5501:
```js
lastHeadX = spawnX; lastHeadY = spawnY; // Reset wrap tracking
```

## Attempts
1. [x] Add lastHeadX/Y initialization to init() - APPLIED - Fixed starting issue
2. [ ] Fix wrap invisibility - INVESTIGATING

## Fix 1 Applied (starting issue)
Added at line 5502 in trains.html:
```js
lastHeadX = spawnX; lastHeadY = spawnY; // Initialize wrap detection
```

## Wrap Invisibility Analysis
Problem: After player wraps (e.g., x=299 → x=0):
- cameraWrapOffsetX becomes 300
- virtualHeadX = 0 + 300 = 300
- Camera targets position 300 (in pixels: ~3600)
- Player drawn at real position 0
- With translate(-3600), player appears at -3600 (off screen!)

Root cause: Camera uses virtual coords but entities draw at real coords.

Possible fixes:
1. Draw player at virtual position: (seg.x + cameraWrapOffsetX) * GRID
2. Keep camera in real bounds, use getWrapPositions()
3. Remove wrap offset system entirely, accept camera jump on wrap

Testing fix 3: Disabled wrap offset system entirely.

## Fix 2 Applied (wrap invisibility)
Disabled the wrap offset accumulation. Camera now always tracks player's real position.
Camera will "jump" on wrap, but player stays visible.

```js
// Update last head position for next frame (wrap offset disabled - was causing invisibility)
lastHeadX = head.x;
lastHeadY = head.y;
// Keep wrap offsets at 0 - camera will jump on wrap but player stays visible
cameraWrapOffsetX = 0;
cameraWrapOffsetY = 0;
```
