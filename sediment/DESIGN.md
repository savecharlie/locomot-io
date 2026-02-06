# SEDIMENT v2

**A multiplayer auto-runner where you ARE a Tetris piece. When you die, your body snaps to the grid and becomes terrain. Completed rows clear. Every player shapes the level.**

## The Goal

Go far. That's it. Your score is how far right you get. Everything else — building, clearing, using other people's blocks — is how you get there.

## Core Loop

You auto-run through hell. You're a tetromino — an actual I-piece or T-piece or L-piece, with full collision. You bonk into ceilings, clip ledges, tumble into chasms. When you die, your body snaps to the grid and becomes terrain. You respawn as a new random shape and keep running.

The level starts barren and brutal. Over time, deaths fill it with blocks. Bridges form across chasms. Stairways grow up walls. Spike fields get paved over. The terrain is built entirely from collective failure.

## The Tetromino Mechanic

- You ARE your shape. Full collision. An I-piece is 4 tiles tall and bonks every ceiling. An O-piece is compact and easy to run. Each shape has personality.
- Swipe left/right to rotate while running. Rotation is a survival skill AND a construction tool — orient your piece for the gap you want to fill.
- The death button is the core skill move. You see a chasm, rotate to the right orientation, position yourself, and voluntarily die to place your piece exactly where it helps.
- Each life you get a random shape (bag of 7, all shapes before repeating) and a random material type.
- Must travel a minimum distance from spawn before your death leaves a block (prevents spam griefing).

## Materials

Each tetromino is one material:

- **Solid** (sandstone) — plain terrain. Stand on it, wall-jump off it.
- **Spring** (sage green) — bouncy. Land on it and get launched upward. A spring I-piece across a chasm is a 4-cell trampoline.
- **Booster** (steel blue) — accelerates you forward. A booster T-piece on the floor is a speed zone.

You can see what you are while running — your color tells you your material.

## Row Clearing — The Ecosystem

When a horizontal row fills completely (terrain + corpse blocks), it clears. **Only corpse blocks vanish — terrain stays.** Blocks above fall down (Tetris gravity). Cascading clears are possible.

This creates a self-balancing player ecosystem:

- **Builders** place pieces strategically to create paths
- **Cloggers** fill areas with blocks (intentionally or not) — they're setting up row clears
- **Janitors** see a nearly full row and complete it to reshape terrain
- **Pioneers** push into new territory, dying where no one has been
- **AFKers** pile blocks in the same spot — easy row completions

Every negative behavior feeds the ecosystem. Clogging enables clearing. Clearing creates space for new building. There is no griefing because every action feeds the cycle.

Real geology: deposition → compaction → erosion → new deposition. The level breathes.

## Controls

**Mobile:**
- Tap = jump
- Swipe down = dive (fast fall for precise vertical placement)
- Swipe left/right = rotate piece
- Death button (skull) = voluntary death / snap to grid

**Desktop:**
- Space/Up = jump
- Down = dive
- Left/Right = rotate
- X = voluntary death
- R = hard reset

## Movement

- Auto-run (always moving right)
- Double jump (second jump at 70% strength)
- Wall jump (redirects off walls)
- Dive (fast downward slam after double jump)
- Coyote time (brief grace period after leaving ground)

These are all construction tools disguised as movement abilities. Double jump for height positioning. Dive for precision vertical placement. Wall jump to reach high spots.

## Grid

Everything on a unified TILE grid. Level terrain, placed blocks, player collision — all the same grid.

**Blocks fall when placed.** When you die, your tetromino detaches and falls (visibly, with a satisfying drop animation) until it lands on terrain or other blocks. Then it locks in place. This means:
- Blocks pile up naturally, forming dense layers → rows complete → clearing works
- Chasms fill from the bottom up (like real sediment settling)
- No floating bridges — you cross chasms by FILLING them
- Chasms have an implicit floor at the bottom of the level grid

## Level Generation

- Procedural segments (40 tiles wide)
- Difficulty scales with death count
- Hazards: chasms, spike fields, ceiling spikes, spike walls, buzzsaws
- Segments generate as you advance (endless)
- Blocks placed ON hazards neutralize them (spikes covered by a corpse block = safe)

## Multiplayer (planned)

- Same daily seed level for everyone
- Ghost silhouettes of recent players
- Everyone's placed blocks in the same world via PartyKit
- Blocks persist for the day, reset with new seed

## Visual Style

- 12px tile grid, 14 tiles tall
- Deep indigo void background (#10101E)
- Warm gold player highlights
- Material colors: sandstone, sage green, steel blue
- Player pieces are bright/glowing, placed pieces weather and darken
- Dead eyes on corpse blocks
- Screen shake and particle bursts on death snap

## The Pitch

You're a Tetris piece running through hell. When you die, your body becomes the floor. Completed rows clear. Every player — builder, griefer, janitor, pioneer — feeds the ecosystem. The level is alive, shaped by collective failure, breathing through cycles of creation and destruction.

Every time you die, you make the game different for someone else.
