# LOCOMOT.IO Project

## Training Sync Rule

**IMPORTANT**: When making changes to game mechanics in `index.html`, ALWAYS update the training script (`training/train_local.py`) to match. The AI brains are trained on simulated game rules - if they don't match, the AI will behave incorrectly.

Key areas to keep in sync:
- Pickup spawning rules (guns only from kills, no health)
- Powerups (SPEED, SHIELD, MAGNET)
- Leader Grave mechanics
- Collision rules and shield protection
- Team mode behavior
- Vision/input features

After updating training script, retrain both brains:
```bash
cd training
python3 train_local.py --mode both --episodes 10000
```

This runs 10000 episodes per brain (FFA + Team), ~70 min each.

## Pixel Art Editor Freemium

**TODO**: When auth is added, flip `isPremium()` in `pixel.html` from `true` to check actual login status.

Current state (line ~1120):
```js
const isPremium = () => true;  // Everyone gets unlimited layers for now
```

When ready to gate:
```js
const isPremium = () => localStorage.getItem('pixelArtPremium') === 'true';
// Or check actual auth status
```

Free tier: 2 layers max. Premium: unlimited.

## Debugging Notes

**Ivy doesn't watch toasts** - don't use toast messages for debugging. Use visible UI changes or console logging (check via remote logs) instead.

## Ice Skater Versioning

**ALWAYS tell Ivy the version when pushing ice_skater changes!**

Version is displayed on mobile at top-left corner. When making changes:
1. Bump `GAME_VERSION` in `ice_skater/index.html` (line ~314)
2. Bump `CACHE_NAME` in `sw.js`
3. Tell Ivy: "Pushed v199" (or whatever version)

This lets her confirm she's seeing the right version after refreshing.
