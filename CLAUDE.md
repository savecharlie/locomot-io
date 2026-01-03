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
