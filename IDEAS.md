# LOCOMOT.IO Game Ideas

## Priority 1: Implement First

### "All Same Gun" System
- **Concept**: Picking up a gun converts ALL your cars to that type
- **Why it's good**: Simple, strategic, every pickup is a meaningful decision
- **Details**:
  - Pick up SHOTGUN → all cars become shotguns
  - Creates "hunting for the right gun" gameplay
  - No more optimal mixed loadout - commit to a playstyle
  - Option: Hold button to pick up vs auto-grab (prevent accidents)

### Gun Type Affects Speed
- **Concept**: Your current gun type determines train speed
- **Examples**:
  - MACHINEGUN: Fastest (light, rapid fire)
  - SHOTGUN: Medium-fast (balanced)
  - CANNON: Slowest (heavy artillery)
  - PULSE: Medium (energy weapons)
- **Why it's good**: Adds another layer to gun choice - speed vs firepower tradeoff

### CTF Mode with Caboose Flag
- **Concept**: Flag is a special caboose car that attaches to your tail
- **Mechanics**:
  - Drive over enemy flag → attaches as your last car
  - Flag caboose has HP, can be shot off
  - If you die, flag drops where your caboose was
  - Your flag must be home to score
  - First to 3 captures wins
- **Map**: Symmetrical with bases, choke points, cover

### Screen Shake + Game Juice
- Screen shake on kills/hits (directional)
- Kill streak announcements ("DOUBLE KILL!")
- Near-miss visual feedback
- Slow-mo on kills (0.2 sec)
- Red screen edges when low HP
- Satisfying pickup sounds/effects

---

## Priority 2: Build Next

### Diep.io-Style Class System
**Engine Classes (Level 10):**
- SPEED ENGINE: +30% movement, -20% HP
- TANK ENGINE: +50% HP, -20% speed
- FIRE ENGINE: +20% fire rate, -15% HP

**Specializations (Level 25):**
- From Speed: SCOUT (invisible on minimap), DASH (boost ability)
- From Tank: FORTRESS (damage reflection), RAM (collision damage)
- From Fire: BURST (all guns fire at once), SNIPER (long range)

**Stat Points (33 total, like Diep.io):**
- Max HP, HP Regen, Movement Speed
- Reload Speed, Bullet Damage, Bullet Penetration
- Bullet Speed, Body Damage

### Kill Streak System
- Track consecutive kills
- Announcements: DOUBLE KILL, TRIPLE KILL, MEGA KILL, etc.
- Combo multiplier for score
- Visual effects at milestones (5 streak = screen flash)

### "Train Your Own AI" Feature
- Record your gameplay (already collecting data!)
- Train personal AI in browser (TensorFlow.js)
- Spawn YOUR AI as opponent
- Share AI with friends via link
- Leaderboard of AI scores

### Daily Challenges
- "Kill 5 enemies with Cannon"
- "Survive 3 minutes"
- "Collect 20 pickups"
- Rewards: Coins, cosmetics
- Streak bonuses for consecutive days

---

## Priority 3: Future Ideas

### New Powerups
- GHOST: Phase through other trains temporarily
- REVERSE: Your head becomes tail (surprise enemies)
- FREEZE: Slow all nearby enemies
- MIRROR: Reflect bullets back
- CLOAK: Invisible on minimap
- RAGE: Double fire rate, double damage taken

### New Gun Types
- LASER: Continuous beam, requires tracking
- MORTAR: Lobs projectiles in arc
- FLAK: Shoots down enemy bullets
- GRAPPLE: Hook and pull toward targets
- MINE LAYER: Drop mines behind you
- TESLA: Chain lightning to nearby enemies
- REPAIR: Heals adjacent friendly segments

### Environmental Features
- Bridges (destructible)
- Tunnels (dark, good for ambush)
- Turntables (spin 180)
- Destructible cover
- Dynamic weather (fog, rain)

### Train-Specific Mechanics
- REVERSE GEAR: Go backward, tail becomes head
- WHISTLE: Stun nearby enemies (cooldown)
- DECOUPLE: Drop last car as mine
- Track marks that boost your speed

### Other Game Modes
- **King of the Hill**: Control zone to score
- **Elimination**: No respawns, last team wins
- **Escort**: Protect VIP train across map
- **Gun Game**: Kill to upgrade gun, cycle through all
- **Infection**: Kills convert enemies to your team

### Cosmetics & Progression
- Train skins (steampunk, modern, cartoon)
- Trail effects (smoke, sparkles, fire)
- Death effects
- Engine visual upgrades at higher levels
- Unlockable via gameplay coins

### Quality of Life
- Tutorial for new players
- Training arena (safe practice)
- Tips on death screen
- Colorblind modes
- Reduced effects option

---

## Design Principles

1. **No Pay-to-Win**: All gameplay content available to everyone
2. **Simple to Learn**: Core loop is move + shoot + collect
3. **Deep to Master**: Gun choice, positioning, class builds
4. **Fair Deaths**: Player should understand why they died
5. **Satisfying Kills**: Every kill feels earned

## Unique Selling Points

- "The .io game where AI learns from YOU"
- "Snake + Guns + Trains"
- Gun type commitment (not mixed loadouts)
- Class system with meaningful builds

---

## Technical Notes

- PartyKit for multiplayer (already set up)
- Neural network training pipeline exists
- Player data collection working
- Metrics tracking on server

## References

- [Slither.io](https://en.wikipedia.org/wiki/Slither.io) - Death drops pellets, speed boost costs length
- [Diep.io](https://diepio.fandom.com/wiki/Tiers) - Class tree, stat points, leveling
- [Game Juice](https://www.bloodmooninteractive.com/articles/juice.html) - Screen shake, feedback
