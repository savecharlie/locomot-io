# LOCOMOT.IO Arena Training System

## Vision
Bots from the genome pool fight each other in simulated games. Winners get fine-tuned. Losers get culled. Dynasties emerge. The game evolves while you sleep.

---

## Phase 1: Arena Simulator ✅ (exists in train_local.py)

The simulation environment already exists:
- Grid-based world matching the game
- Snake movement, collision detection
- Pickup spawning, powerups
- Vision system (133 input features)
- Neural network decision making

**Needed changes:**
- [ ] Load multiple genome brains instead of training from scratch
- [ ] Track per-genome stats (kills, deaths, score, survival)
- [ ] Support mixed populations (behavioral + RL + bred)

---

## Phase 2: Tournament Runner

### 2.1 Match System
```python
class Tournament:
    def __init__(self, genome_ids):
        self.genomes = load_genomes_from_pool(genome_ids)
        self.stats = {gid: {'kills': 0, 'deaths': 0, 'score': 0, 'games': 0} for gid in genome_ids}

    def run_match(self, duration=60):
        # Spawn one bot per genome
        # Run simulation for duration seconds
        # Track who killed who
        # Return results

    def run_tournament(self, matches=100):
        # Run many matches
        # Aggregate stats
        # Rank genomes by fitness
```

### 2.2 Fitness Function
```python
def calculate_fitness(stats):
    return (
        0.3 * (stats['kills'] / max(stats['deaths'], 1)) +  # K/D ratio
        0.3 * (stats['score'] / stats['games']) +            # Avg score
        0.2 * (stats['survival_time'] / stats['games']) +    # Avg survival
        0.2 * stats['win_rate']                              # % games won
    )
```

### 2.3 Output
- Ranked leaderboard of genomes
- Per-matchup stats (IvyBot vs IvyBot Jr: 47-53)
- Identification of weak performers

---

## Phase 3: Fine-Tuning Winners

### 3.1 RL Improvement
Top performers get additional training:
```python
def finetune_genome(genome_id, episodes=1000):
    # Load genome weights as starting point
    # Run RL training (existing PPO code)
    # But opponents are OTHER pool genomes, not random
    # Save improved weights as new version
```

### 3.2 Versioning
- IvyBot → IvyBot_v2 (after fine-tuning)
- Or: IvyBot_trained (suffix indicates RL-improved)
- Track lineage: original → trained version

### 3.3 Opponent Sampling
During fine-tuning, opponents are sampled from pool:
- 50% from top performers (learn to beat the best)
- 30% from similar skill level (competitive matches)
- 20% random (maintain generalization)

---

## Phase 4: Natural Selection

### 4.1 Culling
After each tournament cycle:
```python
def cull_weakest(manifest, keep_top=10):
    # Sort by fitness
    # Mark bottom performers as inactive
    # But NEVER cull original player genomes (Gen 0)
    # Bred/trained versions can be culled
```

### 4.2 Breeding Top Performers
```python
def breed_winners(top_genomes):
    # Top 1 × Top 2 → Child
    # Top 1 self-breed → Mutant
    # Creates next generation
```

### 4.3 Population Control
- Max 15 active genomes
- At least 3 must be original player genomes (Gen 0)
- At least 2 must be RL-trained
- Rest can be bred/hybrid

---

## Phase 5: Automation

### 5.1 Training Daemon
```python
# arena_daemon.py
while True:
    # 1. Fetch latest genome pool
    genomes = fetch_active_genomes()

    # 2. Run tournament (100 matches)
    results = run_tournament(genomes, matches=100)

    # 3. Update stats on server
    upload_tournament_results(results)

    # 4. Fine-tune top 3 performers
    for genome_id in results.top(3):
        improved = finetune_genome(genome_id, episodes=500)
        submit_genome(improved)

    # 5. Breed top performers
    child = breed_genomes(results.top(1), results.top(2))
    submit_genome(child)

    # 6. Cull bottom performers
    cull_weakest(keep_top=10)

    # 7. Sleep until next cycle
    sleep(CYCLE_INTERVAL)  # e.g., 6 hours
```

### 5.2 Run Options
```bash
# One-shot tournament
python3 arena.py --tournament --matches 100

# Fine-tune a specific genome
python3 arena.py --finetune ivy --episodes 1000

# Full daemon mode (runs forever)
python3 arena.py --daemon --cycle-hours 6

# Overnight run (stop after N cycles)
python3 arena.py --daemon --cycles 4 --cycle-hours 2
```

---

## Phase 6: Reporting & UI

### 6.1 Server Endpoints
- `get_tournament_history` - Past tournament results
- `get_genome_matchups` - Head-to-head stats
- `get_dynasty_tree` - Family tree visualization data

### 6.2 Game UI (future)
- Show genome lineage on death screen
- "IvyBot Jr (Gen 1) - 73% win rate - 12 kills today"
- Dynasty leaderboard in menu

---

## Implementation Order

### Today: Core Arena
1. [x] ~~Modify train_local.py to load pool genomes~~ (arena.py loads directly from server)
2. [x] Create arena.py with Tournament class
3. [x] Run first tournament: IvyBot vs IvyBot Jr
4. [x] Track and display results

### Tomorrow: Fine-tuning
5. [x] Add finetune mode to arena.py
6. [ ] Test fine-tuning IvyBot Jr
7. [ ] Submit improved version to pool

### Next: Full Automation
8. [x] Add breeding integration
9. [x] Add culling logic (update_genome_status endpoint)
10. [x] Create daemon mode
11. [ ] Run overnight training session

### Polish: Reporting
12. [ ] Add tournament history endpoint
13. [ ] Add matchup stats endpoint
14. [ ] Update game UI to show lineage

---

## File Structure

```
training/
  train_local.py      # Existing RL training (modify)
  train_behavioral.py # Behavioral cloning (exists)
  breed_genome.py     # Breeding system (exists)
  upload_genome.py    # Chunked upload (exists)

  arena.py            # NEW: Tournament + fine-tuning
  arena_daemon.py     # NEW: Automated evolution daemon

  simulation/
    world.py          # Game world simulation
    snake.py          # Snake/train logic
    vision.py         # Neural network inputs
```

---

## Success Metrics

After one week of arena training:
- [ ] 3+ generations of bred genomes
- [ ] At least one fine-tuned version per top genome
- [ ] Clear skill hierarchy emerging
- [ ] Dynasties visible (IvyBot lineage dominates? Or new challenger?)
- [ ] Bots noticeably smarter than initial versions

---

## The Dream

You wake up. Check the arena stats.

"IvyBot Jr Gen 3 is now the top performer. It beat its parent IvyBot in 67% of matches overnight. A new challenger emerged: Ivy×Mystery_Gen2 trained against the whole pool and climbed from #8 to #2."

The game is evolving while you sleep. Your playstyle lives on through generations of bots, getting sharper with each cycle.

That's the dream. Let's build it. 🚂⚔️
