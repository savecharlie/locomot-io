# LOCOMOT.IO Genome Ecosystem Plan

## Vision
A living AI ecosystem where behavioral genomes (learned from players) and RL genomes (learned from self-play) compete and evolve together. The best performers stay, the worst get replaced. Players literally shape the game's AI by playing.

---

## Phase 1: Foundation (Current State)
- [x] Behavioral data collection (DataRecorder)
- [x] Chunked upload to PartyKit server
- [x] Basic behavioral training (train_behavioral.py)
- [x] Genome registry in game client
- [x] IvyBot genome deployed

---

## Phase 2: Genome Pool System

### 2.1 Server-Side Genome Storage
```
New storage keys:
- genome_{id} - Full brain weights
- genome_manifest - List of all genomes with metadata
- genome_stats_{id} - Performance metrics per genome
```

### 2.2 Genome Manifest
```json
{
  "genomes": [
    {
      "id": "ivy",
      "type": "behavioral",  // behavioral, rl, cluster, hybrid
      "name": "IvyBot",
      "size_bytes": 975317,
      "created": "2026-01-03T15:24:00Z",
      "source": "player:ivy",  // or "self-play", "cluster:aggressive"
      "performance": {
        "games": 0,
        "avg_score": 0,
        "avg_survival": 0,
        "kill_ratio": 0
      }
    }
  ],
  "max_active": 15,
  "last_tournament": null
}
```

### 2.3 Server Endpoints (add to server.ts)
- `get_genome_manifest` - Return manifest with metadata
- `get_genome/{id}` - Return full genome weights
- `update_genome_stats` - Record game performance
- `submit_genome` - Upload new genome for consideration
- `run_tournament` - Trigger tournament evaluation

---

## Phase 3: RL Training Integration

### 3.1 Update train_local.py
- Export trained brains to genome format
- Submit to server as new genome candidates
- Track lineage (which genome was parent)

### 3.2 Self-Play Population
- Run continuous self-play episodes
- Generate new RL genomes periodically
- Submit best performers to genome pool

### 3.3 Hybrid Training
```python
# Fine-tune RL brain on human data
def create_hybrid(rl_genome_id, behavioral_data):
    # Load RL brain as starting point
    # Train few epochs on behavioral data
    # Submit as hybrid genome
```

---

## Phase 4: Behavior Clustering

### 4.1 Feature Extraction (cluster_behaviors.py)
```python
features_per_session = {
    "action_dist": [left%, straight%, right%],
    "turn_frequency": turns_per_second,
    "aggression": kill_attempts / frames,
    "survival": avg_survival_time,
    "collection_rate": pickups / frames,
    "risk_taking": avg_distance_to_threats
}
```

### 4.2 Clustering Pipeline
1. Fetch all behavioral sessions
2. Extract feature vectors
3. Run k-means (k=5 initially)
4. Name clusters: Aggressive, Survivor, Hunter, Farmer, Balanced
5. Train genome on each cluster

### 4.3 Cluster Genomes
- One genome per behavior archetype
- Updated weekly as new data arrives
- Named: "AggressiveV3", "SurvivorV2", etc.

---

## Phase 5: Tournament & Selection

### 5.1 Fitness Metrics
```python
fitness = (
    0.3 * normalized_score +
    0.3 * survival_time +
    0.2 * kill_ratio +
    0.2 * player_engagement  # Do players keep playing?
)
```

### 5.2 Tournament Process (weekly)
1. Simulate N games with all active genomes
2. Track performance of each genome
3. Rank by fitness
4. Replace bottom 2-3 with new candidates

### 5.3 Candidate Queue
- New RL genomes from self-play
- New behavioral genomes from players
- Updated cluster genomes
- Hybrid experiments

### 5.4 Diversity Enforcement
- Max 3 genomes of same type
- Must have at least 1 behavioral, 1 RL
- Named player genomes protected (IvyBot stays)

---

## Phase 6: Client-Side Loading

### 6.1 Progressive Loading
```javascript
// On game start
const manifest = await fetchManifest();
const sorted = manifest.genomes.sort(priorityOrder);
for (const genome of sorted) {
    await genomeRegistry.loadGenome(genome.id, genome.url);
    // Can start spawning this genome type now
}
```

### 6.2 Priority Order
1. Smallest size (fastest to load)
2. Highest performance rank
3. Oldest unloaded (variety)

### 6.3 Spawn Distribution
```javascript
function getSpawnGenome() {
    // 10% chance: named player genome (IvyBot)
    // 30% chance: random from top 5 performers
    // 60% chance: default brain (always loaded)
    return selectedGenome;
}
```

---

## Phase 7: UI & Feedback

### 7.1 Enemy Labels
- Show genome name: "IvyBot", "AggressiveV3"
- Optional genome icon/color

### 7.2 Death Screen
- "Killed by [genome name]"
- Show genome stats: "73% win rate, 47 games"

### 7.3 Contribution Feedback
- "Your gameplay improved the Hunter genome!"
- "IvyBot learned from 47 of your sessions"

### 7.4 Genome Leaderboard
- Top 10 performing genomes
- Player contribution counts
- Genome family trees (lineage)

---

## Implementation Order

### Week 1: Server Infrastructure
- [ ] Add genome storage endpoints
- [ ] Create manifest structure
- [ ] Add stats tracking

### Week 2: RL Integration
- [ ] Update train_local.py export
- [ ] Add genome submission
- [ ] Set up continuous self-play

### Week 3: Clustering
- [ ] Build cluster_behaviors.py
- [ ] Train cluster genomes
- [ ] Add to pool

### Week 4: Tournament
- [ ] Implement tournament runner
- [ ] Add selection logic
- [ ] Automate weekly runs

### Week 5: Client Polish
- [ ] Progressive loading
- [ ] UI improvements
- [ ] Contribution feedback

---

## Files to Create

```
training/
  train_behavioral.py  # [DONE] Behavioral cloning
  train_local.py       # [EXISTS] RL self-play
  cluster_behaviors.py # [NEW] Behavior clustering
  tournament.py        # [NEW] Tournament runner
  genome_manager.py    # [NEW] Genome pool management

server/
  server.ts            # [UPDATE] Add genome endpoints
```

---

## Success Metrics

- 10+ distinct genomes active
- Mix of RL, behavioral, cluster types
- Weekly tournament running
- Player engagement maintained/improved
- Visible genome diversity in gameplay
