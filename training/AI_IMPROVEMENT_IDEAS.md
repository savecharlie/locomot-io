# LOCOMOT.IO AI Improvement Ideas

Research compiled from deep dive into neural network enhancement techniques. Ranked by implementation difficulty and potential impact.

---

## QUICK WIN: Self-Play Training (IMPLEMENTED)

**File:** `self_play_training.ipynb`

Train the AI against copies of itself. Creates an infinite curriculum of progressively harder challenges.

**How it works:**
- Keep a pool of 10-15 past model versions
- Each episode, randomly select opponents from the pool
- Bias toward recent versions (harder opponents)
- Add current model to pool every ~100 episodes

**Expected Results:**
- More aggressive play
- Better at cutting off opponents
- Learns to use length advantage strategically

---

## MEDIUM: NEAT - Evolving Network Topology

**What:** Instead of fixed 48→64→64→3 architecture, let evolution discover the optimal structure.

**JavaScript Libraries:**
- [neataptic](https://github.com/wagenaartje/neataptic) - Most mature, good docs
- [neatjs](https://github.com/OptimusLime/neatjs) - Lighter weight

**Implementation:**
```javascript
const neataptic = require('neataptic');

// Create NEAT network that can evolve
const neat = new neataptic.Neat(48, 3, null, {
    mutation: [
        neataptic.methods.mutation.ADD_NODE,
        neataptic.methods.mutation.ADD_CONN,
        neataptic.methods.mutation.MOD_WEIGHT,
        neataptic.methods.mutation.MOD_BIAS
    ],
    popsize: 100,
    elitism: 10
});

// Fitness function based on game performance
function evaluateFitness(genome) {
    const agent = genome.toJSON();
    // Play game, return score
    return playGame(agent);
}

// Evolution loop
for (let gen = 0; gen < 500; gen++) {
    neat.sort();
    neat.mutate();
    // Evaluate all genomes...
}
```

**Pros:**
- Discovers novel architectures
- Can add/remove neurons automatically
- No gradient computation needed

**Cons:**
- Slow (population-based)
- Hard to parallelize in browser
- May need server-side training

---

## MEDIUM: Population-Based Self-Play

**What:** Maintain multiple AI personalities that train against each other.

**Implementation:**
```javascript
const population = [
    { model: 'aggressive.json', style: 'hunter' },
    { model: 'defensive.json', style: 'survivor' },
    { model: 'collector.json', style: 'greedy' },
    { model: 'balanced.json', style: 'adaptive' }
];

// Each game, randomly pair agents
// Winners reproduce, losers mutate
```

**Why it helps:**
- Prevents overfitting to single strategy
- Creates rock-paper-scissors dynamics
- More interesting gameplay variety

---

## ADVANCED: Hypernetworks

**What:** A neural network that generates the weights of another neural network.

**Concept:**
```
Context (game state, opponent style)
         ↓
   [Hypernetwork]
         ↓
   Generated Weights for Policy Network
         ↓
   [Policy Network with dynamic weights]
         ↓
   Action
```

**Why it matters:**
- AI can adapt its "brain" based on opponent
- Different weights for different situations
- Meta-learning capability

**Papers:**
- [HyperNetworks (Ha et al., 2016)](https://arxiv.org/abs/1609.09106)
- [HyperMAML](https://arxiv.org/abs/2003.00168)

**Difficulty:** High - requires careful architecture design

---

## ADVANCED: Self-Referential Weight Matrix (SRWM)

**What:** The network can modify its own weights during inference.

**Source:** [IDSIA/modern-srwm](https://github.com/IDSIA/modern-srwm)

**How it works:**
1. Network has "fast weights" that change during a single episode
2. Uses outer product updates (like Hopfield networks)
3. Delta rule: `W_new = W_old + lr * (target - output) * input`

**Why it's interesting:**
- True online learning
- Can adapt to new opponents in real-time
- Closest thing to "learning while playing"

**Difficulty:** Very high - cutting-edge research

---

## ADVANCED: Transformers.js + WebGPU

**What:** Replace simple feedforward network with transformer architecture.

**Benefits:**
- Attention over game history (remember past states)
- Better long-term planning
- Can model opponent behavior patterns

**Implementation:**
```javascript
import { pipeline } from '@xenova/transformers';

// Load custom-trained transformer
const agent = await pipeline('text-generation', 'locomot-agent', {
    device: 'webgpu'
});

// Encode game state as tokens
const tokens = encodeGameState(vision, history);
const action = await agent(tokens);
```

**Requirements:**
- Train custom transformer model (Python/PyTorch)
- Convert to ONNX format
- Load with Transformers.js
- Needs 70%+ of browsers (WebGPU support)

**Difficulty:** High - new toolchain

---

## ADVANCED: Graph Neural Networks for Multi-Agent

**What:** Model relationships between all agents as a graph.

**Concept:**
```
Nodes = Agents
Edges = Spatial relationships, threat levels
GNN = Message passing between nodes
Output = Action for each agent
```

**Why it helps:**
- Better coordination (for team mode)
- Explicit modeling of opponent positions
- Scales to many agents

**Papers:**
- [Graph Networks for Multi-Agent RL](https://arxiv.org/abs/1810.09202)
- [QMIX for multi-agent](https://arxiv.org/abs/1803.11485)

---

## Implementation Priority

1. **Self-Play** (done) - Run the Colab notebook
2. **Population Diversity** - Add 3-4 model variants
3. **NEAT** - Try neataptic for topology evolution
4. **Opponent Modeling** - Simple LSTM to predict opponent moves
5. **Transformers.js** - For major upgrade

---

## Quick Wins Without Retraining

These improve AI behavior without new training:

### 1. Tune the Rule Engine
Current rules in `index.html` can be tweaked:
```javascript
ruleEngine.addRule({
    condition: 'nearWall && wallDist < 0.3',
    action: 'bias_away_from_wall',
    priority: 10
});
```

### 2. Adjust Lookahead Depth
```javascript
const lookaheadScores = simulateLookahead(
    head.x, head.y, e.dir, e.segments,
    enemies.filter(oe => oe !== e), pickups, snake,
    8  // Increase from 5 to 8 for smarter planning
);
```

### 3. Add Memory (Simple)
```javascript
// Track last N opponent positions
const opponentHistory = new Map();

function updateHistory(enemyId, position) {
    if (!opponentHistory.has(enemyId)) {
        opponentHistory.set(enemyId, []);
    }
    const history = opponentHistory.get(enemyId);
    history.push(position);
    if (history.length > 10) history.shift();
}

// Use history to predict opponent direction
function predictOpponentMove(enemyId) {
    const history = opponentHistory.get(enemyId);
    if (!history || history.length < 2) return null;
    // Simple velocity prediction
    const dx = history[history.length-1].x - history[history.length-2].x;
    const dy = history[history.length-1].y - history[history.length-2].y;
    return { dx, dy };
}
```

---

## Resources

- [TensorNEAT](https://github.com/EMI-Group/tensorneat) - GPU-accelerated NEAT
- [Transformers.js](https://huggingface.co/docs/transformers.js) - ML in browser
- [SRWM](https://github.com/IDSIA/modern-srwm) - Self-modifying networks
- [SPIRAL](https://arxiv.org/abs/2506.24119) - Self-play framework
- [Neataptic](https://github.com/wagenaartje/neataptic) - JS neuroevolution
