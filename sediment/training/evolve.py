"""Evolutionary training loop for Sediment NN agent.

Uses a genetic algorithm with tournament selection, uniform crossover,
Gaussian mutation, and elitism to evolve weights for a small feedforward NN.

Evaluation runs the ACTUAL game JS in Node.js for zero sim divergence.

Usage:
    python3 evolve.py                     # Train from scratch
    python3 evolve.py --resume best.npz   # Resume from checkpoint
    python3 evolve.py --level 5           # Start on level 5
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NODE_EVALUATOR = os.path.join(SCRIPT_DIR, 'evaluate_node.js')

# NN architecture constants (must match evaluate_node.js SmallNN)
LAYER_SIZES = [77, 48, 32, 5]


def num_params(sizes=None):
    if sizes is None:
        sizes = LAYER_SIZES
    total = 0
    for i in range(len(sizes) - 1):
        total += sizes[i] * sizes[i+1] + sizes[i+1]
    return total


def random_weights():
    return np.random.randn(num_params()).astype(np.float32) * 0.5


def migrate_weights(old_flat, old_sizes, new_sizes):
    """Migrate weights from old architecture to new, zero-padding new inputs."""
    new_n = num_params(new_sizes)
    new_flat = np.zeros(new_n, dtype=np.float32)

    # Layer 1: old is old_in x hidden, new is new_in x hidden
    old_in, new_in = old_sizes[0], new_sizes[0]
    hidden = old_sizes[1]  # same hidden size

    old_idx = 0
    new_idx = 0
    # Copy w1 row by row, zero-padding the new input columns
    for i in range(hidden):
        new_flat[new_idx:new_idx + old_in] = old_flat[old_idx:old_idx + old_in]
        # new_flat[new_idx + old_in : new_idx + new_in] stays 0
        old_idx += old_in
        new_idx += new_in

    # Copy everything after w1 (b1, w2, b2, w3, b3) — unchanged
    remaining = len(old_flat) - old_idx
    new_flat[new_idx:new_idx + remaining] = old_flat[old_idx:old_idx + remaining]

    return new_flat


def seeded_weights(action_bias):
    """Create weights biased toward a specific action (0-4).

    Sets the bias of one output neuron high so the agent starts
    preferring that action. Evolution can then refine WHEN to use it.
    """
    w = np.random.randn(num_params()).astype(np.float32) * 0.3  # smaller init
    # Output bias is the last 5 params (bias of final layer)
    if action_bias is not None and 0 <= action_bias < 5:
        w[-5 + action_bias] += 2.0  # strong preference
    return w


def heuristic_weights(noise_std=0.0):
    """Create NN weights encoding: walk by default, jump on spike, rotate when stuck.

    Hand-wires specific connections so the agent has intelligent behavior
    from generation 0. Evolution refines timing and edge cases.

    Input indices (from get_nn_inputs / getInputs):
      1: jumps_left/2    2: on_ground    7: stuck_timer/0.5
      30,31: spike/gap 1 tile   34,35: spike/gap 2 tiles   38,39: spike/gap 3 tiles

    Weight layout [5237]:
      w1: 48x72=3456  b1: 48  w2: 32x48=1536  b2: 32  w3: 5x32=160  b3: 5
    """
    n = num_params()
    w = np.zeros(n, dtype=np.float32)

    sizes = LAYER_SIZES  # [72, 48, 32, 5]

    # Offsets into flat array
    w1_off = 0                                    # 0
    b1_off = sizes[0] * sizes[1]                  # 3456
    w2_off = b1_off + sizes[1]                    # 3504
    b2_off = w2_off + sizes[1] * sizes[2]         # 5040
    w3_off = b2_off + sizes[2]                    # 5072
    b3_off = w3_off + sizes[2] * sizes[3]         # 5232

    def set_w1(neuron, inp, val):
        w[w1_off + neuron * sizes[0] + inp] = val

    def set_b1(neuron, val):
        w[b1_off + neuron] = val

    def set_w2(neuron, inp, val):
        w[w2_off + neuron * sizes[1] + inp] = val

    def set_b2(neuron, val):
        w[b2_off + neuron] = val

    def set_w3(output, inp, val):
        w[w3_off + output * sizes[2] + inp] = val

    def set_b3(output, val):
        w[b3_off + output] = val

    # === Layer 1: Feature detectors ===

    # N0 "danger_near": spikes/gaps in columns 1-4 (2-7 tiles ahead)
    # Jump EARLY — by the time gap is 1 tile away it's too late
    set_w1(0, 30, 1.5)   # spike 1 tile (emergency)
    set_w1(0, 31, 1.5)   # gap 1 tile (emergency)
    set_w1(0, 34, 3.0)   # spike 2 tiles (primary)
    set_w1(0, 35, 3.0)   # gap 2 tiles (primary)
    set_w1(0, 38, 3.0)   # spike 3 tiles (primary)
    set_w1(0, 39, 3.0)   # gap 3 tiles (primary)
    set_w1(0, 42, 2.0)   # spike 5 tiles (early warning)
    set_w1(0, 43, 2.0)   # gap 5 tiles (early warning)
    set_b1(0, -0.5)      # low threshold — sensitive to any danger

    # N1 "on_ground"
    set_w1(1, 2, 5.0)
    set_b1(1, -2.0)

    # N2 "stuck" — fire after ~0.2s stuck
    set_w1(2, 7, 5.0)
    set_b1(2, -1.0)

    # N3 "has_jumps"
    set_w1(3, 1, 5.0)
    set_b1(3, -1.0)

    # N4 "desperate" — fires when stuck for a long time (>0.4s)
    set_w1(4, 7, 5.0)    # stuck_timer input
    set_b1(4, -4.0)      # high threshold — only after prolonged stuck

    # === Layer 2: Combiners ===

    # N0 "jump_signal": danger required, grounded + jumps boost
    set_w2(0, 0, 6.0)    # danger_near (strong gate — MUST have danger)
    set_w2(0, 1, 1.0)    # on_ground (soft boost)
    set_w2(0, 3, 1.0)    # has_jumps (soft boost)
    set_b2(0, -6.0)      # needs danger to fire (0+3+1.5-6 = -1.5 → 0 when calm)

    # N1 "rotate_signal": stuck
    set_w2(1, 2, 5.0)
    set_b2(1, -1.5)

    # N2 "die_signal": desperate AND still stuck after rotating
    set_w2(2, 4, 5.0)    # desperate (long stuck)
    set_w2(2, 2, 2.0)    # still stuck (reinforces)
    set_b2(2, -4.0)      # needs both signals

    # === Layer 3: Output routing ===

    set_b3(0, 1.5)       # walk = default winner
    set_w3(1, 0, 4.0)    # jump from jump_signal
    set_b3(1, 0.0)
    set_w3(2, 1, 4.0)    # rotate_cw from rotate_signal
    set_b3(2, 0.0)
    set_b3(3, -5.0)      # never rotate_ccw
    set_b3(4, -2.0)      # reachable — dive when desperate
    set_w3(4, 2, 5.0)    # die from die_signal

    # Add noise for diversity
    if noise_std > 0:
        w += np.random.randn(n).astype(np.float32) * noise_std

    return w


def diverse_population(pop_size):
    """Create population with heuristic seeds + behavioral archetypes."""
    pop = []
    quarter = pop_size // 4

    # Heuristic: walk default, jump on spike, rotate when stuck
    # Multiple noise levels for diversity
    for i in range(quarter):
        if i < quarter // 3:
            pop.append(heuristic_weights(noise_std=0.1))   # close to heuristic
        elif i < 2 * quarter // 3:
            pop.append(heuristic_weights(noise_std=0.3))   # moderate variation
        else:
            pop.append(heuristic_weights(noise_std=0.5))   # significant variation

    # Walkers: prefer "nothing" — learn when to jump
    for _ in range(quarter):
        pop.append(seeded_weights(0))

    # Jumpers: prefer "jump" — baseline comparator
    for _ in range(quarter):
        pop.append(seeded_weights(1))

    # Wild: fully random — exploration
    while len(pop) < pop_size:
        pop.append(random_weights())

    np.random.shuffle(pop)
    return pop


def export_weights_json(weights):
    """Convert flat weight array to layered JSON for browser."""
    sizes = LAYER_SIZES
    idx = 0
    layers = {}
    for i in range(len(sizes) - 1):
        n_in, n_out = sizes[i], sizes[i+1]
        w = weights[idx:idx + n_in * n_out].reshape(n_out, n_in)
        idx += n_in * n_out
        b = weights[idx:idx + n_out]
        idx += n_out
        layers[f'w{i+1}'] = w.tolist()
        layers[f'b{i+1}'] = b.tolist()
    return json.dumps(layers)


class Evolver:
    """Genetic algorithm for evolving Sediment NN agents."""

    def __init__(
        self,
        pop_size=200,
        level_num=1,
        eval_time=30.0,
        elite_frac=0.05,
        tournament_size=3,
        mutation_rate=0.25,
        mutation_scale=0.5,
        crossover_rate=0.7,
        num_evals=3,
    ):
        self.pop_size = pop_size
        self.num_params = num_params()
        self.level_num = level_num
        self.eval_time = eval_time
        self.elite_frac = elite_frac
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.crossover_rate = crossover_rate
        self.num_evals = num_evals  # average over N runs per agent

        # Population
        self.population = diverse_population(pop_size)
        self.fitnesses = np.zeros(pop_size)
        self.generation = 0
        self.best_fitness = 0.0
        self.best_weights = None
        self.history = []  # (gen, best, mean, median, level)

        # Curriculum
        self.levels_beaten = set()

    def evaluate_population(self):
        """Evaluate all agents using Node.js (exact game JS, zero divergence)."""
        gen_seed_base = self.generation * 10000

        # Build batch: each agent evaluated num_evals times with different seeds
        all_weights = []
        all_seeds = []
        for i, w in enumerate(self.population):
            for e in range(self.num_evals):
                seed = gen_seed_base + e * 1000 + i
                all_weights.append(w.tolist())
                all_seeds.append(seed)

        payload = json.dumps({
            'weights': all_weights,
            'levelNum': self.level_num,
            'maxTime': self.eval_time,
            'seeds': all_seeds,
            'noCorpses': False,  # Corpses back on — agents learned to play properly
        })

        result = subprocess.run(
            ['node', NODE_EVALUATOR],
            input=payload, capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            print(f"Node eval error: {result.stderr[:500]}", file=sys.stderr)
            raise RuntimeError("Node evaluation failed")

        all_results = json.loads(result.stdout)

        # Average fitness per agent, track completions
        self.completions = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            agent_results = all_results[i * self.num_evals : (i + 1) * self.num_evals]
            self.fitnesses[i] = np.mean([r['fitness'] for r in agent_results])
            self.completions[i] = sum(1 for r in agent_results if r['completed'])

        # Track best
        best_idx = np.argmax(self.fitnesses)
        if self.fitnesses[best_idx] > self.best_fitness:
            self.best_fitness = self.fitnesses[best_idx]
            self.best_weights = self.population[best_idx].copy()

    def _tournament_select(self):
        """Select one individual via tournament selection."""
        indices = np.random.randint(0, self.pop_size, size=self.tournament_size)
        best = indices[np.argmax(self.fitnesses[indices])]
        return self.population[best]

    def _crossover(self, parent_a, parent_b):
        """Uniform crossover."""
        mask = np.random.random(self.num_params) < 0.5
        child = np.where(mask, parent_a, parent_b)
        return child

    def _mutate(self, weights):
        """Gaussian mutation with adaptive scale."""
        mask = np.random.random(self.num_params) < self.mutation_rate
        noise = np.random.randn(self.num_params).astype(np.float32) * self.mutation_scale
        weights[mask] += noise[mask]
        return weights

    def evolve_generation(self):
        """Create next generation from current population."""
        sorted_indices = np.argsort(self.fitnesses)[::-1]
        elite_count = max(2, int(self.pop_size * self.elite_frac))

        new_pop = []

        # Elitism: keep top performers unchanged
        for i in range(elite_count):
            new_pop.append(self.population[sorted_indices[i]].copy())

        # Fill rest with tournament selection + crossover + mutation
        while len(new_pop) < self.pop_size:
            parent_a = self._tournament_select()
            if np.random.random() < self.crossover_rate:
                parent_b = self._tournament_select()
                child = self._crossover(parent_a, parent_b)
            else:
                child = parent_a.copy()
            child = self._mutate(child)
            new_pop.append(child)

        self.population = new_pop
        self.generation += 1

    def check_curriculum(self):
        """Advance to next level only if agents actually complete it."""
        # Count how many agents completed the level in ALL their eval runs
        fully_completed = np.sum(self.completions == self.num_evals)
        any_completed = np.sum(self.completions > 0)

        # Advance only if: 5+ agents complete in every run AND 20%+ complete at least once
        if fully_completed >= 5 and any_completed >= self.pop_size * 0.2:
            self.levels_beaten.add(self.level_num)
            self.level_num += 1
            return True
        return False

    def save_checkpoint(self, path):
        """Save full training state."""
        np.savez_compressed(
            path,
            population=np.array(self.population),
            fitnesses=self.fitnesses,
            best_weights=self.best_weights if self.best_weights is not None else np.zeros(1),
            best_fitness=self.best_fitness,
            generation=self.generation,
            level_num=self.level_num,
            history=np.array(self.history) if self.history else np.zeros((0, 5)),
        )

    def load_checkpoint(self, path):
        """Load training state from checkpoint, auto-migrating old architectures."""
        data = np.load(path, allow_pickle=True)
        self.population = list(data['population'])
        self.fitnesses = data['fitnesses']
        self.best_weights = data['best_weights']
        self.best_fitness = float(data['best_fitness'])
        self.generation = int(data['generation'])
        self.level_num = int(data['level_num'])
        if 'history' in data and data['history'].shape[0] > 0:
            self.history = data['history'].tolist()

        # Auto-migrate if old architecture (72 inputs → 76 inputs)
        expected = num_params()
        if len(self.population[0]) != expected:
            old_sizes = [72, 48, 32, 5]
            print(f"  Migrating {len(self.population)} agents: {old_sizes} → {LAYER_SIZES}")
            self.population = [migrate_weights(w, old_sizes, LAYER_SIZES) for w in self.population]
            if self.best_weights is not None and len(self.best_weights) > 1:
                self.best_weights = migrate_weights(self.best_weights, old_sizes, LAYER_SIZES)

    def export_best(self, path):
        """Export best weights as JSON for browser integration."""
        if self.best_weights is None:
            print("No best weights to export!")
            return
        with open(path, 'w') as f:
            f.write(export_weights_json(self.best_weights))
        print(f"Exported best weights to {path}")

    def train(self, generations=500, checkpoint_every=25, checkpoint_dir='.'):
        """Main training loop."""
        os.makedirs(checkpoint_dir, exist_ok=True)

        print(f"Evolving {self.pop_size} agents × {self.num_params} params (Node.js eval)")
        print(f"Starting level: {self.level_num} | Eval time: {self.eval_time}s")
        print(f"Evaluations per agent: {self.num_evals}")
        print("-" * 70)

        for gen in range(generations):
            t0 = time.time()

            self.evaluate_population()

            best = np.max(self.fitnesses)
            mean = np.mean(self.fitnesses)
            median = np.median(self.fitnesses)
            self.history.append([self.generation, best, mean, median, self.level_num])

            elapsed = time.time() - t0

            # Progress bar
            TILE = 12; SEGMENT_W = 40
            segs = min(7, 3 + int((self.level_num - 1) * 0.4))
            goal_x = (segs * SEGMENT_W - 4) * TILE
            pct = min(100, best / goal_x * 100)
            bar_len = 30
            filled = int(bar_len * pct / 100)
            bar = '█' * filled + '░' * (bar_len - filled)

            completers = int(np.sum(self.completions > 0))
            print(
                f"Gen {self.generation:4d} | L{self.level_num} |{bar}| "
                f"best:{best:7.0f} mean:{mean:7.0f} med:{median:7.0f} | "
                f"fin:{completers}/{self.pop_size} | {elapsed:.1f}s"
            )
            sys.stdout.flush()

            # Curriculum
            advanced = self.check_curriculum()
            if advanced:
                print(f"  >>> Advanced to level {self.level_num}!")

            # Always export latest best weights for live browser preview
            self.export_best(os.path.join(checkpoint_dir, 'best_weights.json'))

            # Full checkpoint (population + state) less frequently
            if (self.generation + 1) % checkpoint_every == 0 or advanced:
                cp_path = os.path.join(checkpoint_dir, f'checkpoint_gen{self.generation}.npz')
                self.save_checkpoint(cp_path)
                self.save_checkpoint(os.path.join(checkpoint_dir, 'best.npz'))

            self.evolve_generation()

        # Final save
        self.save_checkpoint(os.path.join(checkpoint_dir, 'best.npz'))
        self.export_best(os.path.join(checkpoint_dir, 'best_weights.json'))
        print(f"\nTraining complete. Best fitness: {self.best_fitness:.0f}")
        print(f"Levels beaten: {sorted(self.levels_beaten) if self.levels_beaten else 'none yet'}")


def main():
    parser = argparse.ArgumentParser(description='Evolve Sediment NN agent')
    parser.add_argument('--pop', type=int, default=200, help='Population size')
    parser.add_argument('--level', type=int, default=1, help='Starting level')
    parser.add_argument('--gens', type=int, default=500, help='Number of generations')
    parser.add_argument('--eval-time', type=float, default=30.0, help='Eval time per agent (seconds)')
    parser.add_argument('--num-evals', type=int, default=3, help='Evaluations per agent (averaged)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint file')
    parser.add_argument('--checkpoint-dir', type=str, default='.', help='Directory for checkpoints')
    parser.add_argument('--mutation-rate', type=float, default=0.15, help='Mutation probability per weight')
    parser.add_argument('--mutation-scale', type=float, default=0.3, help='Mutation noise std dev')
    parser.add_argument('--beam-seeds', type=str, default=None, help='Beam search seed weights file (.npz)')
    args = parser.parse_args()

    evolver = Evolver(
        pop_size=args.pop,
        level_num=args.level,
        eval_time=args.eval_time,
        num_evals=args.num_evals,
        mutation_rate=args.mutation_rate,
        mutation_scale=args.mutation_scale,
    )

    if args.resume:
        print(f"Resuming from {args.resume}")
        evolver.load_checkpoint(args.resume)

    # Seed population with beam search expert brains
    if args.beam_seeds and os.path.exists(args.beam_seeds):
        data = np.load(args.beam_seeds)
        seed_weights = data['weights']
        n_seeds = min(len(seed_weights), args.pop // 4)  # max 25% of population
        for i in range(n_seeds):
            evolver.population[i] = seed_weights[i].astype(np.float32)
        print(f"Seeded {n_seeds} expert brains from {args.beam_seeds}")
    # --level always wins (override checkpoint)
    evolver.level_num = args.level

    evolver.train(
        generations=args.gens,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == '__main__':
    main()
