"""Rolling horizon search + NN imitation learning for Sediment.

Finds expert playthroughs by looking ahead at each decision point,
then fits NNs to imitate those playthroughs. Output is used to
seed the genetic population in evolve.py.

Usage:
    python3 beam_search.py --level 2 --seeds 20
    python3 beam_search.py --level 2 --seeds 10 --horizon 20
"""

import argparse
import copy
import os
import random
import sys
import time

import numpy as np

from sediment_sim import SedimentSim, SmallNN, TILE, LEVEL_H_TILES

# Actions to search (skip rotate_ccw and die to reduce branching)
SEARCH_ACTIONS = [0, 1, 2]  # nothing, jump, rotate_cw
ALL_ACTIONS = 5
DECISION_FRAMES = 7  # frames per decision (~0.12s at 60fps)
DT = 1 / 60


def count_ground_tiles(sim, start_tile_x):
    """Count how many unique tiles were crossed while grounded since start_tile_x."""
    p = sim.player
    if p['dead']:
        return 0
    current_tile_x = int(p['x'] / TILE)
    if p['on_ground'] and current_tile_x > start_tile_x:
        return current_tile_x - start_tile_x
    return 0


def simulate_horizon(sim, action_sequence, dt=DT):
    """Run a sim forward through a sequence of actions.
    Each action is held for DECISION_FRAMES frames.
    Returns ground tiles gained during this horizon."""
    p = sim.player
    start_x = int(p['x'] / TILE)
    ground_tiles = 0
    max_ground_x = start_x

    for action in action_sequence:
        for _ in range(DECISION_FRAMES):
            if p['dead']:
                return ground_tiles
            sim.step(dt, action)
            if p['on_ground'] and not p['dead']:
                cur_x = int(p['x'] / TILE)
                if cur_x > max_ground_x:
                    ground_tiles += cur_x - max_ground_x
                    max_ground_x = cur_x
    return ground_tiles


def rolling_horizon_search(level_num, seed, horizon=15, max_time=30.0):
    """Find an expert playthrough using rolling horizon lookahead.

    At each decision point, tries each action with `horizon` steps of
    lookahead, picks the action that gains the most ground tiles.

    Returns:
        demos: list of (input_vector, action) pairs
        ground_dist: total ground tiles crossed
        total_dist: best distance reached (for comparison)
    """
    # Set up deterministic sim
    random.seed(seed)
    np.random.seed(seed)
    sim = SedimentSim(level_num, corpses=False)

    demos = []
    total_ground = 0
    max_ground_x = 0
    decisions = 0
    deaths = 0

    while sim.time < max_time:
        p = sim.player

        # If dead, just step through respawn
        if p['dead']:
            sim.step(DT, action=0)
            continue

        # Get current state for the demo
        inputs = sim.get_nn_inputs()

        # Save sim state for branching
        rng_state = random.getstate()
        np_rng_state = np.random.get_state()
        saved_sim = copy.deepcopy(sim)

        # Try each action with lookahead
        best_action = 0
        best_score = -float('inf')

        for action in SEARCH_ACTIONS:
            # Restore sim to saved state
            trial_sim = copy.deepcopy(saved_sim)
            random.setstate(rng_state)
            np.random.set_state(np_rng_state)

            # Execute the candidate action for one decision step
            trial_p = trial_sim.player
            trial_ground = 0
            trial_total = 0
            trial_max_gx = int(trial_p['x'] / TILE)

            for _ in range(DECISION_FRAMES):
                if trial_p['dead']:
                    break
                trial_sim.step(DT, action)
                if trial_p['on_ground'] and not trial_p['dead']:
                    gx = int(trial_p['x'] / TILE)
                    if gx > trial_max_gx:
                        trial_ground += gx - trial_max_gx
                        trial_max_gx = gx
            trial_total += DECISION_FRAMES

            # Then run horizon-1 more steps with reactive heuristic:
            # jump if spike ahead in columns 1-3, otherwise walk
            for _ in range(horizon - 1):
                if trial_p['dead']:
                    break
                # Simple spike-dodge heuristic for lookahead
                heur_action = 0  # walk by default
                if not trial_p['dead']:
                    inp = trial_sim.get_nn_inputs()
                    # Terrain lookahead starts at index 28
                    # Each col: [ground_h, ceiling_h, has_spike, has_gap]
                    # Col 0 (1 tile ahead): spike at idx 30, gap at 31
                    # Col 1 (2 tiles ahead): spike at idx 34, gap at 35
                    # Col 2 (3 tiles ahead): spike at idx 38, gap at 39
                    for col in range(3):
                        spike_idx = 28 + col * 4 + 2
                        gap_idx = 28 + col * 4 + 3
                        if (inp[spike_idx] > 0.5 or inp[gap_idx] > 0.5) and trial_p['on_ground']:
                            heur_action = 1  # jump!
                            break

                for _ in range(DECISION_FRAMES):
                    if trial_p['dead']:
                        break
                    trial_sim.step(DT, heur_action)
                    if trial_p['on_ground'] and not trial_p['dead']:
                        gx = int(trial_p['x'] / TILE)
                        if gx > trial_max_gx:
                            trial_ground += gx - trial_max_gx
                            trial_max_gx = gx
                trial_total += DECISION_FRAMES

            # Score: distance reached (survival matters!) + ground bonus
            if trial_p['dead']:
                score = trial_p['best_distance'] * 0.3  # harsh death penalty
            else:
                # Reward: raw distance + ground tile bonus
                score = trial_p['x'] + trial_ground * TILE * 2

            if score > best_score:
                best_score = score
                best_action = action

        # Record the demo pair
        demos.append((inputs, best_action))
        decisions += 1

        # Restore and execute the chosen action on the real sim
        random.setstate(rng_state)
        np.random.set_state(np_rng_state)
        # sim is already at the saved state (we deepcopied for trials)
        # We need to use saved_sim since sim might have been modified... no.
        # Actually sim was NOT modified — we only deepcopied it. sim is still at decision point.
        # Wait, we deepcopied saved_sim FROM sim. sim itself is untouched.

        was_dead = p['dead']
        for _ in range(DECISION_FRAMES):
            if p['dead']:
                if not was_dead:
                    deaths += 1
                    was_dead = True
                break
            sim.step(DT, best_action)

        # Track ground distance
        if not p['dead'] and p['on_ground']:
            cur_x = int(p['x'] / TILE)
            if cur_x > max_ground_x:
                total_ground += cur_x - max_ground_x
                max_ground_x = cur_x

    return demos, total_ground, sim.player['best_distance'], deaths


def fit_nn_to_demos(all_demos, epochs=2000, lr=0.001):
    """Train a SmallNN to predict expert actions from states.

    Uses simple numpy backprop through the 3-layer ReLU network.
    Returns flat weight array [5237].
    """
    if not all_demos:
        return SmallNN.random_weights()

    # Prepare data
    X = np.array([d[0] for d in all_demos], dtype=np.float32)  # [N, 72]
    y = np.array([d[1] for d in all_demos], dtype=np.int32)    # [N]
    N = len(X)

    # One-hot targets
    Y_onehot = np.zeros((N, ALL_ACTIONS), dtype=np.float32)
    Y_onehot[np.arange(N), y] = 1.0

    # Initialize weights (Xavier init for better convergence)
    sizes = SmallNN.LAYER_SIZES  # [72, 48, 32, 5]
    w1 = np.random.randn(sizes[1], sizes[0]).astype(np.float32) * np.sqrt(2.0 / sizes[0])
    b1 = np.zeros(sizes[1], dtype=np.float32)
    w2 = np.random.randn(sizes[2], sizes[1]).astype(np.float32) * np.sqrt(2.0 / sizes[1])
    b2 = np.zeros(sizes[2], dtype=np.float32)
    w3 = np.random.randn(sizes[3], sizes[2]).astype(np.float32) * np.sqrt(2.0 / sizes[2])
    b3 = np.zeros(sizes[3], dtype=np.float32)

    # Adam optimizer state
    params = [w1, b1, w2, b2, w3, b3]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    best_loss = float('inf')
    best_weights = None

    for epoch in range(epochs):
        # Forward pass (batch)
        z1 = X @ w1.T + b1          # [N, 48]
        h1 = np.maximum(0, z1)       # ReLU
        z2 = h1 @ w2.T + b2          # [N, 32]
        h2 = np.maximum(0, z2)       # ReLU
        logits = h2 @ w3.T + b3      # [N, 5]

        # Softmax
        logits_max = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits_max)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        # Cross-entropy loss
        loss = -np.mean(np.sum(Y_onehot * np.log(probs + 1e-10), axis=1))

        # Track best
        if loss < best_loss:
            best_loss = loss
            best_weights = _pack_weights(w1, b1, w2, b2, w3, b3)

        # Backward pass
        dlogits = (probs - Y_onehot) / N     # [N, 5]

        dw3 = dlogits.T @ h2                  # [5, 32]
        db3 = dlogits.sum(axis=0)             # [5]
        dh2 = dlogits @ w3                     # [N, 32]

        dz2 = dh2 * (z2 > 0)                  # ReLU grad
        dw2 = dz2.T @ h1                      # [32, 48]
        db2 = dz2.sum(axis=0)                 # [32]
        dh1 = dz2 @ w2                         # [N, 48]

        dz1 = dh1 * (z1 > 0)                  # ReLU grad
        dw1 = dz1.T @ X                       # [48, 72]
        db1 = dz1.sum(axis=0)                 # [48]

        # Adam update
        grads = [dw1, db1, dw2, db2, dw3, db3]
        t = epoch + 1
        for i, (p, g) in enumerate(zip(params, grads)):
            m[i] = beta1 * m[i] + (1 - beta1) * g
            v[i] = beta2 * v[i] + (1 - beta2) * g ** 2
            m_hat = m[i] / (1 - beta1 ** t)
            v_hat = v[i] / (1 - beta2 ** t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)

        if epoch % 200 == 0:
            acc = np.mean(np.argmax(logits, axis=1) == y)
            print(f"  Epoch {epoch:4d} | loss: {loss:.4f} | acc: {acc:.1%}")

    final_acc = np.mean(np.argmax(logits, axis=1) == y)
    print(f"  Final    | loss: {best_loss:.4f} | acc: {final_acc:.1%}")
    return best_weights


def _pack_weights(w1, b1, w2, b2, w3, b3):
    """Pack weight matrices back into flat [5237] array."""
    return np.concatenate([
        w1.flatten(), b1.flatten(),
        w2.flatten(), b2.flatten(),
        w3.flatten(), b3.flatten(),
    ]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description='Beam search + imitation learning for Sediment')
    parser.add_argument('--level', type=int, default=2, help='Level to search')
    parser.add_argument('--seeds', type=int, default=20, help='Number of random seeds to try')
    parser.add_argument('--horizon', type=int, default=15, help='Lookahead steps per decision')
    parser.add_argument('--max-time', type=float, default=30.0, help='Max eval time per search')
    parser.add_argument('--output', type=str, default='beam_seeds.npz', help='Output file')
    args = parser.parse_args()

    print(f"Rolling Horizon Search: L{args.level} | {args.seeds} seeds | horizon={args.horizon}")
    print("=" * 60)

    all_demos = []
    results = []

    for i in range(args.seeds):
        seed = 42 + i * 137  # spread seeds
        t0 = time.time()

        demos, ground_dist, best_dist, deaths = rolling_horizon_search(
            args.level, seed, horizon=args.horizon, max_time=args.max_time
        )

        elapsed = time.time() - t0
        results.append((seed, ground_dist, best_dist, deaths, len(demos)))
        all_demos.extend(demos)

        bar_len = 30
        pct = min(100, ground_dist / 200 * 100)
        filled = int(bar_len * pct / 100)
        bar = '#' * filled + '.' * (bar_len - filled)

        print(
            f"  Seed {seed:5d} |{bar}| "
            f"ground:{ground_dist:4d} dist:{best_dist:6.0f}px "
            f"deaths:{deaths:2d} decisions:{len(demos):3d} | {elapsed:.1f}s"
        )
        sys.stdout.flush()

    # Summary
    ground_dists = [r[1] for r in results]
    print(f"\nSearch complete: {len(all_demos)} demo pairs from {args.seeds} seeds")
    print(f"Ground distance: best={max(ground_dists)} mean={np.mean(ground_dists):.0f} med={np.median(ground_dists):.0f}")

    # Fit NNs to demos
    print(f"\nFitting NN to {len(all_demos)} demo pairs...")
    fitted_weights = []

    # Fit multiple NNs with different initializations for population diversity
    num_fits = min(20, max(5, args.seeds))
    for i in range(num_fits):
        print(f"\nFit {i+1}/{num_fits}:")
        np.random.seed(1000 + i)
        w = fit_nn_to_demos(all_demos, epochs=2000, lr=0.001)
        fitted_weights.append(w)

    # Also fit on subsets for more diversity
    if len(results) > 5:
        # Fit on best half of demos
        sorted_results = sorted(results, key=lambda r: r[1], reverse=True)
        best_seeds = {r[0] for r in sorted_results[:len(sorted_results)//2]}
        best_demos = []
        seed_idx = 0
        for r in results:
            n = r[4]  # num demos for this seed
            if r[0] in best_seeds:
                best_demos.extend(all_demos[seed_idx:seed_idx+n])
            seed_idx += n

        if best_demos:
            print(f"\nFitting NN to top {len(best_demos)} demos (best seeds)...")
            np.random.seed(9999)
            w = fit_nn_to_demos(best_demos, epochs=2000, lr=0.001)
            fitted_weights.append(w)

    # Save
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    np.savez_compressed(
        output_path,
        weights=np.array(fitted_weights),
        level=args.level,
    )
    print(f"\nSaved {len(fitted_weights)} seed brains to {output_path}")
    print("Use: python3 evolve.py --level 2 --beam-seeds beam_seeds.npz")


if __name__ == '__main__':
    main()
