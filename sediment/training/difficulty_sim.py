#!/usr/bin/env python3
"""Simulate the trained bot on each level to measure difficulty (deaths to complete)."""
import json, subprocess, sys, os

EVALUATOR = os.path.join(os.path.dirname(__file__), 'evaluate_node.js')
WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), 'best_weights.json')
RUNS_PER_LEVEL = 10
MAX_TIME = 120  # generous time limit
MAX_DEATHS = 100  # let the bot really try
LEVELS = range(1, 13)

with open(WEIGHTS_FILE) as f:
    weights_json = json.load(f)

# Flatten weights from layered format to flat array
flat = []
for i in range(3):
    w_key = f'w{i+1}'
    b_key = f'b{i+1}'
    for row in weights_json[w_key]:
        flat.extend(row)
    flat.extend(weights_json[b_key])

print(f"{'Level':>5} | {'Completed':>9} | {'Avg Deaths':>10} | {'Min':>4} | {'Max':>4} | {'Avg Dist':>8}")
print("-" * 60)

for lvl in LEVELS:
    all_weights = [flat] * RUNS_PER_LEVEL
    seeds = [lvl * 1000 + i for i in range(RUNS_PER_LEVEL)]

    payload = json.dumps({
        'weights': all_weights,
        'levelNum': lvl,
        'maxTime': MAX_TIME,
        'maxDeaths': MAX_DEATHS,
        'seeds': seeds,
        'noCorpses': False,
    })

    result = subprocess.run(
        ['node', EVALUATOR],
        input=payload, capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        print(f"  L{lvl}: ERROR - {result.stderr[:200]}", file=sys.stderr)
        continue

    results = json.loads(result.stdout)
    deaths = [r['deaths'] for r in results]
    completed = sum(1 for r in results if r['completed'])
    distances = [r['distance'] for r in results]

    avg_deaths = sum(deaths) / len(deaths)
    min_d = min(deaths)
    max_d = max(deaths)
    avg_dist = sum(distances) / len(distances)

    print(f"  L{lvl:>2} | {completed:>4}/{RUNS_PER_LEVEL:<4} | {avg_deaths:>10.1f} | {min_d:>4} | {max_d:>4} | {avg_dist:>8.0f}")

    sys.stdout.flush()
