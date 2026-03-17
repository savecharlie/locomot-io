#!/bin/bash
# Build step_server.js from evaluate_node.js
# Takes the game code (everything before batch eval) and adds step protocol

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$SCRIPT_DIR/evaluate_node.js"
DST="$SCRIPT_DIR/step_server.js"

# Get line number of "function evaluateAgent"
CUT_LINE=$(grep -n 'function evaluateAgent(' "$SRC" | head -1 | cut -d: -f1)

if [ -z "$CUT_LINE" ]; then
    echo "ERROR: Could not find evaluateAgent in $SRC"
    exit 1
fi

# Take everything before evaluateAgent (stubs + game code + getInputs + SmallNN)
head -n $((CUT_LINE - 1)) "$SRC" > "$DST"

# Append step server protocol
cat >> "$DST" << 'STEP_SERVER_CODE'

// ══════════════════════════════════════════════════════════════
// ── STEP SERVER FOR RL TRAINING ──
// ══════════════════════════════════════════════════════════════

var _stepDt = 1 / 60;
var _stepCount = 0;
var _maxSteps = 30 * 60;  // 30s at 60fps
var _prevDistance = 0;
var _prevDeaths = 0;

function resetEnv(levelNum) {
    _simTime = 0;
    _stepCount = 0;

    currentLevel = levelNum || 2;
    initLevel(currentLevel);
    started = true;
    paused = false;
    tutorialStep = 3;
    tutorialComplete = true;
    jumpBuffered = false;
    levelComplete = false;

    // Explicitly reset death counter so it doesn't persist across episodes
    player.deathCount = 0;
    _prevDistance = 0;
    _prevDeaths = 0;

    return {
        state: Array.from(getInputs()),
        info: { level: currentLevel, distance: 0, deaths: 0 }
    };
}

function stepEnv(action) {
    _stepCount++;
    _simTime = _stepCount * _stepDt;

    // Apply action: 0=nothing, 1=jump, 2=rotate_cw, 3=rotate_ccw, 4=die
    if (!player.dead && !levelComplete && !gameWon) {
        if (action === 1) jumpBuffered = true;
        else if (action === 2) tryRotate(1);
        else if (action === 3) tryRotate(-1);
        else if (action === 4) die();
    }

    // Step physics
    update(_stepDt);

    // Instant respawn: skip death animation, get back to learning
    if (player.dead) {
        respawn();
    }

    // Reward
    var reward = 0;
    var curDist = player.distance || 0;
    var curDeaths = player.deathCount || 0;

    // +1 per new tile forward
    if (curDist > _prevDistance) reward += (curDist - _prevDistance);
    // Death penalty: quadratic (1st=-2, 2nd=-4, 3rd=-6)
    if (curDeaths > _prevDeaths) reward -= 2 * curDeaths;
    // Small airtime penalty
    if (!player.onGround && !player.dead) reward -= 0.02;
    // +20 level complete
    if (levelComplete || gameWon) reward += 20;
    // Time penalty — motivates faster progress
    reward -= 0.02;

    _prevDistance = curDist;
    _prevDeaths = curDeaths;

    // End after 5 deaths (quadratic penalty makes later deaths very costly)
    var done = gameWon || _stepCount >= _maxSteps || curDeaths >= 5;

    var state;
    try {
        state = !done ? Array.from(getInputs()) : new Array(72).fill(0);
    } catch (e) {
        state = new Array(72).fill(0);
    }

    return {
        state: state,
        reward: reward,
        done: done,
        info: {
            level: currentLevel,
            distance: player.bestDistance || 0,
            deaths: curDeaths,
            steps: _stepCount,
            completed: levelComplete || gameWon
        }
    };
}

// ── Line-based JSON protocol ──
var readline = require('readline');
var rl = readline.createInterface({ input: process.stdin });

rl.on('line', function(line) {
    var msg;
    try { msg = JSON.parse(line); } catch (e) {
        process.stdout.write(JSON.stringify({error: 'bad json'}) + '\n');
        return;
    }

    var result;
    if (msg.cmd === 'reset') result = resetEnv(msg.levelNum);
    else if (msg.cmd === 'step') result = stepEnv(msg.action || 0);
    else result = {error: 'unknown: ' + msg.cmd};

    process.stdout.write(JSON.stringify(result) + '\n');
});

process.stderr.write('Sediment step server ready\n');
STEP_SERVER_CODE

echo "Built $DST ($(wc -l < "$DST") lines)"
