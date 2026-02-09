/**
 * Sediment NN Agent — Browser inference for evolved neural network.
 *
 * Loads trained weights (JSON) and runs a 76→48→32→5 feedforward NN
 * to play the game autonomously. Toggle with 'N' key.
 *
 * Expects these game globals: player, level, corpseGrid, buzzsaws,
 * movingPlatforms, goalX, levelH, TILE, SEGMENT_W, segments, shapes, currentLevel
 */

const nnAgent = (() => {
    let weights = null;
    let active = false;
    let loaded = false;
    let progressCheckX = 0;
    let progressTimer = 0;
    // Death memory
    let dmDeaths = 0;
    let dmLastDeathX = 0;
    let dmBestX = 0;
    let dmTimeSinceDeath = 999;
    let dmPrevDeathCount = 0;
    // Loop detector
    let actionHist = [];
    let loopBreakCount = 0;

    // ── Matrix math ──

    function matmul(W, x, b) {
        // W: [rows][cols], x: [cols], b: [rows] → out: [rows]
        const rows = W.length;
        const cols = W[0].length;
        const out = new Float32Array(rows);
        for (let i = 0; i < rows; i++) {
            let sum = b[i];
            const row = W[i];
            for (let j = 0; j < cols; j++) {
                sum += row[j] * x[j];
            }
            out[i] = sum;
        }
        return out;
    }

    function relu(x) {
        const out = new Float32Array(x.length);
        for (let i = 0; i < x.length; i++) {
            out[i] = x[i] > 0 ? x[i] : 0;
        }
        return out;
    }

    function argmax(x) {
        let best = 0;
        for (let i = 1; i < x.length; i++) {
            if (x[i] > x[best]) best = i;
        }
        return best;
    }

    // ── Forward pass ──

    function forward(inputs) {
        if (!weights) return 0;
        let h = relu(matmul(weights.w1, inputs, weights.b1));
        h = relu(matmul(weights.w2, h, weights.b2));
        const logits = matmul(weights.w3, h, weights.b3);
        return argmax(logits);
    }

    // ── Input extraction (must match Python get_nn_inputs exactly) ──

    function getInputs() {
        const p = player;
        const inputs = new Float32Array(72);
        let idx = 0;
        const lW = segments.length * SEGMENT_W;
        const lH = levelH || level.length;

        // A. Player state (9)
        inputs[idx++] = p.vy / 400.0;
        inputs[idx++] = (p.jumpsLeft || 0) / 2.0;
        inputs[idx++] = p.onGround ? 1.0 : 0.0;
        inputs[idx++] = ((p.onWall || 0) + 1) / 2.0;
        inputs[idx++] = p.y / (lH * TILE);
        inputs[idx++] = (p.boostTimer || 0) > 0 ? 1.0 : 0.0;
        inputs[idx++] = (p.quakeTimer || p.crunchTimer || 0) > 0 ? 1.0 : 0.0;
        inputs[idx++] = Math.min(1.0, (p.stuckTimer || 0) / 0.5);
        inputs[idx++] = Math.min(1.0, (p.distance || 0) / Math.max(1, goalX));

        // B. Shape geometry 4x4 (16)
        const cells = getCells(p.shape, p.rotation);
        let minC = 99, minR = 99;
        for (const [c, r] of cells) {
            if (c < minC) minC = c;
            if (r < minR) minR = r;
        }
        const grid = new Float32Array(16);
        for (const [c, r] of cells) {
            const gc = c - minC;
            const gr = r - minR;
            if (gc >= 0 && gc < 4 && gr >= 0 && gr < 4) {
                grid[gr * 4 + gc] = 1.0;
            }
        }
        for (let i = 0; i < 16; i++) inputs[idx++] = grid[i];

        // C. Material (3)
        inputs[idx++] = p.material === 'spring' ? 1.0 : 0.0;
        inputs[idx++] = p.material === 'booster' ? 1.0 : 0.0;
        inputs[idx++] = p.material === 'spike' ? 1.0 : 0.0;

        // D. Terrain lookahead: 10 columns × 4 features (40)
        // Log-spaced: dense near, sparse far — see as far as the player can
        const _colOffsets = [1, 2, 3, 5, 7, 10, 15, 20, 28, 38];
        const playerCol = Math.floor(p.x / TILE);
        for (let ci = 0; ci < 10; ci++) {
            const col = playerCol + _colOffsets[ci];
            let groundH = 0.0;
            let ceilingH = 0.0;
            let hasSpike = 0.0;
            let hasGap = 1.0;

            // Scan top to bottom for ceiling
            for (let y = 0; y < lH; y++) {
                let solid = false;
                if (col >= 0 && col < lW) {
                    if (level[y] && level[y][col] === 1) solid = true;
                    else if (level[y] && level[y][col] === 2) hasSpike = 1.0;
                    const c = corpseGrid[y] && corpseGrid[y][col];
                    if (c) {
                        solid = true;
                        if (c.mat === 'spike') hasSpike = 1.0;
                    }
                }
                if (solid && ceilingH === 0.0 && y > 0) {
                    ceilingH = y / lH;
                }
            }

            // Scan bottom to top for ground
            for (let y = lH - 1; y >= 0; y--) {
                if (col >= 0 && col < lW) {
                    if ((level[y] && level[y][col] === 1) ||
                        (corpseGrid[y] && corpseGrid[y][col] !== null)) {
                        groundH = (lH - y) / lH;
                        hasGap = 0.0;
                        break;
                    }
                }
            }

            inputs[idx++] = groundH;
            inputs[idx++] = ceilingH;
            inputs[idx++] = hasSpike;
            inputs[idx++] = hasGap;
        }

        // E. Buzzsaw proximity (4) — 2 nearest
        const pCX = p.x + TILE * 1.5;
        const pCY = p.y + TILE * 1.5;
        const sawDists = [];
        for (const saw of buzzsaws) {
            const dx = (saw.x - pCX) / (10 * TILE);
            const dy = (saw.y - pCY) / (lH * TILE);
            const dist = dx * dx + dy * dy;
            sawDists.push({ dist, dx, dy });
        }
        sawDists.sort((a, b) => a.dist - b.dist);

        for (let i = 0; i < 2; i++) {
            if (i < sawDists.length) {
                inputs[idx++] = Math.max(-1, Math.min(1, sawDists[i].dx));
                inputs[idx++] = Math.max(-1, Math.min(1, sawDists[i].dy));
            } else {
                inputs[idx++] = 0.0;
                inputs[idx++] = 0.0;
            }
        }

        return inputs;
    }

    // ── Public API ──

    return {
        get active() { return active && loaded; },

        async load(url) {
            try {
                const resp = await fetch(url + '?' + Date.now());
                weights = await resp.json();
                // Convert arrays to typed arrays for speed
                weights.w1 = weights.w1.map(r => Float32Array.from(r));
                weights.b1 = Float32Array.from(weights.b1);
                weights.w2 = weights.w2.map(r => Float32Array.from(r));
                weights.b2 = Float32Array.from(weights.b2);
                weights.w3 = weights.w3.map(r => Float32Array.from(r));
                weights.b3 = Float32Array.from(weights.b3);
                loaded = true;
                console.log('NN agent loaded (' +
                    weights.w1.length + '×' + weights.w1[0].length + ' → ' +
                    weights.w2.length + '×' + weights.w2[0].length + ' → ' +
                    weights.w3.length + '×' + weights.w3[0].length + ')');
            } catch (e) {
                console.warn('NN agent: failed to load weights:', e);
                loaded = false;
            }
        },

        toggle() {
            if (!loaded) {
                console.warn('NN agent: no weights loaded');
                return;
            }
            active = !active;
            console.log('NN agent: ' + (active ? 'ON' : 'OFF'));
        },

        /** Called each frame from update(). Returns action 0-4. */
        getAction() {
            if (!loaded || !active) return 0;

            // Update death memory
            dmTimeSinceDeath += 1/60;
            if (player.x > dmBestX) dmBestX = player.x;
            if ((player.deathCount || 0) > dmPrevDeathCount) {
                dmDeaths = player.deathCount;
                dmLastDeathX = player.x;
                dmTimeSinceDeath = 0;
                dmPrevDeathCount = player.deathCount;
            }

            if (player.dead) {
                progressTimer = 0;
                progressCheckX = player.x;
                return 0;
            }
            // Progress detector: erratic actions to shake loose
            if (player.x > progressCheckX + 2 * TILE) {
                progressCheckX = player.x;
                progressTimer = 0;
            } else {
                progressTimer += 1/60;
                if (progressTimer > 0.5 && player.onGround && Math.random() < 0.15) return 1;
                if (progressTimer > 0.8 && Math.random() < 0.08) return Math.random() < 0.7 ? 2 : 3;
                if (progressTimer > 2.0) { progressTimer = 0; }
            }
            // Get base 72 inputs + append 4 death memory
            const base = getInputs();
            const inputs = new Float32Array(76);
            inputs.set(base);
            inputs[72] = Math.min(1.0, dmDeaths / 5.0);
            const deathDx = dmDeaths > 0 ? (dmLastDeathX - player.x) / (10 * TILE) : 0;
            inputs[73] = Math.max(-1, Math.min(1, deathDx));
            inputs[74] = dmBestX > 0 ? Math.min(1.0, player.x / dmBestX) : 0;
            inputs[75] = Math.min(1.0, dmTimeSinceDeath / 5.0);
            let action = forward(inputs);

            // Loop detector: break repeating action sequences
            actionHist.push(action);
            if (actionHist.length > 24) actionHist.shift();
            for (let patLen = 3; patLen <= 8 && patLen * 2 <= actionHist.length; patLen++) {
                let isLoop = true;
                const start = actionHist.length - patLen * 2;
                for (let ki = 0; ki < patLen; ki++) {
                    if (actionHist[start + ki] !== actionHist[start + patLen + ki]) {
                        isLoop = false; break;
                    }
                }
                if (isLoop) {
                    const alts = [0,1,2,3].filter(a => a !== action);
                    action = alts[Math.floor(Math.random() * alts.length)];
                    loopBreakCount++;
                    break;
                }
            }
            return action;
        },

        /** Reset death memory (call on level restart) */
        resetMemory() {
            dmDeaths = 0; dmLastDeathX = 0; dmBestX = 0;
            dmTimeSinceDeath = 999; dmPrevDeathCount = 0;
            actionHist = []; loopBreakCount = 0;
        },
    };
})();

// Load weights on startup (silently fails if file doesn't exist yet)
nnAgent.load('nn_weights.json');

// Auto-reload weights every 30s during training
setInterval(() => {
    if (nnAgent.active) {
        nnAgent.load('nn_weights.json');
    }
}, 5000);
