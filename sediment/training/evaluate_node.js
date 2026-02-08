// ══════════════════════════════════════════════════════════════
// Sediment Headless Evaluator for NN Training
// Runs the EXACT game JS from index.html in Node.js
// Zero sim-to-game divergence guaranteed
// ══════════════════════════════════════════════════════════════

'use strict';

// ── Simulation time (controlled by evaluator) ──
var _simTime = 0;

// ── Browser API Stubs ──
// These must be declared before the game code loads so that
// browser-only APIs don't crash in Node.js

var _noop = function() {};

// Gain node stub (used by Web Audio)
function _makeGainNode() {
    return {
        connect: _noop,
        gain: {
            value: 0,
            setValueAtTime: _noop,
            exponentialRampToValueAtTime: _noop,
            linearRampToValueAtTime: _noop
        }
    };
}

// Oscillator stub
function _makeOscillator() {
    return {
        connect: _noop,
        start: _noop,
        stop: _noop,
        type: '',
        frequency: {
            value: 0,
            setValueAtTime: _noop,
            exponentialRampToValueAtTime: _noop,
            linearRampToValueAtTime: _noop
        }
    };
}

// Buffer source stub
function _makeBufferSource() {
    return {
        connect: _noop,
        start: _noop,
        stop: _noop,
        buffer: null
    };
}

// AudioContext stub
function _MockAudioContext() {
    this.state = 'running';
    this.sampleRate = 44100;
    this.currentTime = 0;
    this.destination = {};
}
_MockAudioContext.prototype.resume = _noop;
_MockAudioContext.prototype.createOscillator = _makeOscillator;
_MockAudioContext.prototype.createGain = _makeGainNode;
_MockAudioContext.prototype.createBuffer = function(ch, len, sr) {
    return { getChannelData: function() { return new Float32Array(len || 1); } };
};
_MockAudioContext.prototype.createBufferSource = _makeBufferSource;

// DOM element stub factory
function _makeEl() {
    var el = {
        addEventListener: _noop,
        removeEventListener: _noop,
        appendChild: _noop,
        getBoundingClientRect: function() { return {left:0,top:0,width:400,height:168}; },
        getContext: function() { return _ctxProxy; },
        style: {},
        width: 400,
        height: 168,
        canPlayType: function() { return ''; },
        play: function() { return Promise.resolve(); },
        pause: _noop
    };
    // Setters that silently accept values
    Object.defineProperty(el, 'innerHTML', { get: function(){return '';}, set: _noop, configurable: true });
    Object.defineProperty(el, 'textContent', { get: function(){return '';}, set: _noop, configurable: true });
    Object.defineProperty(el, 'loop', { get: function(){return false;}, set: _noop, configurable: true });
    Object.defineProperty(el, 'src', { get: function(){return '';}, set: _noop, configurable: true });
    Object.defineProperty(el, 'volume', { get: function(){return 0;}, set: _noop, configurable: true });
    Object.defineProperty(el, 'muted', { get: function(){return false;}, set: _noop, configurable: true });
    return el;
}

// Canvas 2D context proxy — all draw calls are no-ops
var _ctxProxy = new Proxy({}, {
    get: function(target, prop) {
        if (prop === 'fillStyle' || prop === 'strokeStyle' || prop === 'globalAlpha' ||
            prop === 'font' || prop === 'textAlign' || prop === 'textBaseline' ||
            prop === 'lineWidth' || prop === 'lineCap' || prop === 'shadowColor' ||
            prop === 'shadowBlur' || prop === 'imageSmoothingEnabled') {
            return '';
        }
        return _noop;
    },
    set: function(target, prop, value) { return true; }
});

// ── Global browser API stubs ──
var document = {
    getElementById: function(id) { return _makeEl(); },
    createElement: function(tag) { return _makeEl(); },
    addEventListener: _noop,
    documentElement: { clientWidth: 400, clientHeight: 168 }
};

var window = {
    innerWidth: 400,
    innerHeight: 168,
    addEventListener: _noop,
    AudioContext: _MockAudioContext,
    webkitAudioContext: _MockAudioContext
};

var screen = { width: 400, height: 168 };

var performance = { now: function() { return _simTime * 1000; } };

var localStorage = {
    getItem: function() { return null; },
    setItem: _noop,
    removeItem: _noop
};

var navigator = { maxTouchPoints: 0 };

var Audio = function() { return _makeEl(); };

var requestAnimationFrame = _noop;
var setTimeout = function(fn, ms) { return 0; };
var setInterval = function(fn, ms) { return 0; };
var clearInterval = _noop;
var clearTimeout = _noop;
var alert = _noop;


// ══════════════════════════════════════════════════════════════
// ── GAME CODE (verbatim from index.html lines 139-2223) ──
// ══════════════════════════════════════════════════════════════

// ── SEDIMENT v2 ──
// You ARE the shape. When you die, your body snaps to the grid.
// Completed rows clear. Every player shapes the level.
// by Ivy & Iris

const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');

// ── Config ──
const VERSION = 'v49';
const TILE = 12;
const GRAVITY = 700;
const JUMP_FORCE = -280;
const RUN_SPEED = 100;
const WALL_SLIDE_SPEED = 40;
const MIN_DISTANCE_FOR_CORPSE = 5; // tiles from spawn before death leaves blocks
const COLUMN_CLEAR_THRESHOLD = 6; // filled tiles needed to clear a column (classic=10, tunnel=14)

// ── Colors ──
const C = {
    bg: '#10101E',
    tile: '#38344E',
    tileHL: '#504A68',
    tileSH: '#242038',
    player: '#E8A838',
    playerDark: '#302818',
    spike: '#C83848',
    saw: '#E05838',
    sawBlade: '#582830',
    solid: '#B0A098',       // warm gray stone
    solidHL: '#D0C0B0',
    solidSH: '#807068',
    spring: '#58B848',      // spring green
    springHL: '#A0E890',
    springDark: '#285020',
    booster: '#5898C8',     // steel blue
    boosterHL: '#88C0E0',
    boosterDark: '#283850',
    corpseSpike: '#A83040',     // darker red for corpse spikes
    corpseSpikeHL: '#D05060',
    corpseSpikeDark: '#581828',
    text: '#D8D0C0',
    ghost: 'rgba(232,168,56,0.18)',
    platform: '#6878A8',       // moving platform
    platformHL: '#8898C8',
    platformSH: '#384060'
};

// Material color lookup
function matColor(mat) {
    if (mat === 'spring') return C.spring;
    if (mat === 'booster') return C.booster;
    if (mat === 'spike') return C.corpseSpike;
    return C.solid;
}
function matHL(mat) {
    if (mat === 'spring') return C.springHL;
    if (mat === 'booster') return C.boosterHL;
    if (mat === 'spike') return C.corpseSpikeHL;
    return C.solidHL;
}
function matDark(mat) {
    if (mat === 'spring') return C.springDark;
    if (mat === 'booster') return C.boosterDark;
    if (mat === 'spike') return C.corpseSpikeDark;
    return C.solidSH;
}

// ══════════════════════════════════════
// ── SOUND EFFECTS (procedural Web Audio) ──
// ══════════════════════════════════════

let audioCtx = null;
let sfxMaster = null; // master gain node for all SFX
let sfxMuted = false;
let sfxVolume = 0.7;

function ensureAudio() {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        sfxMaster = audioCtx.createGain();
        sfxMaster.connect(audioCtx.destination);
        sfxMaster.gain.value = sfxMuted ? 0 : sfxVolume;
    }
    if (audioCtx.state === 'suspended') audioCtx.resume();
    return audioCtx;
}

function sfxDest() { return sfxMaster || ensureAudio() && sfxMaster; }

const SFX = {
    jump() {
        const ctx = ensureAudio();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain); gain.connect(sfxDest());
        osc.type = 'square';
        osc.frequency.setValueAtTime(280, ctx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(560, ctx.currentTime + 0.08);
        gain.gain.setValueAtTime(0.12, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.1);
        osc.start(ctx.currentTime);
        osc.stop(ctx.currentTime + 0.1);
    },
    doubleJump() {
        const ctx = ensureAudio();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain); gain.connect(sfxDest());
        osc.type = 'square';
        osc.frequency.setValueAtTime(400, ctx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(800, ctx.currentTime + 0.07);
        gain.gain.setValueAtTime(0.1, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.09);
        osc.start(ctx.currentTime);
        osc.stop(ctx.currentTime + 0.09);
    },
    death() {
        const ctx = ensureAudio();
        // Low thud
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain); gain.connect(sfxDest());
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(150, ctx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(40, ctx.currentTime + 0.2);
        gain.gain.setValueAtTime(0.15, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.25);
        osc.start(ctx.currentTime);
        osc.stop(ctx.currentTime + 0.25);
        // Noise burst
        const buf = ctx.createBuffer(1, ctx.sampleRate * 0.1, ctx.sampleRate);
        const data = buf.getChannelData(0);
        for (let i = 0; i < data.length; i++) data[i] = (Math.random() * 2 - 1) * (1 - i / data.length);
        const noise = ctx.createBufferSource();
        noise.buffer = buf;
        const ng = ctx.createGain();
        noise.connect(ng); ng.connect(sfxDest());
        ng.gain.setValueAtTime(0.08, ctx.currentTime);
        ng.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.1);
        noise.start(ctx.currentTime);
    },
    spring() {
        const ctx = ensureAudio();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain); gain.connect(sfxDest());
        osc.type = 'sine';
        osc.frequency.setValueAtTime(220, ctx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(880, ctx.currentTime + 0.12);
        osc.frequency.exponentialRampToValueAtTime(660, ctx.currentTime + 0.2);
        gain.gain.setValueAtTime(0.15, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.25);
        osc.start(ctx.currentTime);
        osc.stop(ctx.currentTime + 0.25);
    },
    booster() {
        const ctx = ensureAudio();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain); gain.connect(sfxDest());
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(200, ctx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(600, ctx.currentTime + 0.15);
        gain.gain.setValueAtTime(0.1, ctx.currentTime);
        gain.gain.linearRampToValueAtTime(0.12, ctx.currentTime + 0.05);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.2);
        osc.start(ctx.currentTime);
        osc.stop(ctx.currentTime + 0.2);
    },
    crunch() {
        const ctx = ensureAudio();
        // Satisfying chunky clear — layered
        // Bass hit
        const osc1 = ctx.createOscillator();
        const g1 = ctx.createGain();
        osc1.connect(g1); g1.connect(sfxDest());
        osc1.type = 'square';
        osc1.frequency.setValueAtTime(120, ctx.currentTime);
        osc1.frequency.exponentialRampToValueAtTime(60, ctx.currentTime + 0.15);
        g1.gain.setValueAtTime(0.12, ctx.currentTime);
        g1.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.2);
        osc1.start(ctx.currentTime);
        osc1.stop(ctx.currentTime + 0.2);
        // High sparkle
        const osc2 = ctx.createOscillator();
        const g2 = ctx.createGain();
        osc2.connect(g2); g2.connect(sfxDest());
        osc2.type = 'sine';
        osc2.frequency.setValueAtTime(800, ctx.currentTime + 0.05);
        osc2.frequency.exponentialRampToValueAtTime(1200, ctx.currentTime + 0.15);
        g2.gain.setValueAtTime(0.001, ctx.currentTime);
        g2.gain.linearRampToValueAtTime(0.08, ctx.currentTime + 0.05);
        g2.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.25);
        osc2.start(ctx.currentTime);
        osc2.stop(ctx.currentTime + 0.25);
        // Noise crunch
        const buf = ctx.createBuffer(1, ctx.sampleRate * 0.08, ctx.sampleRate);
        const data = buf.getChannelData(0);
        for (let i = 0; i < data.length; i++) data[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / data.length, 2);
        const noise = ctx.createBufferSource();
        noise.buffer = buf;
        const ng = ctx.createGain();
        noise.connect(ng); ng.connect(sfxDest());
        ng.gain.setValueAtTime(0.1, ctx.currentTime);
        ng.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.08);
        noise.start(ctx.currentTime);
    },
    levelComplete() {
        const ctx = ensureAudio();
        // Short fanfare — ascending arpeggio
        const notes = [523, 659, 784, 1047]; // C5 E5 G5 C6
        notes.forEach((freq, i) => {
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.connect(gain); gain.connect(sfxDest());
            osc.type = 'square';
            const t = ctx.currentTime + i * 0.1;
            osc.frequency.setValueAtTime(freq, t);
            gain.gain.setValueAtTime(0.001, ctx.currentTime);
            gain.gain.linearRampToValueAtTime(0.1, t);
            gain.gain.setValueAtTime(0.1, t);
            gain.gain.exponentialRampToValueAtTime(0.001, t + 0.2);
            osc.start(t);
            osc.stop(t + 0.2);
        });
        // Final chord
        [1047, 1319, 1568].forEach(freq => {
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.connect(gain); gain.connect(sfxDest());
            osc.type = 'sine';
            const t = ctx.currentTime + 0.4;
            osc.frequency.setValueAtTime(freq, t);
            gain.gain.setValueAtTime(0.001, ctx.currentTime);
            gain.gain.linearRampToValueAtTime(0.06, t);
            gain.gain.setValueAtTime(0.06, t);
            gain.gain.exponentialRampToValueAtTime(0.001, t + 0.5);
            osc.start(t);
            osc.stop(t + 0.5);
        });
    },
    wallSlide() {
        const ctx = ensureAudio();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain); gain.connect(sfxDest());
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(100 + Math.random() * 50, ctx.currentTime);
        gain.gain.setValueAtTime(0.03, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.05);
        osc.start(ctx.currentTime);
        osc.stop(ctx.currentTime + 0.05);
    },
    rotate() {
        const ctx = ensureAudio();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain); gain.connect(sfxDest());
        osc.type = 'square';
        osc.frequency.setValueAtTime(440, ctx.currentTime);
        osc.frequency.setValueAtTime(520, ctx.currentTime + 0.03);
        gain.gain.setValueAtTime(0.06, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.06);
        osc.start(ctx.currentTime);
        osc.stop(ctx.currentTime + 0.06);
    },
    land() {
        const ctx = ensureAudio();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain); gain.connect(sfxDest());
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(160, ctx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(80, ctx.currentTime + 0.06);
        gain.gain.setValueAtTime(0.08, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.08);
        osc.start(ctx.currentTime);
        osc.stop(ctx.currentTime + 0.08);
    },
    win() {
        const ctx = ensureAudio();
        // Triumphant fanfare for beating the game
        const melody = [523, 659, 784, 1047, 1319, 1568, 2093];
        melody.forEach((freq, i) => {
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.connect(gain); gain.connect(sfxDest());
            osc.type = i < 4 ? 'square' : 'sine';
            const t = ctx.currentTime + i * 0.12;
            osc.frequency.setValueAtTime(freq, t);
            gain.gain.setValueAtTime(0.001, ctx.currentTime);
            gain.gain.linearRampToValueAtTime(0.1, t);
            gain.gain.setValueAtTime(0.1, t);
            gain.gain.exponentialRampToValueAtTime(0.001, t + 0.3);
            osc.start(t);
            osc.stop(t + 0.3);
        });
    }
};

// ══════════════════════════════════════
// ── MUSIC ──
// ══════════════════════════════════════

let musicEl = null;
let musicMuted = false;
let musicVolume = 0.15; // default quieter
let musicStarted = false;
const VOLUME_STEPS = [0, 0.05, 0.10, 0.15, 0.25, 0.40, 0.60];

function initMusic() {
    if (musicEl) return;
    musicEl = document.createElement('audio');
    musicEl.loop = true;
    // Try OGG first, fall back to MP3
    const canOgg = musicEl.canPlayType('audio/ogg; codecs=opus');
    musicEl.src = canOgg ? 'music.ogg' : 'music.mp3';
    // Load saved preferences
    try {
        const saved = JSON.parse(localStorage.getItem('sediment_audio') || '{}');
        if (saved.musicMuted !== undefined) musicMuted = saved.musicMuted;
        if (saved.musicVolume !== undefined) musicVolume = saved.musicVolume;
        if (saved.sfxMuted !== undefined) sfxMuted = saved.sfxMuted;
    } catch(e) {}
    musicEl.volume = musicVolume;
    musicEl.muted = musicMuted;
    if (sfxMaster) sfxMaster.gain.value = sfxMuted ? 0 : sfxVolume;
}

function startMusic() {
    if (musicStarted) return;
    initMusic();
    musicStarted = true;
    musicEl.play().catch(() => {}); // may fail without user gesture
}

function saveAudioPrefs() {
    try { localStorage.setItem('sediment_audio', JSON.stringify({ musicMuted, musicVolume, sfxMuted })); } catch(e) {}
}

function toggleMusicMute() {
    musicMuted = !musicMuted;
    if (musicEl) musicEl.muted = musicMuted;
    saveAudioPrefs();
}

function adjustMusicVolume(dir) {
    // Find current step index and move
    let idx = 0;
    for (let i = 0; i < VOLUME_STEPS.length; i++) {
        if (VOLUME_STEPS[i] <= musicVolume + 0.01) idx = i;
    }
    idx = Math.max(0, Math.min(VOLUME_STEPS.length - 1, idx + dir));
    musicVolume = VOLUME_STEPS[idx];
    if (musicEl) musicEl.volume = musicVolume;
    if (musicVolume === 0) { musicMuted = true; if (musicEl) musicEl.muted = true; }
    else if (musicMuted) { musicMuted = false; if (musicEl) musicEl.muted = false; }
    saveAudioPrefs();
}

function toggleSfxMute() {
    sfxMuted = !sfxMuted;
    if (sfxMaster) sfxMaster.gain.value = sfxMuted ? 0 : sfxVolume;
    saveAudioPrefs();
}

// ══════════════════════════════════════
// ── SAVE / LOAD (localStorage) ──
// ══════════════════════════════════════

const SAVE_KEY = 'sediment_save';

function saveProgress() {
    try {
        const data = {
            highestLevel: player._highestLevel || currentLevel,
            totalDeaths: player._totalDeaths || 0,
            totalQuakes: player._totalQuakes || 0,
            bestRun: player._bestRun || 0,
            hasWon: player._hasWon || false
        };
        localStorage.setItem(SAVE_KEY, JSON.stringify(data));
    } catch (e) { /* localStorage may be blocked */ }
}

function loadProgress() {
    try {
        const raw = localStorage.getItem(SAVE_KEY);
        if (raw) {
            const data = JSON.parse(raw);
            player._highestLevel = data.highestLevel || 1;
            player._totalDeaths = data.totalDeaths || 0;
            player._totalQuakes = data.totalQuakes || data.totalCrunches || 0;
            player._bestRun = data.bestRun || 0;
            player._hasWon = data.hasWon || false;
            return data;
        }
    } catch (e) {}
    player._highestLevel = 1;
    player._totalDeaths = 0;
    player._totalQuakes = 0;
    player._bestRun = 0;
    player._hasWon = false;
    return null;
}

// ── Game resolution ──
const LEVEL_H_TILES = 14;
let W, H, scale;
function resize() {
    const sw = window.innerWidth || document.documentElement.clientWidth || screen.width;
    const sh = window.innerHeight || document.documentElement.clientHeight || screen.height;
    if (sw < 1 || sh < 1) return;
    scale = sh / (LEVEL_H_TILES * TILE);
    W = Math.max(1, Math.floor(sw / scale));
    H = LEVEL_H_TILES * TILE;
    canvas.width = W;
    canvas.height = H;
    canvas.style.imageRendering = 'pixelated';
}
resize();
window.addEventListener('resize', resize);
window.addEventListener('orientationchange', () => { setTimeout(resize, 100); setTimeout(resize, 300); });
let _resizeChecks = 0;
const _resizeInterval = setInterval(() => {
    resize();
    if (++_resizeChecks > 10) clearInterval(_resizeInterval);
}, 500);

// ══════════════════════════════════════
// ── TETROMINO SYSTEM ──
// ══════════════════════════════════════

// Shapes defined as cell offsets [col, row] from origin
// 4 rotation states each, pre-computed for speed
const SHAPES = {
    I: [
        [[0,1],[1,1],[2,1],[3,1]],
        [[2,0],[2,1],[2,2],[2,3]],
        [[0,2],[1,2],[2,2],[3,2]],
        [[1,0],[1,1],[1,2],[1,3]]
    ],
    O: [
        [[0,0],[1,0],[0,1],[1,1]],
        [[0,0],[1,0],[0,1],[1,1]],
        [[0,0],[1,0],[0,1],[1,1]],
        [[0,0],[1,0],[0,1],[1,1]]
    ],
    T: [
        [[0,0],[1,0],[2,0],[1,1]],
        [[1,0],[1,1],[1,2],[0,1]],
        [[1,1],[0,2],[1,2],[2,2]],
        [[1,0],[1,1],[1,2],[2,1]]
    ],
    S: [
        [[1,0],[2,0],[0,1],[1,1]],
        [[0,0],[0,1],[1,1],[1,2]],
        [[1,1],[2,1],[0,2],[1,2]],
        [[1,0],[1,1],[2,1],[2,2]]
    ],
    Z: [
        [[0,0],[1,0],[1,1],[2,1]],
        [[2,0],[1,1],[2,1],[1,2]],
        [[0,1],[1,1],[1,2],[2,2]],
        [[1,0],[0,1],[1,1],[0,2]]
    ],
    L: [
        [[0,0],[1,0],[2,0],[0,1]],
        [[0,0],[1,0],[1,1],[1,2]],
        [[2,1],[0,2],[1,2],[2,2]],
        [[1,0],[1,1],[1,2],[2,2]]
    ],
    J: [
        [[0,0],[1,0],[2,0],[2,1]],
        [[1,0],[1,1],[0,2],[1,2]],
        [[0,1],[0,2],[1,2],[2,2]],
        [[1,0],[2,0],[1,1],[1,2]]
    ]
};

const SHAPE_NAMES = ['I', 'O', 'T', 'S', 'Z', 'L', 'J'];
const MATERIAL_TYPES = ['solid', 'spring', 'booster'];
// Spike material added in later levels — see pickMaterial()

// SRS wall kick data (for non-I pieces)
const KICK_DATA = [
    // 0→1
    [[0,0],[-1,0],[-1,-1],[0,2],[-1,2]],
    // 1→2
    [[0,0],[1,0],[1,1],[0,-2],[1,-2]],
    // 2→3
    [[0,0],[1,0],[1,-1],[0,2],[1,2]],
    // 3→0
    [[0,0],[-1,0],[-1,1],[0,-2],[-1,-2]]
];
const KICK_DATA_I = [
    // 0→1
    [[0,0],[-2,0],[1,0],[-2,1],[1,-2]],
    // 1→2
    [[0,0],[-1,0],[2,0],[-1,-2],[2,1]],
    // 2→3
    [[0,0],[2,0],[-1,0],[2,-1],[-1,2]],
    // 3→0
    [[0,0],[1,0],[-2,0],[1,2],[-2,-1]]
];

// Bag of 7 — all shapes before repeating
let shapeBag = [];
function nextShape() {
    if (shapeBag.length === 0) {
        shapeBag = [...SHAPE_NAMES];
        // Fisher-Yates shuffle
        for (let i = shapeBag.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [shapeBag[i], shapeBag[j]] = [shapeBag[j], shapeBag[i]];
        }
    }
    return shapeBag.pop();
}

// Pick material — spike is a possible material (level 4+, 25% chance)
function pickMaterial(lvl) {
    if (lvl >= 4 && Math.random() < 0.25) return 'spike';
    return MATERIAL_TYPES[Math.floor(Math.random() * 3)];
}

function getCells(shapeName, rotation) {
    return SHAPES[shapeName][rotation];
}

// Bounding box of a shape rotation
function shapeBounds(shapeName, rotation) {
    const cells = getCells(shapeName, rotation);
    let minC = 99, maxC = -99, minR = 99, maxR = -99;
    for (const [c, r] of cells) {
        if (c < minC) minC = c;
        if (c > maxC) maxC = c;
        if (r < minR) minR = r;
        if (r > maxR) maxR = r;
    }
    return { minC, maxC, minR, maxR, w: maxC - minC + 1, h: maxR - minR + 1 };
}

// ══════════════════════════════════════
// ── LEVEL ──
// ══════════════════════════════════════

const SEGMENT_W = 40;
let levelH = 0;
let level = [];      // terrain grid: 0=empty, 1=solid, 2=spike
let corpseGrid = []; // placed blocks: null or {mat, age, eyes}
let segments = [];
let particles = [];
let buzzsaws = [];
let movingPlatforms = [];
let fallingGroup = null; // {cells: [{x,y,mat,eyes}], drop:0, targetDrop:N, fallSpeed:0}
let pendingClears = []; // {cells:[{x,y}], timer:0.35} — flash before delete
let timeSlowTimer = 0; // brief time-slow for quake juice
let encounterLabels = []; // [{text, x, y, color, timer}] — first-encounter pop-ups

function encounterLabel(text, x, y, color) {
    encounterLabels.push({ text, x, y, color, timer: 1.5 });
}

// ── Level system ──
let currentLevel = 1;
let levelDeaths = 0;
let levelQuakes = 0;
let quakeFlash = 0; // timer for "QUAKE!" flash on HUD
let levelComplete = false;
let levelTransitionTimer = 0;
let levelTitleTimer = 0;
let goalX = 0; // pixel X of the finish line
let gameWon = false;
let winTimer = 0;
let winParticleTimer = 0;

function getLevelDifficulty(lvl) {
    return Math.min(10, 1 + (lvl - 1) * 0.82);
}

function getSegmentsForLevel(lvl) {
    return Math.min(7, 3 + Math.floor((lvl - 1) * 0.4));
}

function generateSegment(segIndex, lvl) {
    const h = LEVEL_H_TILES;
    levelH = h;
    const startX = segIndex * SEGMENT_W;
    const baseDiff = getLevelDifficulty(lvl);
    // Ramp within level: later segments slightly harder
    const d = Math.min(10, baseDiff + segIndex * 0.3);

    for (let y = 0; y < h; y++) {
        if (!level[y]) level[y] = [];
        if (!corpseGrid[y]) corpseGrid[y] = [];
        for (let x = startX; x < startX + SEGMENT_W; x++) {
            if (y === 0 || y === h - 1) {
                level[y][x] = 1;
            } else {
                level[y][x] = 0;
            }
            corpseGrid[y][x] = null;
        }
    }

    // ── Mechanic gating by level number ──
    // L1: chasms only | L2: floor spikes | L3: ceiling spikes
    // L4: spike material | L5: static platforms | L6: buzzsaws
    // L7: moving platforms (H) | L8: spiked chasms | L9: spike walls
    // L10: moving platforms (V) | L11-12: intensity

    // Chasms (always)
    const numChasms = 2 + Math.floor(d * 0.8);
    const chasmMinX = segIndex === 0 ? 10 : 3;
    for (let i = 0; i < numChasms; i++) {
        const cx = startX + chasmMinX + Math.floor(Math.random() * (SEGMENT_W - chasmMinX - 4));
        const cw = 2 + Math.floor(Math.random() * (1 + d * 0.4));
        for (let dx = 0; dx < cw; dx++) {
            if (lvl >= 8 && Math.random() < Math.min(0.8, d * 0.1)) {
                // Spiked chasms (level 8+)
                level[h - 2][cx + dx] = 2;
            } else {
                level[h - 1][cx + dx] = 0;
            }
        }
    }

    // Floor spikes (level 2+)
    if (lvl >= 2) {
        const numSpikeFields = 1 + Math.floor(d * 0.6);
        const spikeMinX = segIndex === 0 ? 10 : 3;
        for (let i = 0; i < numSpikeFields; i++) {
            const sx = startX + spikeMinX + Math.floor(Math.random() * (SEGMENT_W - spikeMinX - 3));
            const sw = 2 + Math.floor(Math.random() * 3);
            for (let dx = 0; dx < sw; dx++) {
                if (level[h - 2] && level[h - 1][sx + dx] === 1) {
                    level[h - 2][sx + dx] = 2;
                }
            }
        }
    }

    // Ceiling spikes (level 3+)
    if (lvl >= 3) {
        const numCeilingSpikes = Math.max(1, Math.floor(d * 0.5));
        for (let i = 0; i < numCeilingSpikes; i++) {
            const sx = startX + 2 + Math.floor(Math.random() * (SEGMENT_W - 4));
            if (level[1]) level[1][sx] = 2;
            if (d > 6 && Math.random() < 0.4 && level[2]) level[2][sx] = 2;
        }
    }

    // Static platforms (level 5+)
    if (lvl >= 5) {
        const numPlats = 1 + Math.floor(d * 0.3);
        const platMinX = segIndex === 0 ? 14 : 4;
        for (let i = 0; i < numPlats; i++) {
            const px = startX + platMinX + Math.floor(Math.random() * (SEGMENT_W - platMinX - 6));
            const pw = 2 + Math.floor(Math.random() * 3);
            const py = 3 + Math.floor(Math.random() * (h - 6));
            for (let dx = 0; dx < pw; dx++) {
                if (level[py]) level[py][px + dx] = 1;
            }
        }
    }

    // Buzzsaws (level 6+)
    if (lvl >= 6) {
        const numSaws = Math.max(1, Math.floor((d - 4) * 0.6));
        for (let i = 0; i < numSaws; i++) {
            const sawX = (startX + 8 + Math.floor(Math.random() * (SEGMENT_W - 16))) * TILE;
            const sawY = (2 + Math.floor(Math.random() * (h - 5))) * TILE;
            const vertical = Math.random() < 0.5;
            const range = (2 + Math.floor(Math.random() * 3)) * TILE;
            buzzsaws.push({
                x: sawX, y: sawY,
                baseX: sawX, baseY: sawY,
                r: 6, vertical, range,
                speed: 30 + Math.random() * 40,
                phase: Math.random() * Math.PI * 2,
                spin: 0
            });
        }
    }

    // Moving platforms — horizontal (level 7+)
    if (lvl >= 7) {
        const numMoving = 1 + Math.floor((d - 5) * 0.4);
        const movMinX = segIndex === 0 ? 16 : 5;
        for (let i = 0; i < numMoving; i++) {
            const mx = (startX + movMinX + Math.floor(Math.random() * (SEGMENT_W - movMinX - 8))) * TILE;
            const my = (3 + Math.floor(Math.random() * (h - 6))) * TILE;
            const vert = lvl >= 10 && Math.random() < 0.4; // vertical only level 10+
            movingPlatforms.push({
                x: mx, y: my, w: 3,
                baseX: mx, baseY: my,
                vertical: vert,
                range: (2 + Math.floor(Math.random() * 2)) * TILE,
                speed: 20 + Math.random() * 30,
                phase: Math.random() * Math.PI * 2
            });
        }
    }

    // Spike walls (level 9+)
    if (lvl >= 9) {
        const numWalls = Math.max(1, Math.floor((d - 6) * 0.5));
        for (let i = 0; i < numWalls; i++) {
            const wx = startX + 6 + Math.floor(Math.random() * (SEGMENT_W - 12));
            const wallH = 2 + Math.floor(Math.random() * 3);
            const startY = h - 2 - wallH;
            for (let dy = 0; dy < wallH; dy++) {
                if (level[startY + dy]) level[startY + dy][wx] = 2;
            }
        }
    }

    // Safe spawn
    if (segIndex === 0) {
        for (let y = 1; y < h - 1; y++) {
            for (let x = 0; x < 10; x++) {
                if (level[y] && level[y][x] === 2) level[y][x] = 0;
            }
        }
        for (let x = 0; x < 10; x++) {
            level[h - 1][x] = 1;
        }
    }

    segments.push(segIndex);
}

// ══════════════════════════════════════
// ── COLLISION (tetromino-aware) ──
// ══════════════════════════════════════

// Check if a single tile-aligned cell overlaps solid terrain or corpse blocks
// ALL corpse blocks are solid (sandstone, spring, booster) — you can't walk through any of them
// Interactions (bounce, boost) happen at collision boundaries
function tileIsSolid(tx, ty) {
    if (ty < 0) return true; // ceiling
    if (ty >= levelH) return false; // void below
    if (level[ty] && level[ty][tx] === 1) return true;
    if (corpseGrid[ty] && corpseGrid[ty][tx]) return true;
    // Moving platforms
    for (const p of movingPlatforms) {
        const px1 = Math.floor(p.x / TILE);
        const py = Math.floor(p.y / TILE);
        if (ty === py && tx >= px1 && tx < px1 + p.w) return true;
    }
    return false;
}

// Check what corpse block is at a tile position (returns mat string or null)
function corpseMatAt(tx, ty) {
    if (ty < 0 || ty >= levelH) return null;
    if (corpseGrid[ty] && corpseGrid[ty][tx]) return corpseGrid[ty][tx].mat;
    return null;
}

// Check tiles adjacent to player's tetromino in a direction
// Returns first matching material found, or null
function checkAdjacentCorpse(px, py, shapeName, rotation, dir, mat) {
    const cells = getCells(shapeName, rotation);
    for (const [cx, cy] of cells) {
        let checkTX, checkTY;
        if (dir === 'below') {
            checkTX = Math.floor((px + cx * TILE + TILE/2) / TILE);
            checkTY = Math.floor((py + (cy + 1) * TILE) / TILE);
        } else if (dir === 'above') {
            checkTX = Math.floor((px + cx * TILE + TILE/2) / TILE);
            checkTY = Math.floor((py + cy * TILE - 1) / TILE);
        } else if (dir === 'right') {
            checkTX = Math.floor((px + (cx + 1) * TILE) / TILE);
            checkTY = Math.floor((py + cy * TILE + TILE/2) / TILE);
        } else if (dir === 'left') {
            checkTX = Math.floor((px + cx * TILE - 1) / TILE);
            checkTY = Math.floor((py + cy * TILE + TILE/2) / TILE);
        }
        if (corpseMatAt(checkTX, checkTY) === mat) {
            return { tx: checkTX, ty: checkTY };
        }
    }
    return null;
}

function tileIsSpike(tx, ty) {
    if (ty < 0 || ty >= levelH) return false;
    return level[ty] && level[ty][tx] === 2;
}

// Check if a tetromino at pixel position (px, py) collides with solids
// px, py is top-left of cell grid origin
function tetSolid(px, py, shapeName, rotation) {
    const cells = getCells(shapeName, rotation);
    for (const [cx, cy] of cells) {
        const worldX = px + cx * TILE;
        const worldY = py + cy * TILE;
        // Check all grid tiles this cell touches
        const l = Math.floor(worldX / TILE);
        const r = Math.floor((worldX + TILE - 1) / TILE);
        const t = Math.floor(worldY / TILE);
        const b = Math.floor((worldY + TILE - 1) / TILE);
        for (let ty = t; ty <= b; ty++) {
            for (let tx = l; tx <= r; tx++) {
                if (tileIsSolid(tx, ty)) return true;
            }
        }
    }
    return false;
}

function tetSpike(px, py, shapeName, rotation) {
    const cells = getCells(shapeName, rotation);
    for (const [cx, cy] of cells) {
        const worldX = px + cx * TILE + 1; // slight inset
        const worldY = py + cy * TILE + 1;
        const l = Math.floor(worldX / TILE);
        const r = Math.floor((worldX + TILE - 3) / TILE);
        const t = Math.floor(worldY / TILE);
        const b = Math.floor((worldY + TILE - 3) / TILE);
        for (let ty = t; ty <= b; ty++) {
            for (let tx = l; tx <= r; tx++) {
                if (tileIsSpike(tx, ty)) return true;
            }
        }
    }
    return false;
}

// Monte Carlo escape check: can the player get past the obstacle to the right?
// Pixel-by-pixel upward scan + rotation with SRS kicks.
// Returns true if any escape route exists (player should wait, not die).
function canEscapeRight() {
    const shape = player.shape;
    const rot = player.rotation;
    const maxUp = 8 * TILE; // jump + double jump reach

    // 1) Scan upward pixel-by-pixel — continuous reachability check
    //    Must be contiguous: if any height is blocked, can't reach higher
    for (let dy = -1; dy >= -maxUp; dy--) {
        const testY = player.y + dy;
        // Can the piece exist at this height? (continuous upward path)
        if (tetSolid(player.x, testY, shape, rot)) break;
        // At this height, can we move right?
        if (!tetSolid(player.x + 1, testY, shape, rot)) return true;
    }

    // 2) Check each rotation (CW and CCW) with SRS wall kicks
    for (const dir of [1, -1]) {
        const newRot = (rot + dir + 4) % 4;
        const kicks = shape === 'I' ? KICK_DATA_I : KICK_DATA;
        const kickIdx = dir === 1 ? rot : newRot;
        const kickTests = kicks[kickIdx];

        for (const [kx, ky] of kickTests) {
            const testX = player.x + (dir === 1 ? kx : -kx) * TILE;
            const testY = player.y + (dir === 1 ? -ky : ky) * TILE;
            // Can rotated piece fit here?
            if (!tetSolid(testX, testY, shape, newRot)) {
                // Can it move right from there?
                if (!tetSolid(testX + 1, testY, shape, newRot)) return true;
                // Can it jump and clear from there? (pixel-by-pixel)
                for (let dy = -1; dy >= -maxUp; dy--) {
                    const jy = testY + dy;
                    if (tetSolid(testX, jy, shape, newRot)) break;
                    if (!tetSolid(testX + 1, jy, shape, newRot)) return true;
                }
            }
        }
    }

    return false; // truly stuck — no escape route found
}

// (tetCorpseInteraction removed — interactions now use checkAdjacentCorpse at collision boundaries)

// ══════════════════════════════════════
// ── PLAYER ──
// ══════════════════════════════════════

const player = {
    x: 2 * TILE, y: 0,
    vx: RUN_SPEED, vy: 0,
    shape: 'T',
    rotation: 0,
    material: 'solid',
    onGround: false,
    onWall: 0,
    dead: false,
    facing: 1,
    squash: 1,
    trail: [],
    boostTimer: 0,
    deathCount: 0,
    chasmDeaths: 0,
    quakeCount: 0,
    quakeTimer: 0,
    distance: 0,
    bestDistance: 0,
    spawnX: 2 * TILE,
    // Double jump
    jumpsLeft: 2,
    coyoteTimer: 0,
    deathTimer: 0,
    stuckTimer: 0,
    stuckHighX: 0
};

const cam = { x: 0, y: 0, sx: 0, sy: 0, shake: 0 };
const flash = { a: 0, r: 1, g: 1, b: 1 };
let paused = false;
let started = false;
let tutorialStep = 3; // 0=jump, 1=rotate, 2=place, 3=done
let tutorialComplete = false;


// Next piece preview
let nextPieceShape = null;
let nextPieceMaterial = null;

function rollNextPiece() {
    nextPieceShape = nextShape();
    nextPieceMaterial = pickMaterial(currentLevel);
}

function initLevel(lvl) {
    level = [];
    corpseGrid = [];
    segments = [];
    particles = [];
    buzzsaws = [];
    movingPlatforms = [];
    fallingGroup = null;
    pendingClears = [];
    levelComplete = false;
    levelTransitionTimer = 0;
    levelDeaths = 0;
    levelQuakes = 0;
    quakeFlash = 0;
    levelTitleTimer = 3.0;

    // Tutorial on level 1 (first time only)
    if (lvl === 1 && !tutorialComplete) {
        tutorialStep = 0;
    } else {
        tutorialStep = 3;
    }

    if (!player._hasRun) {
    }

    // Generate all segments for this level
    const numSegments = getSegmentsForLevel(lvl);
    for (let i = 0; i < numSegments; i++) {
        generateSegment(i, lvl);
    }

    // Goal line near the end of the last segment
    goalX = (numSegments * SEGMENT_W - 4) * TILE;

    const h = LEVEL_H_TILES;

    const spawnH = h;
    // Use pre-rolled next piece if available, otherwise roll fresh
    if (nextPieceShape) {
        player.shape = nextPieceShape;
        player.material = nextPieceMaterial;
    } else {
        player.shape = nextShape();
        player.material = pickMaterial(currentLevel);
    }
    player.rotation = 0;
    rollNextPiece(); // pre-roll for preview
    player.x = 2 * TILE;
    player.y = (spawnH - 3) * TILE;
    player.spawnX = player.x;
    player.vx = RUN_SPEED;
    player.vy = 0;
    player.dead = false;
    player.distance = 0;
    player.squash = 1;
    player.trail = [];
    player.boostTimer = 0;
    player.jumpsLeft = 2;
    player.coyoteTimer = 0;
    player.stuckTimer = 0;
    player.stuckHighX = player.x;
    player._hasRun = true;
    cam.x = 0; cam.y = 0;
}

function init() {
    loadProgress();
    currentLevel = 1;
    initLevel(currentLevel);
}

function hardReset() {
    player._hasRun = false;
    player.deathCount = 0;
    player.quakeCount = 0;
    player.quakeTimer = 0;
    player.bestDistance = 0;
    player._hitSpring = false;
    player._hitBooster = false;
    shapeBag = [];
    currentLevel = 1;
    tutorialComplete = false;
    gameWon = false;
    encounterLabels = [];
    init();
}

// ── Particles ──
function spawnP(x, y, color, count, speed, life) {
    for (let i = 0; i < count; i++) {
        const a = Math.random() * Math.PI * 2;
        const s = speed * (0.5 + Math.random() * 0.5);
        particles.push({
            x, y, vx: Math.cos(a) * s, vy: Math.sin(a) * s - speed * 0.3,
            life: life * (0.5 + Math.random() * 0.5), maxLife: life,
            color, size: 1 + Math.random()
        });
    }
}

// ══════════════════════════════════════
// ── DEATH + GRID SNAP + GRAVITY ──
// ══════════════════════════════════════

function die() {
    if (player.dead) return;
    player.dead = true;
    player.deathTimer = 0;
    player.deathCount++;
    levelDeaths++;
    player._totalDeaths = (player._totalDeaths || 0) + 1;

    cam.shake = Math.max(cam.shake, 6);
    flash.a = 0.6; flash.r = 0.91; flash.g = 0.66; flash.b = 0.22;
    SFX.death();

    const cells = getCells(player.shape, player.rotation);
    const cx = player.x + getShapeCenterX(player.shape, player.rotation);
    const cy = player.y + getShapeCenterY(player.shape, player.rotation);
    spawnP(cx, cy, C.player, 12, 80, 0.4);
    spawnP(cx, cy, C.spike, 6, 50, 0.3);

    // Check minimum distance
    const distFromSpawn = Math.floor((player.x - player.spawnX) / TILE);

    if (distFromSpawn >= MIN_DISTANCE_FOR_CORPSE && !_evalNoCorpses) {
        // Snap each cell to the grid — clamp to valid range so chasm deaths still place blocks
        const groupCells = [];
        for (let i = 0; i < cells.length; i++) {
            const [cellC, cellR] = cells[i];
            const worldPx = player.x + cellC * TILE;
            const worldPy = player.y + cellR * TILE;
            const gridX = Math.round(worldPx / TILE);
            const gridY = Math.max(0, Math.min(Math.round(worldPy / TILE), levelH - 1));
            if (gridX >= 0) {
                groupCells.push({ x: gridX, y: gridY, mat: player.material, eyes: i === 0 });
            }
        }

        // Compute how far this group can fall as a unit
        const targetDrop = computeGroupDrop(groupCells);

        // Create falling group — they fall together, maintaining shape
        fallingGroup = {
            cells: groupCells,
            drop: 0,
            targetDrop,
            fallSpeed: 0
        };

        // Snap particles at each cell
        for (const { x: gx, y: gy } of groupCells) {
            spawnP(gx * TILE + TILE/2, gy * TILE + TILE/2, matColor(player.material), 3, 40, 0.25);
        }
    }
}

function getShapeCenterX(shapeName, rotation) {
    const cells = getCells(shapeName, rotation);
    let sum = 0;
    for (const [c, r] of cells) sum += c;
    return (sum / cells.length) * TILE + TILE / 2;
}

function getShapeCenterY(shapeName, rotation) {
    const cells = getCells(shapeName, rotation);
    let sum = 0;
    for (const [c, r] of cells) sum += r;
    return (sum / cells.length) * TILE + TILE / 2;
}

// Compute how far a group of cells can fall as a unit
function computeGroupDrop(cells) {
    for (let drop = 0; drop < levelH; drop++) {
        // Can the group drop one more tile?
        for (const { x: gx, y: gy } of cells) {
            const testY = gy + drop + 1;
            if (testY >= levelH) return drop;
            // Blocked by terrain or existing corpse?
            const blocked = (level[testY] && level[testY][gx] === 1) ||
                            (corpseGrid[testY] && corpseGrid[testY][gx]);
            if (blocked) return drop;
        }
    }
    return 0;
}

// Update falling group — all cells fall together
function updateFallingGroup(dt) {
    if (!fallingGroup) return;

    fallingGroup.fallSpeed += 600 * dt;
    fallingGroup.drop += fallingGroup.fallSpeed * dt;

    if (fallingGroup.drop >= fallingGroup.targetDrop) {
        // Landed! Place all cells
        fallingGroup.drop = fallingGroup.targetDrop;
        for (const cell of fallingGroup.cells) {
            const landY = cell.y + fallingGroup.targetDrop;
            if (landY >= 0 && landY < levelH) {
                if (!corpseGrid[landY]) corpseGrid[landY] = [];
                if (!(level[landY] && level[landY][cell.x] === 1)) {
                    if (cell.mat === 'spike') {
                        // Spike material → becomes permanent terrain spike
                        if (!level[landY]) level[landY] = [];
                        level[landY][cell.x] = 2;
                    } else {
                        corpseGrid[landY][cell.x] = { mat: cell.mat, eyes: cell.eyes, age: 0, deathNum: player.deathCount };
                        // Non-spike corpses neutralize terrain spikes
                        if (level[landY] && level[landY][cell.x] === 2) {
                            level[landY][cell.x] = 0;
                        }
                    }
                }
            }
            spawnP(cell.x * TILE + TILE/2, (cell.y + fallingGroup.targetDrop) * TILE + TILE, C.tile, 3, 25, 0.15);
        }
        cam.shake = Math.max(cam.shake, 2);
        fallingGroup = null;
        checkRowClears();
    }
}

// ══════════════════════════════════════
// ── ROW CLEARING ──
// ══════════════════════════════════════

function checkRowClears() {
    const maxGenX = segments.length * SEGMENT_W;
    let cleared = false;

    // VERTICAL column clearing — 6+ CONSECUTIVE non-spike corpse blocks
    // clearColumn already clears the whole column, so just need to detect threshold
    for (let x = 0; x < maxGenX; x++) {
        let run = 0;
        let found = false;
        for (let y = 0; y < levelH; y++) {
            const b = corpseGrid[y] && corpseGrid[y][x];
            if (b && b.mat !== 'spike') {
                run++;
            } else {
                if (run >= COLUMN_CLEAR_THRESHOLD) { found = true; break; }
                run = 0;
            }
        }
        if (run >= COLUMN_CLEAR_THRESHOLD) found = true;
        if (found) {
            clearColumn(x);
            cleared = true;
        }
    }

    // HORIZONTAL row clearing — 6+ CONSECUTIVE non-spike corpse blocks
    // Let runs grow past 6 — clear the full segment when it ends
    for (let y = 1; y < levelH - 1; y++) {
        let run = 0;
        let runStart = 0;
        for (let x = 0; x < maxGenX; x++) {
            const b = corpseGrid[y] && corpseGrid[y][x];
            if (b && b.mat !== 'spike') {
                if (run === 0) runStart = x;
                run++;
            } else {
                if (run >= COLUMN_CLEAR_THRESHOLD) {
                    clearRowSegment(y, runStart, runStart + run - 1);
                    cleared = true;
                }
                run = 0;
            }
        }
        // Check run at end of row
        if (run >= COLUMN_CLEAR_THRESHOLD) {
            clearRowSegment(y, runStart, runStart + run - 1);
            cleared = true;
        }
    }

    // Gravity now applied when pendingClears timers expire
}

function onQuake() {
    player.quakeCount++;
    levelQuakes++;
    player._totalQuakes = (player._totalQuakes || 0) + 1;
    SFX.crunch();
    timeSlowTimer = 0.3; // brief time-slow for juice
    // Stack invincibility — 5s base, combos stack +5s each
    player.quakeTimer += 5.0;
    quakeFlash = 1.5;
    // Screen shake — bigger for combos
    const combo = player.quakeTimer > 5.5 ? Math.floor(player.quakeTimer / 5) + 1 : 0;
    cam.shake = Math.max(cam.shake, combo >= 2 ? 12 : 8);
    // Big screen feedback — combo counter
    const cx = player.x + getShapeCenterX(player.shape, player.rotation);
    const cy = player.y + getShapeCenterY(player.shape, player.rotation);
    const label = combo >= 2 ? `QUAKE x${combo}!` : 'QUAKE!';
    encounterLabel(label, cx, cy - 10, '#FFD080');
}

function clearColumn(x) {
    const cells = [];
    for (let y = 0; y < levelH; y++) {
        if (corpseGrid[y] && corpseGrid[y][x]) {
            cells.push({ x, y });
            corpseGrid[y][x].clearing = true;
        }
    }
    if (cells.length) {
        pendingClears.push({ cells, timer: 0.35 });
        onQuake();
    }
}

function clearRowSegment(y, startX, endX) {
    const cells = [];
    for (let x = startX; x <= endX; x++) {
        if (corpseGrid[y] && corpseGrid[y][x]) {
            cells.push({ x, y });
            corpseGrid[y][x].clearing = true;
        }
    }
    if (cells.length) {
        pendingClears.push({ cells, timer: 0.35 });
        onQuake();
    }
}

function updatePendingClears(dt) {
    for (let i = pendingClears.length - 1; i >= 0; i--) {
        const pc = pendingClears[i];
        pc.timer -= dt;
        if (pc.timer <= 0) {
            // Actually delete + particles
            cam.shake = Math.max(cam.shake, 8);
            flash.a = 0.6; flash.r = 1; flash.g = 0.94; flash.b = 0.82;
            for (const c of pc.cells) {
                if (corpseGrid[c.y] && corpseGrid[c.y][c.x]) {
                    const block = corpseGrid[c.y][c.x];
                    spawnP(c.x * TILE + TILE/2, c.y * TILE + TILE/2, matColor(block.mat), 3, 80, 0.4);
                    corpseGrid[c.y][c.x] = null;
                }
            }
            // Big burst at center of group — extra juice
            let cx = 0, cy = 0;
            for (const c of pc.cells) { cx += c.x; cy += c.y; }
            cx = (cx / pc.cells.length) * TILE + TILE/2;
            cy = (cy / pc.cells.length) * TILE + TILE/2;
            spawnP(cx, cy, C.player, 20, 140, 0.6);
            spawnP(cx, cy, '#FFF0D0', 8, 100, 0.4);
            spawnP(cx, cy, C.spring, 5, 80, 0.3);
            pendingClears.splice(i, 1);
            applyCorpseGravity();
        }
    }
}

function applyCorpseGravity() {
    // For each column, drop corpse blocks down into empty spaces
    const maxX = segments.length * SEGMENT_W;
    let dropped = false;

    // Work bottom to top
    for (let x = 0; x < maxX; x++) {
        for (let y = levelH - 2; y >= 1; y--) {
            if (!corpseGrid[y] || !corpseGrid[y][x]) continue;
            // This cell has a corpse block — check if it can fall
            let newY = y;
            for (let ty = y + 1; ty < levelH; ty++) {
                const isSolid = (level[ty] && level[ty][x] === 1) || (corpseGrid[ty] && corpseGrid[ty][x]);
                if (isSolid) break;
                newY = ty;
            }
            if (newY !== y) {
                if (!corpseGrid[newY]) corpseGrid[newY] = [];
                corpseGrid[newY][x] = corpseGrid[y][x];
                corpseGrid[y][x] = null;
                dropped = true;
            }
        }
    }

    // If blocks dropped, check for cascading clears
    if (dropped) {
        checkRowClears();
    }
}

// ══════════════════════════════════════
// ── ROTATION ──
// ══════════════════════════════════════

function tryRotate(dir) {
    // dir: 1 = clockwise, -1 = counter-clockwise
    // Rate limit: max ~4 rotations/sec
    if (player.rotateCooldown > 0) return false;
    const oldRot = player.rotation;
    const newRot = (oldRot + dir + 4) % 4;
    const kicks = player.shape === 'I' ? KICK_DATA_I : KICK_DATA;

    // Determine kick table index
    let kickIdx;
    if (dir === 1) {
        kickIdx = oldRot; // 0→1, 1→2, 2→3, 3→0
    } else {
        // Reverse: use the target→old kicks, but negated
        kickIdx = newRot;
    }

    const kickTests = kicks[kickIdx];
    for (const [kx, ky] of kickTests) {
        const testX = player.x + (dir === 1 ? kx : -kx) * TILE;
        const testY = player.y + (dir === 1 ? -ky : ky) * TILE;
        if (!tetSolid(testX, testY, player.shape, newRot)) {
            player.x = testX;
            player.y = testY;
            player.rotation = newRot;
            player.squash = 1.15;
            const cx = player.x + getShapeCenterX(player.shape, player.rotation);
            const cy = player.y + getShapeCenterY(player.shape, player.rotation);
            // Spin boost/brake
            // Brake (CCW) = always free
            // Boost (CW) = cooldown; spam = forced brake
            if (dir === -1) {
                player.vx = 20;
                player.boostTimer = 0.5;
            } else if (!player.spinCooldown || player.spinCooldown <= 0) {
                player.vx = 200;
                player.boostTimer = 0.5;
                player.spinCooldown = 1.0;
            } else {
                // CW spam = brake check
                player.vx = 20;
                player.boostTimer = 0.5;
            }
            SFX.rotate();
            player.rotateCooldown = 0.25;
            return true;
        }
    }
    return false; // rotation failed
}

// ══════════════════════════════════════
// ── RESPAWN ──
// ══════════════════════════════════════

function respawn() {
    const h = LEVEL_H_TILES;
    player.shape = nextPieceShape;
    player.material = nextPieceMaterial;
    player.rotation = 0;
    rollNextPiece(); // pre-roll for preview
    player.x = 2 * TILE;
    player.y = (h - 3) * TILE;
    player.spawnX = player.x;
    player.vx = RUN_SPEED;
    player.vy = 0;
    player.dead = false;
    player.onGround = false;
    player.squash = 0.5;
    player.trail = [];
    player.boostTimer = 0;
    // NOTE: quakeTimer is NOT reset here — invincibility persists through respawn
    player.jumpsLeft = 2;
    player.spinCooldown = 0;
    player.rotateCooldown = 0;
    player.coyoteTimer = 0;
    player.distance = 0;
    player.stuckTimer = 0;
    player.stuckHighX = player.x;
    cam.x = 0; cam.y = 0;
}

// ══════════════════════════════════════
// ── INPUT ──
// ══════════════════════════════════════

let jumpBuffered = false;
let _skipBtnBounds = null; // {x,y,w,h} in game pixels when pause menu is showing
let _muteBtnBounds = null;
let _volDownBounds = null;
let _volUpBounds = null;
let _sfxBtnBounds = null;
let touchStartX = 0, touchStartY = 0;
let touchStartTime = 0;
let touchHandled = false;
const SWIPE_THRESHOLD = 25; // pixels

// Input method detection — desktop vs touch
let isTouch = ('ontouchstart' in window) || (navigator.maxTouchPoints > 0);
let lastInputWasTouch = isTouch;

function ctrlHint(action) {
    if (lastInputWasTouch) {
        if (action === 'jump') return 'tap = jump';
        if (action === 'die') return 'swipe \u2193 = die';
        if (action === 'rotate') return 'swipe \u2190\u2192 = rotate';
        return 'tap=jump  swipe \u2193=die  swipe \u2190\u2192=rotate';
    } else {
        if (action === 'jump') return 'SPACE / W = jump';
        if (action === 'die') return 'X / S = die';
        if (action === 'rotate') return 'A / D = rotate';
        return 'SPACE=jump  X=die  A/D=rotate';
    }
}

function doJump() {
    if (!started) { started = true; startMusic(); return; }
    if (tutorialStep === 0) tutorialStep = 1;
    jumpBuffered = true;
}

function doDie() {
    if (!started) { started = true; startMusic(); return; }
    if (tutorialStep === 0) return; // blocked until jump
    if (tutorialStep === 2) { tutorialStep = 3; tutorialComplete = true; }
    die();
}

function doRotateCW() {
    if (!started || player.dead) return;
    if (tutorialStep === 0) return; // blocked until jump
    if (tutorialStep === 1) tutorialStep = 2;
    tryRotate(1);
}

function doRotateCCW() {
    if (!started || player.dead) return;
    if (tutorialStep === 0) return; // blocked until jump
    if (tutorialStep === 1) tutorialStep = 2;
    tryRotate(-1);
}

// Touch input — tap=jump, swipe down=die, swipe left/right=rotate
canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    lastInputWasTouch = true;
    const t = e.changedTouches[0];
    touchStartX = t.clientX;
    touchStartY = t.clientY;
    touchStartTime = performance.now();
    touchHandled = false;
}, { passive: false });

canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    if (touchHandled) return;
    const t = e.changedTouches[0];
    const dx = t.clientX - touchStartX;
    const dy = t.clientY - touchStartY;

    if (Math.abs(dy) > SWIPE_THRESHOLD && dy > 0 && Math.abs(dy) > Math.abs(dx)) {
        // Swipe down = voluntary death
        touchHandled = true;
        doDie();
    } else if (Math.abs(dx) > SWIPE_THRESHOLD && Math.abs(dx) > Math.abs(dy)) {
        touchHandled = true;
        if (dx > 0) doRotateCW();
        else doRotateCCW();
    }
}, { passive: false });

canvas.addEventListener('touchend', (e) => {
    e.preventDefault();
    if (!touchHandled) {
        const elapsed = performance.now() - touchStartTime;
        if (elapsed < 300) {
            // Check if tapping pause menu buttons
            if (paused) {
                const t = e.changedTouches[0];
                const rect = canvas.getBoundingClientRect();
                const tapX = (t.clientX - rect.left) / rect.width * W;
                const tapY = (t.clientY - rect.top) / rect.height * H;
                if (_skipBtnBounds) {
                    const b = _skipBtnBounds;
                    if (tapX >= b.x && tapX <= b.x + b.w && tapY >= b.y && tapY <= b.y + b.h) {
                        restartLevel();
                        return;
                    }
                }
                if (_volDownBounds) {
                    const b = _volDownBounds;
                    if (tapX >= b.x && tapX <= b.x + b.w && tapY >= b.y && tapY <= b.y + b.h) {
                        adjustMusicVolume(-1);
                        return;
                    }
                }
                if (_volUpBounds) {
                    const b = _volUpBounds;
                    if (tapX >= b.x && tapX <= b.x + b.w && tapY >= b.y && tapY <= b.y + b.h) {
                        adjustMusicVolume(1);
                        return;
                    }
                }
                if (_sfxBtnBounds) {
                    const b = _sfxBtnBounds;
                    if (tapX >= b.x && tapX <= b.x + b.w && tapY >= b.y && tapY <= b.y + b.h) {
                        toggleSfxMute();
                        return;
                    }
                }
                // Tap anywhere else = resume
                togglePause();
                return;
            }
            doJump();
        }
    }
}, { passive: false });

// Keyboard
document.addEventListener('keydown', (e) => {
    if (e.repeat) return;
    lastInputWasTouch = false;
    if (e.code === 'Space' || e.code === 'ArrowUp' || e.code === 'KeyW') doJump();
    if (e.code === 'ArrowDown' || e.code === 'KeyS' || e.code === 'KeyX') doDie();
    if (e.code === 'ArrowRight' || e.code === 'KeyD') doRotateCW();
    if (e.code === 'ArrowLeft' || e.code === 'KeyA') doRotateCCW();
    if (e.code === 'KeyR') hardReset();
    if (e.code === 'KeyP' || e.code === 'Escape') togglePause();
    if (e.code === 'KeyN' && paused) restartLevel();
    if (e.code === 'KeyM') toggleMusicMute();
    if (e.code === 'KeyF') toggleSfxMute();
    if (e.code === 'Comma' && paused) adjustMusicVolume(-1);
    if (e.code === 'Period' && paused) adjustMusicVolume(1);
    if (e.code === 'KeyB') { if (typeof nnAgent !== 'undefined') nnAgent.toggle(); }
});

// Pause
function togglePause() {
    paused = !paused;
    document.getElementById('pauseBtn').innerHTML = paused ? '&#9654;' : '&#9646;&#9646;';
}

function restartLevel() {
    if (!paused) return;
    paused = false;
    document.getElementById('pauseBtn').innerHTML = '&#9646;&#9646;';
    initLevel(currentLevel);
}

document.getElementById('pauseBtn').addEventListener('touchstart', (e) => {
    e.preventDefault();
    e.stopPropagation();
    togglePause();
}, { passive: false });

document.getElementById('pauseBtn').addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    togglePause();
});

// Click on pause menu buttons (desktop)
canvas.addEventListener('click', (e) => {
    if (paused) {
        const rect = canvas.getBoundingClientRect();
        const clickX = (e.clientX - rect.left) / rect.width * W;
        const clickY = (e.clientY - rect.top) / rect.height * H;
        if (_skipBtnBounds) {
            const b = _skipBtnBounds;
            if (clickX >= b.x && clickX <= b.x + b.w && clickY >= b.y && clickY <= b.y + b.h) {
                restartLevel();
                return;
            }
        }
        if (_volDownBounds) {
            const b = _volDownBounds;
            if (clickX >= b.x && clickX <= b.x + b.w && clickY >= b.y && clickY <= b.y + b.h) {
                adjustMusicVolume(-1);
                return;
            }
        }
        if (_volUpBounds) {
            const b = _volUpBounds;
            if (clickX >= b.x && clickX <= b.x + b.w && clickY >= b.y && clickY <= b.y + b.h) {
                adjustMusicVolume(1);
                return;
            }
        }
        if (_sfxBtnBounds) {
            const b = _sfxBtnBounds;
            if (clickX >= b.x && clickX <= b.x + b.w && clickY >= b.y && clickY <= b.y + b.h) {
                toggleSfxMute();
                return;
            }
        }
        // Click anywhere else = resume
        togglePause();
    }
});

// ══════════════════════════════════════
// ── UPDATE ──
// ══════════════════════════════════════

let lastTime = 0;

function update(dt) {
    if (paused || !started) return;

    // Win screen — just tick timer and spawn celebration particles
    if (gameWon) {
        winTimer += dt;
        winParticleTimer -= dt;
        if (winParticleTimer <= 0) {
            winParticleTimer = 0.15;
            const px = Math.random() * W;
            const py = H * 0.3 + Math.random() * H * 0.2;
            const colors = [C.player, C.spring, C.booster, '#FFF0D0', '#FFD080'];
            spawnP(px + cam.x, py + cam.y, colors[Math.floor(Math.random() * colors.length)], 3, 50, 0.8);
        }
        // Allow tap/key to continue to endless mode
        if (winTimer > 2 && jumpBuffered) {
            jumpBuffered = false;
            gameWon = false;
            currentLevel = 13;
            initLevel(currentLevel);
        }
        // Keep particles alive
        for (let i = particles.length - 1; i >= 0; i--) {
            const p = particles[i];
            p.x += p.vx * dt; p.y += p.vy * dt;
            p.vy += 60 * dt; p.life -= dt;
            if (p.life <= 0) particles.splice(i, 1);
        }
        return;
    }

    // Time-slow effect (quake juice)
    if (timeSlowTimer > 0) {
        timeSlowTimer -= dt;
        dt *= 0.3; // slow everything to 30% speed
    }

    // Update encounter labels
    for (let i = encounterLabels.length - 1; i >= 0; i--) {
        encounterLabels[i].timer -= dt;
        encounterLabels[i].y -= 20 * dt; // float upward
        if (encounterLabels[i].timer <= 0) encounterLabels.splice(i, 1);
    }

    if (levelTitleTimer > 0) levelTitleTimer -= dt;

    // Level complete — freeze gameplay, count down to next level
    if (levelComplete) {
        levelTransitionTimer -= dt;
        // Keep particles alive during transition
        for (let i = particles.length - 1; i >= 0; i--) {
            const p = particles[i];
            p.x += p.vx * dt; p.y += p.vy * dt;
            p.vy += 180 * dt; p.life -= dt;
            if (p.life <= 0) particles.splice(i, 1);
        }
        if (flash.a > 0) flash.a -= dt * 2; // slower flash fade
        if (cam.shake > 0) { cam.sx = (Math.random()*2-1)*cam.shake; cam.sy = (Math.random()*2-1)*cam.shake; cam.shake *= (1-6*dt); if (cam.shake < 0.3) cam.shake = 0; }
        if (levelTransitionTimer <= 0) {
            if (currentLevel >= 12 && !player._hasWon) {
                // BEAT THE GAME!
                player._hasWon = true;
                player._highestLevel = Math.max(player._highestLevel || 1, 13);
                player._bestRun = Math.max(player._bestRun || 0, 12);
                saveProgress();
                gameWon = true;
                winTimer = 0;
                winParticleTimer = 0;
                return;
            }
            currentLevel++;
            if (currentLevel > (player._highestLevel || 1)) {
                player._highestLevel = currentLevel;
            }
            player._bestRun = Math.max(player._bestRun || 0, currentLevel - 1);
            saveProgress();
            initLevel(currentLevel);
        }
        return;
    }

    // Particles
    for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i];
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        p.vy += 180 * dt;
        p.life -= dt;
        if (p.life <= 0) particles.splice(i, 1);
    }

    // Flash
    if (flash.a > 0) flash.a -= dt * 4;
    if (quakeFlash > 0) quakeFlash -= dt;

    // Quake invincibility countdown + sparkle trail
    if (player.quakeTimer > 0) {
        player.quakeTimer -= dt;
        if (Math.random() < 0.5) {
            const cx = player.x + getShapeCenterX(player.shape, player.rotation);
            const cy = player.y + getShapeCenterY(player.shape, player.rotation);
            spawnP(cx + (Math.random() - 0.5) * 24, cy + (Math.random() - 0.5) * 24,
                Math.random() < 0.5 ? '#FFD080' : C.player, 1, 25, 0.4);
        }
    }

    // Falling group
    updateFallingGroup(dt);

    // Clear animations
    updatePendingClears(dt);

    // Camera — use bounding box center of shape
    const shapeCX = player.x + getShapeCenterX(player.shape, player.rotation);
    const targetX = shapeCX - W / 2 + 40;
    cam.x += (targetX - cam.x) * 6 * dt;
    cam.y = 0;
    if (cam.shake > 0) {
        cam.sx = (Math.random() * 2 - 1) * cam.shake;
        cam.sy = (Math.random() * 2 - 1) * cam.shake;
        cam.shake *= (1 - 10 * dt);
        if (cam.shake < 0.3) { cam.shake = 0; cam.sx = 0; cam.sy = 0; }
    }

    if (player.dead) {
        player.deathTimer += dt;
        if (player.deathTimer > 0.5) respawn();
        return;
    }

    // NN agent: inject actions when in bot mode (rate-limited to ~8 actions/sec)
    if (typeof nnAgent !== 'undefined' && nnAgent.active) {
        if (!nnAgent._cooldown) nnAgent._cooldown = 0;
        nnAgent._cooldown -= dt;
        if (nnAgent._cooldown <= 0) {
            const act = nnAgent.getAction();
            if (act === 1) jumpBuffered = true;
            else if (act === 2) tryRotate(1);
            else if (act === 3) tryRotate(-1);
            else if (act === 4) die();
            nnAgent._cooldown = 0.12; // ~8 decisions per second
        }
    }

    // Squash lerp
    player.squash += (1 - player.squash) * 10 * dt;

    // Trail
    player.trail.unshift({ x: player.x, y: player.y, life: 0.12 });
    for (let i = player.trail.length - 1; i >= 0; i--) {
        player.trail[i].life -= dt;
        if (player.trail[i].life <= 0) player.trail.splice(i, 1);
    }
    while (player.trail.length > 4) player.trail.pop();

    // Boost timer
    if (player.boostTimer > 0) player.boostTimer -= dt;
    if (player.spinCooldown > 0) player.spinCooldown -= dt;
    if (player.rotateCooldown > 0) player.rotateCooldown -= dt;

    // Auto-run (frozen during tutorial step 0)
    if (tutorialStep === 0) {
        player.vx = 0;
    } else if (player.boostTimer <= 0) {
        player.vx = RUN_SPEED;
        player.facing = 1;
    }

    // Gravity
    player.vy += GRAVITY * dt;
    if (player.vy > 400) player.vy = 400;

    // Wall detection
    player.onWall = 0;
    if (!player.onGround) {
        // Check right side of bounding box
        const bounds = shapeBounds(player.shape, player.rotation);
        const rightEdge = player.x + (bounds.maxC + 1) * TILE;
        // Check if any cell has solid to its right
        const cells = getCells(player.shape, player.rotation);
        for (const [cx, cy] of cells) {
            const checkX = player.x + (cx + 1) * TILE;
            const checkY = player.y + cy * TILE;
            if (tileIsSolid(Math.floor(checkX / TILE), Math.floor((checkY + TILE/2) / TILE))) {
                player.onWall = 1;
                break;
            }
        }
    }

    // Wall slide
    if (player.onWall !== 0 && player.vy > 0) {
        player.vy = Math.min(player.vy, WALL_SLIDE_SPEED);
        if (Math.random() < 0.3) {
            const bounds = shapeBounds(player.shape, player.rotation);
            spawnP(player.x + (bounds.maxC + 1) * TILE, player.y + bounds.h * TILE / 2, C.tile, 1, 15, 0.2);
        }
    }

    // Coyote time
    if (player.onGround) {
        player.coyoteTimer = 0.06;
        player.jumpsLeft = 2;
    } else {
        player.coyoteTimer -= dt;
    }

    // Jump
    if (jumpBuffered) {
        jumpBuffered = false;
        if (player.onWall !== 0) {
            player.vx = -player.onWall * 180;
            player.vy = JUMP_FORCE * 0.9;
            player.facing = -player.onWall;
            player.jumpsLeft = 1;
            player.boostTimer = 0.15;
            player.squash = 1.3;
            const bounds = shapeBounds(player.shape, player.rotation);
            spawnP(player.x + (bounds.maxC + 1) * TILE, player.y + bounds.h * TILE / 2, C.tile, 4, 40, 0.2);
            SFX.jump();
        } else if (player.onGround || player.coyoteTimer > 0) {
            player.vy = JUMP_FORCE;
            player.onGround = false;
            player.coyoteTimer = 0;
            player.jumpsLeft = 1;
            player.squash = 1.3;
            const cx = player.x + getShapeCenterX(player.shape, player.rotation);
            const bounds = shapeBounds(player.shape, player.rotation);
            spawnP(cx, player.y + (bounds.maxR + 1) * TILE, C.player, 3, 25, 0.15);
            SFX.jump();
        } else if (player.jumpsLeft > 0) {
            player.vy = JUMP_FORCE * 0.7;
            player.jumpsLeft = 0;
            player.squash = 1.2;
            const cx = player.x + getShapeCenterX(player.shape, player.rotation);
            const bounds = shapeBounds(player.shape, player.rotation);
            spawnP(cx, player.y + (bounds.maxR + 1) * TILE, 'rgba(232,168,56,0.5)', 4, 35, 0.2);
            SFX.doubleJump();
        }
    }

    // Move X
    const wasInAir = !player.onGround;
    let hitWallX = false;
    const dx = player.vx * dt;
    if (dx !== 0) {
        const sign = dx > 0 ? 1 : -1;
        const steps = Math.ceil(Math.abs(dx));
        for (let i = 0; i < steps; i++) {
            const step = Math.min(1, Math.abs(dx) - i) * sign;
            if (!tetSolid(player.x + step, player.y, player.shape, player.rotation)) {
                player.x += step;
            } else {
                hitWallX = true;
                player.vx = 0;
                break;
            }
        }
    }

    // Booster: hit from side during X movement
    if (hitWallX) {
        const dir = dx > 0 ? 'right' : 'left';
        const bh = checkAdjacentCorpse(player.x, player.y, player.shape, player.rotation, dir, 'booster');
        if (bh) {
            const now = performance.now() / 1000;
            const key = `${bh.tx},${bh.ty}`;
            if (!player._lastBoost || player._lastBoost.key !== key || (now - player._lastBoost.time) > 0.5) {
                player._lastBoost = { key, time: now };
                player.vx = (dx > 0 ? 1 : -1) * 350;
                player.vy = JUMP_FORCE * 0.5;
                player.jumpsLeft = 1;
                player.boostTimer = 0.25;
                player.squash = 1.5;
                cam.shake = Math.max(cam.shake, 4);
                spawnP(bh.tx * TILE + TILE/2, bh.ty * TILE + TILE/2, C.booster, 7, 70, 0.3);
                SFX.booster();
                if (!player._hitBooster) { player._hitBooster = true; encounterLabel('BOOST!', bh.tx * TILE + TILE/2, bh.ty * TILE + TILE/2, C.booster); }
            }
        }
    }

    // Move Y
    const prevVY = player.vy;
    player.onGround = false;
    let hitCeiling = false;
    const dy = player.vy * dt;
    if (dy !== 0) {
        const sign = dy > 0 ? 1 : -1;
        const steps = Math.ceil(Math.abs(dy));
        for (let i = 0; i < steps; i++) {
            const step = Math.min(1, Math.abs(dy) - i) * sign;
            if (!tetSolid(player.x, player.y + step, player.shape, player.rotation)) {
                player.y += step;
            } else {
                if (player.vy > 0) {
                    player.onGround = true;
                    if (wasInAir && player.vy > 80) {
                        player.squash = 0.6;
                        const cx = player.x + getShapeCenterX(player.shape, player.rotation);
                        const bounds = shapeBounds(player.shape, player.rotation);
                        spawnP(cx, player.y + (bounds.maxR + 1) * TILE, C.player, 3, 30, 0.2);
                        if (player.vy > 250) cam.shake = Math.max(cam.shake, 2);
                        SFX.land();
                    }
                } else {
                    hitCeiling = true;
                }
                player.vy = 0;
                break;
            }
        }
    }
    if (tetSolid(player.x, player.y + 1, player.shape, player.rotation)) player.onGround = true;

    // ── Corpse block interactions (at collision boundaries) ──

    // SPRING: landed on top → bounce UP
    if (player.onGround && prevVY > 0) {
        const sh = checkAdjacentCorpse(player.x, player.y, player.shape, player.rotation, 'below', 'spring');
        if (sh) {
            player.vy = JUMP_FORCE * 1.5;
            player.onGround = false;
            player.jumpsLeft = 1;
            player.squash = 1.4;
            cam.shake = Math.max(cam.shake, 3);
            spawnP(sh.tx * TILE + TILE/2, sh.ty * TILE, C.spring, 5, 60, 0.3);
            SFX.spring();
            if (!player._hitSpring) { player._hitSpring = true; encounterLabel('BOUNCE!', sh.tx * TILE + TILE/2, sh.ty * TILE, C.spring); }
        }
    }

    // SPRING: hit from below → bounce DOWN
    if (hitCeiling && prevVY < 0) {
        const sh = checkAdjacentCorpse(player.x, player.y, player.shape, player.rotation, 'above', 'spring');
        if (sh) {
            player.vy = Math.abs(JUMP_FORCE) * 1.2; // positive = downward
            player.squash = 0.7;
            cam.shake = Math.max(cam.shake, 2);
            spawnP(sh.tx * TILE + TILE/2, (sh.ty + 1) * TILE, C.spring, 4, 50, 0.25);
            SFX.spring();
        }
    }

    // BOOSTER: standing on top → accelerate forward
    if (player.onGround) {
        const bh = checkAdjacentCorpse(player.x, player.y, player.shape, player.rotation, 'below', 'booster');
        if (bh) {
            const now = performance.now() / 1000;
            const key = `${bh.tx},${bh.ty}`;
            if (!player._lastBoost || player._lastBoost.key !== key || (now - player._lastBoost.time) > 0.5) {
                player._lastBoost = { key, time: now };
                player.vx = player.facing * 350;
                player.vy = JUMP_FORCE * 0.3;
                player.jumpsLeft = 1;
                player.boostTimer = 0.25;
                player.squash = 1.5;
                cam.shake = Math.max(cam.shake, 4);
                spawnP(bh.tx * TILE + TILE/2, bh.ty * TILE + TILE/2, C.booster, 7, 70, 0.3);
                SFX.booster();
                if (!player._hitBooster) { player._hitBooster = true; encounterLabel('BOOST!', bh.tx * TILE + TILE/2, bh.ty * TILE + TILE/2, C.booster); }
            }
        }
    }

    // Stuck detection — Monte Carlo look-ahead
    // Uses hitWallX from actual movement (not re-check) + escape simulation
    // If no escape route exists → die. If escape exists → wait forever (timing buzzsaws etc.)
    if (hitWallX && dx > 0 && !canEscapeRight()) {
        player.stuckTimer += dt;
        if (player.stuckTimer > 0.5) {
            die();
            return;
        }
    } else {
        player.stuckTimer = 0;
    }

    // Void death (auto-bounce when invincible)
    if (player.y > levelH * TILE + 20) {
        if (player.quakeTimer > 0) {
            // Bounce back up from chasm bottom
            player.vy = -420;
            player.y = levelH * TILE - TILE;
            player.grounded = false;
            player.jumpsLeft = 2;
            cam.shake = Math.max(cam.shake, 6);
            SFX.spring();
        } else {
            player.chasmDeaths++;
            die();
            return;
        }
    }

    // Update moving platforms + carry player
    for (const p of movingPlatforms) {
        const prevX = p.x, prevY = p.y;
        const t = performance.now() / 1000;
        if (p.vertical) {
            p.y = p.baseY + Math.sin(t * p.speed * 0.05 + p.phase) * p.range;
        } else {
            p.x = p.baseX + Math.sin(t * p.speed * 0.05 + p.phase) * p.range;
        }
        // Carry player if standing on this platform
        if (!player.dead) {
            const cells = getCells(player.shape, player.rotation);
            for (const [cx, cy] of cells) {
                const footTX = Math.floor((player.x + cx * TILE + TILE/2) / TILE);
                const footTY = Math.floor((player.y + (cy + 1) * TILE) / TILE);
                const px1 = Math.floor(prevX / TILE);
                const ppy = Math.floor(prevY / TILE);
                if (footTY === ppy && footTX >= px1 && footTX < px1 + p.w) {
                    player.x += (p.x - prevX);
                    player.y += (p.y - prevY);
                    break;
                }
            }
        }
    }

    // Update buzzsaws
    for (const saw of buzzsaws) {
        saw.spin += dt * 8;
        const t = performance.now() / 1000;
        if (saw.vertical) {
            saw.y = saw.baseY + Math.sin(t * saw.speed * 0.05 + saw.phase) * saw.range;
        } else {
            saw.x = saw.baseX + Math.sin(t * saw.speed * 0.05 + saw.phase) * saw.range;
        }
    }

    // Spike check (skipped during quake invincibility)
    if (player.quakeTimer <= 0 && tetSpike(player.x, player.y, player.shape, player.rotation)) {
        die();
        return;
    }

    // Buzzsaw collision (skipped during quake invincibility)
    if (player.quakeTimer <= 0) {
        const sawCells = getCells(player.shape, player.rotation);
        for (const saw of buzzsaws) {
            for (const [cx, cy] of sawCells) {
                const cellX = player.x + cx * TILE;
                const cellY = player.y + cy * TILE;
                const nearX = Math.max(cellX, Math.min(saw.x, cellX + TILE));
                const nearY = Math.max(cellY, Math.min(saw.y, cellY + TILE));
                const ddx = nearX - saw.x;
                const ddy = nearY - saw.y;
                if (ddx * ddx + ddy * ddy < saw.r * saw.r) {
                    die();
                    return;
                }
            }
        }
    }

    // Track distance
    player.distance = Math.max(player.distance, Math.floor(player.x / TILE));
    if (player.distance > player.bestDistance) player.bestDistance = player.distance;

    // Goal detection — any cell crosses the finish line?
    const bounds = shapeBounds(player.shape, player.rotation);
    const rightmostX = player.x + (bounds.maxC + 1) * TILE;
    if (!levelComplete && rightmostX >= goalX) {
        levelComplete = true;
        levelTransitionTimer = 3.0;
        cam.shake = Math.max(cam.shake, 10);
        flash.a = 1.0; flash.r = 0.91; flash.g = 0.84; flash.b = 0.42;
        // Big celebration particles
        const cx = player.x + getShapeCenterX(player.shape, player.rotation);
        const cy = player.y + getShapeCenterY(player.shape, player.rotation);
        spawnP(cx, cy, C.player, 30, 120, 0.6);
        spawnP(cx, cy, '#FFF0D0', 20, 80, 0.5);
        if (currentLevel >= 12) {
            SFX.win();
        } else {
            SFX.levelComplete();
        }
    }
}


// ══════════════════════════════════════════════════════════════
// ── SEEDED PRNG ──
// ══════════════════════════════════════════════════════════════

function mulberry32(a) {
    return function() {
        a |= 0;
        a = a + 0x6D2B79F5 | 0;
        var t = Math.imul(a ^ a >>> 15, 1 | a);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}


// ══════════════════════════════════════════════════════════════
// ── SMALL NN (72 → 48 → 32 → 5, ReLU, argmax) ──
// ══════════════════════════════════════════════════════════════

class SmallNN {
    constructor(flat) {
        // Architecture: 72 input → 48 hidden → 32 hidden → 5 output
        // Flat layout: w1(48*72), b1(48), w2(32*48), b2(32), w3(5*32), b3(5)
        // Total: 3456 + 48 + 1536 + 32 + 160 + 5 = 5237
        if (!flat || flat.length < 5237) {
            this.valid = false;
            return;
        }
        this.valid = true;
        let idx = 0;

        // Layer 1: 48 x 72
        this.w1 = [];
        for (let i = 0; i < 48; i++) {
            this.w1.push(new Float32Array(flat.slice(idx, idx + 72)));
            idx += 72;
        }
        this.b1 = new Float32Array(flat.slice(idx, idx + 48));
        idx += 48;

        // Layer 2: 32 x 48
        this.w2 = [];
        for (let i = 0; i < 32; i++) {
            this.w2.push(new Float32Array(flat.slice(idx, idx + 48)));
            idx += 48;
        }
        this.b2 = new Float32Array(flat.slice(idx, idx + 32));
        idx += 32;

        // Layer 3: 5 x 32
        this.w3 = [];
        for (let i = 0; i < 5; i++) {
            this.w3.push(new Float32Array(flat.slice(idx, idx + 32)));
            idx += 32;
        }
        this.b3 = new Float32Array(flat.slice(idx, idx + 5));
    }

    forward(inputs) {
        if (!this.valid) return 0;

        // Layer 1: ReLU(w1 * x + b1)
        const h1 = new Float32Array(48);
        for (let i = 0; i < 48; i++) {
            let sum = this.b1[i];
            const row = this.w1[i];
            for (let j = 0; j < 72; j++) sum += row[j] * inputs[j];
            h1[i] = sum > 0 ? sum : 0; // ReLU
        }

        // Layer 2: ReLU(w2 * h1 + b2)
        const h2 = new Float32Array(32);
        for (let i = 0; i < 32; i++) {
            let sum = this.b2[i];
            const row = this.w2[i];
            for (let j = 0; j < 48; j++) sum += row[j] * h1[j];
            h2[i] = sum > 0 ? sum : 0; // ReLU
        }

        // Layer 3: w3 * h2 + b3 (no activation — raw logits)
        const out = new Float32Array(5);
        for (let i = 0; i < 5; i++) {
            let sum = this.b3[i];
            const row = this.w3[i];
            for (let j = 0; j < 32; j++) sum += row[j] * h2[j];
            out[i] = sum;
        }

        // Argmax
        let best = 0;
        for (let i = 1; i < 5; i++) {
            if (out[i] > out[best]) best = i;
        }
        return best;
    }
}


// ══════════════════════════════════════════════════════════════
// ── NN INPUT EXTRACTION (from nn_agent.js, adapted for Node) ──
// ══════════════════════════════════════════════════════════════

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

    // D. Terrain lookahead: 10 columns x 4 features (40)
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


// ══════════════════════════════════════════════════════════════
// ── AGENT EVALUATOR ──
// ══════════════════════════════════════════════════════════════

var _evalNoCorpses = false;
function evaluateAgent(flatWeights, levelNum, maxTime, seed, noCorpses) {
    _evalNoCorpses = !!noCorpses;
    // Install seeded PRNG
    var prng = mulberry32(seed);
    Math.random = prng;

    // Reset simulation time
    _simTime = 0;

    // Reset all game state
    shapeBag = [];
    gameWon = false;
    winTimer = 0;
    winParticleTimer = 0;
    fallingGroup = null;
    pendingClears = [];
    encounterLabels = [];
    particles = [];
    timeSlowTimer = 0;
    nextPieceShape = null;
    nextPieceMaterial = null;

    // Reset player stats
    player.deathCount = 0;
    player.chasmDeaths = 0;
    player.quakeCount = 0;
    player.quakeTimer = 0;
    player.bestDistance = 0;
    player._hitSpring = false;
    player._hitBooster = false;
    player._lastBoost = null;
    player._hasRun = false;
    player._totalDeaths = 0;
    player._totalQuakes = 0;
    player._highestLevel = 1;
    player._bestRun = 0;
    player._hasWon = false;

    // Initialize level
    currentLevel = levelNum;
    initLevel(currentLevel);

    // Start the game
    started = true;
    paused = false;
    tutorialStep = 3;
    tutorialComplete = true;
    jumpBuffered = false;
    levelComplete = false;

    // Create NN
    var nn = new SmallNN(flatWeights);

    var dt = 1 / 60;
    var elapsed = 0;
    var decisionTimer = 0;
    var jumpCooldown = 0;      // cooldown after a successful jump
    var wasOnGround = false;   // track landing to reset cooldown
    var groundedFrames = 0;    // frames spent on ground (for ground ratio)
    var totalFrames = 0;       // total simulation frames
    var startLevel = currentLevel;
    var stagnantTimer = 0;     // time since last forward progress
    var lastProgressX = 0;     // x position at last progress check

    while (elapsed < maxTime) {
        _simTime = elapsed;

        // Track ground time for ground ratio fitness
        totalFrames++;
        if (player.onGround && !player.dead) groundedFrames++;

        // Stagnation: 0.25s without forward progress = death
        // Forces agents to immediately rotate/jump when hooked, or die
        if (!player.dead) {
            if (player.x > lastProgressX + TILE) {
                lastProgressX = player.x;
                stagnantTimer = 0;
            } else {
                stagnantTimer += dt;
                if (stagnantTimer > 2.0) {
                    die();
                    stagnantTimer = 0;
                }
            }
        } else {
            stagnantTimer = 0;
            lastProgressX = player.x;
        }

        // NN decision (every ~0.12s, ~8 decisions/sec)
        if (!player.dead && !levelComplete && !gameWon) {
            decisionTimer -= dt;
            if (decisionTimer <= 0) {
                decisionTimer = 0.12;
                var action;
                // Stuck detector: force decisions when not progressing
                if (stagnantTimer > 0.8) {
                    action = 4; // phase 3: die and retry
                } else if (stagnantTimer > 0.5) {
                    action = 1; // phase 2: jump out
                } else if (stagnantTimer > 0.3) {
                    action = 2; // phase 1: spin to change shape
                } else {
                    var inputs = getInputs();
                    action = nn.forward(inputs);
                }
                // Actions: 0=nothing, 1=jump, 2=rotate_cw, 3=rotate_ccw, 4=die
                if (action === 1) {
                    jumpBuffered = true;
                }
                else if (action === 2) tryRotate(1);
                else if (action === 3) tryRotate(-1);
                else if (action === 4) die();
            }
        }

        update(dt);
        elapsed += dt;

        // Check if game won
        if (gameWon) break;

        // Death cap: eval ends after 20 deaths (forces strategic dying)
        if (player.deathCount >= 20) break;
    }

    var levelsCompleted = currentLevel - startLevel + (levelComplete ? 1 : 0);
    var fitness = player.bestDistance + 500 * levelsCompleted;

    return {
        fitness: fitness,
        completed: levelComplete || gameWon,
        distance: player.bestDistance,
        deaths: player.deathCount
    };
}


// ══════════════════════════════════════════════════════════════
// ── BATCH EVALUATION (stdin/stdout JSON) ──
// ══════════════════════════════════════════════════════════════

// Read JSON from stdin, evaluate all agents, output results to stdout
// Input:  { weights: [[...], [...]], levelNum: 1, maxTime: 30, seeds: [42, 43] }
// Output: [ {fitness, completed, distance, deaths}, ... ]

var inputChunks = [];
process.stdin.setEncoding('utf8');
process.stdin.on('data', function(chunk) { inputChunks.push(chunk); });
process.stdin.on('end', function() {
    var input;
    try {
        input = JSON.parse(inputChunks.join(''));
    } catch (e) {
        process.stderr.write('ERROR: Invalid JSON input: ' + e.message + '\n');
        process.exit(1);
    }

    var weightsArray = input.weights || [[]];
    var levelNum = input.levelNum || 1;
    var maxTime = input.maxTime || 30;
    var seeds = input.seeds || weightsArray.map(function(_, i) { return 42 + i; });
    var noCorpses = input.noCorpses || false;

    // Save original Math.random
    var _origRandom = Math.random;

    var _noCorpses = noCorpses;  // expose to evaluateAgent scope

    var results = [];
    for (var i = 0; i < weightsArray.length; i++) {
        var result = evaluateAgent(weightsArray[i], levelNum, maxTime, seeds[i], _noCorpses);
        results.push(result);
    }

    // Restore Math.random
    Math.random = _origRandom;

    process.stdout.write(JSON.stringify(results) + '\n');
});
