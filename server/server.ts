import type * as Party from "partykit/server";

// Session tracking for metrics
interface PlayerSession {
  id: string;
  joinTime: number;
  leaveTime?: number;
  region?: string;
  country?: string;
}

// Player state
interface Player {
  id: string;
  name: string;
  x: number;
  y: number;
  dir: number; // 0=up, 1=right, 2=down, 3=left
  segments: { x: number; y: number }[];
  color: string;
  score: number;
  team: 'blue' | 'red';
  invincibleUntil: number;
  lastUpdate: number;
}

// Game state for a room
interface GameState {
  players: Map<string, Player>;
  pickups: { x: number; y: number; type: string }[];
  worldSize: { cols: number; rows: number };
  hostId: string | null;
}

const COLORS = ['#f0f', '#0ff', '#ff0', '#f80', '#8f0', '#08f', '#f08', '#80f'];
const WORLD_COLS = 200;
const WORLD_ROWS = 150;

// Spike detection config - alert on ANY non-AZ player for now
const SPIKE_COOLDOWN_MS = 600000; // 10 min cooldown between alerts
const EXCLUDED_REGIONS = ['AZ', 'Arizona']; // Your region

export default class LocomotServer implements Party.Server {
  state: GameState;
  cleanupInterval: ReturnType<typeof setInterval> | null = null;
  sessions: PlayerSession[] = []; // All sessions for metrics
  lastSpikeAlert: number = 0;

  constructor(readonly room: Party.Room) {
    this.state = {
      players: new Map(),
      pickups: this.generatePickups(50),
      worldSize: { cols: WORLD_COLS, rows: WORLD_ROWS },
      hostId: null
    };

    // Clean up stale players every 10 seconds
    this.cleanupInterval = setInterval(() => this.cleanupStalePlayers(), 10000);
  }

  // Load sessions from storage on room start
  async onStart() {
    const stored = await this.room.storage.get<PlayerSession[]>('sessions');
    if (stored) {
      this.sessions = stored;
      console.log(`Loaded ${this.sessions.length} sessions from storage`);
    }
  }

  // Save sessions to storage
  async saveSessions() {
    await this.room.storage.put('sessions', this.sessions);
  }

  // Send email notification via formsubmit.co
  async sendPlayerAlert(region: string, country: string, playerCount: number) {
    const now = Date.now();
    if (now - this.lastSpikeAlert < SPIKE_COOLDOWN_MS) {
      console.log('Alert on cooldown, skipping');
      return;
    }

    this.lastSpikeAlert = now;
    const message = `🚂 LOCOMOT.IO New Player!\n\nSomeone from ${region}, ${country} is playing!\nTotal players: ${playerCount}\nRoom: ${this.room.id}\nTime: ${new Date().toISOString()}`;

    try {
      await fetch("https://formsubmit.co/ajax/savecharlie@gmail.com", {
        method: "POST",
        headers: { "Content-Type": "application/json", "Accept": "application/json" },
        body: JSON.stringify({
          message: message,
          _subject: `🚂 LOCOMOT.IO: Player from ${region}!`
        })
      });
      console.log(`Alert sent: player from ${region}, ${country}`);
    } catch (e) {
      console.error('Failed to send alert:', e);
    }
  }

  // Track join with geo info
  async trackJoin(connId: string, region?: string, country?: string) {
    const session: PlayerSession = {
      id: connId,
      joinTime: Date.now(),
      region: region || 'unknown',
      country: country || 'unknown'
    };
    this.sessions.push(session);

    // Keep last 1000 sessions max
    if (this.sessions.length > 1000) {
      this.sessions = this.sessions.slice(-1000);
    }

    // Save to storage
    await this.saveSessions();

    // Alert if non-AZ player
    const isExcluded = EXCLUDED_REGIONS.some(r =>
      region?.toUpperCase().includes(r.toUpperCase())
    );

    if (!isExcluded) {
      console.log(`Non-AZ player joined from ${region}, ${country}`);
      this.sendPlayerAlert(region || 'unknown', country || 'unknown', this.state.players.size);
    } else {
      console.log(`AZ player joined (excluded from alerts)`);
    }
  }

  // Track leave
  async trackLeave(connId: string) {
    const session = this.sessions.find(s => s.id === connId && !s.leaveTime);
    if (session) {
      session.leaveTime = Date.now();
      await this.saveSessions();
    }
  }

  // Get metrics data
  getMetrics() {
    const now = Date.now();
    const day = 24 * 60 * 60 * 1000;
    const hour = 60 * 60 * 1000;

    // Filter out AZ sessions for stats
    const nonAzSessions = this.sessions.filter(s =>
      !EXCLUDED_REGIONS.some(r => s.region?.toUpperCase().includes(r.toUpperCase()))
    );

    const last24h = nonAzSessions.filter(s => now - s.joinTime < day);
    const lastHour = nonAzSessions.filter(s => now - s.joinTime < hour);

    // Group by region
    const byRegion: Record<string, number> = {};
    for (const s of nonAzSessions) {
      const key = s.region || 'unknown';
      byRegion[key] = (byRegion[key] || 0) + 1;
    }

    // Group by day for trend
    const byDay: Record<string, number> = {};
    for (const s of nonAzSessions) {
      const date = new Date(s.joinTime).toISOString().split('T')[0];
      byDay[date] = (byDay[date] || 0) + 1;
    }

    // Average session duration
    const completedSessions = nonAzSessions.filter(s => s.leaveTime);
    const avgDuration = completedSessions.length > 0
      ? completedSessions.reduce((sum, s) => sum + (s.leaveTime! - s.joinTime), 0) / completedSessions.length
      : 0;

    return {
      currentPlayers: this.state.players.size,
      totalSessions: nonAzSessions.length,
      last24h: last24h.length,
      lastHour: lastHour.length,
      avgSessionMinutes: Math.round(avgDuration / 60000),
      byRegion: Object.entries(byRegion).sort((a, b) => b[1] - a[1]),
      byDay: Object.entries(byDay).sort((a, b) => a[0].localeCompare(b[0])),
      recentSessions: nonAzSessions.slice(-20).reverse().map(s => ({
        region: s.region,
        country: s.country,
        joinTime: new Date(s.joinTime).toISOString(),
        durationMinutes: s.leaveTime ? Math.round((s.leaveTime - s.joinTime) / 60000) : 'active'
      }))
    };
  }

  // Remove players who haven't sent an update in 30 seconds
  cleanupStalePlayers() {
    const now = Date.now();
    const staleTimeout = 30000; // 30 seconds
    const toRemove: string[] = [];

    for (const [id, player] of this.state.players) {
      if (now - player.lastUpdate > staleTimeout) {
        toRemove.push(id);
        console.log(`Cleaning up stale player: ${player.name} (${id}) - no update for ${Math.floor((now - player.lastUpdate) / 1000)}s`);
      }
    }

    for (const id of toRemove) {
      this.state.players.delete(id);
      this.room.broadcast(JSON.stringify({
        type: 'player_left',
        playerId: id
      }));
    }

    this.ensureHost();

    if (toRemove.length > 0) {
      console.log(`Cleaned up ${toRemove.length} stale player(s). Remaining: ${this.state.players.size}`);
    }
  }

  assignHost(newHostId: string | null) {
    if (this.state.hostId === newHostId) return;
    this.state.hostId = newHostId;
    if (newHostId) {
      this.room.broadcast(JSON.stringify({
        type: 'host_assigned',
        hostId: newHostId
      }));
    }
  }

  ensureHost() {
    if (this.state.hostId && this.state.players.has(this.state.hostId)) return;
    const remainingIds = Array.from(this.state.players.keys()).sort();
    const nextHost = remainingIds.length > 0 ? remainingIds[0] : null;
    this.assignHost(nextHost);
  }

  generatePickups(count: number) {
    const pickups = [];
    for (let i = 0; i < count; i++) {
      pickups.push({
        x: Math.floor(Math.random() * WORLD_COLS),
        y: Math.floor(Math.random() * WORLD_ROWS),
        type: 'food'
      });
    }
    return pickups;
  }

  getSpawnPosition(): { x: number; y: number } {
    // Find a safe spawn away from other players
    let x, y, attempts = 0;
    do {
      x = Math.floor(Math.random() * (WORLD_COLS - 20)) + 10;
      y = Math.floor(Math.random() * (WORLD_ROWS - 20)) + 10;
      attempts++;

      let safe = true;
      for (const player of this.state.players.values()) {
        const dist = Math.abs(player.x - x) + Math.abs(player.y - y);
        if (dist < 20) safe = false;
      }
      if (safe || attempts > 50) break;
    } while (true);

    return { x, y };
  }

  getNextTeam(): 'blue' | 'red' {
    let blue = 0;
    let red = 0;
    for (const p of this.state.players.values()) {
      if (p.team === 'blue') blue++;
      else if (p.team === 'red') red++;
    }
    return blue <= red ? 'blue' : 'red';
  }

  onConnect(conn: Party.Connection, ctx: Party.ConnectionContext) {
    const spawn = this.getSpawnPosition();
    // "All Same Gun" - everyone starts with MACHINEGUN color (orange)
    const color = '#f80';
    const team = this.getNextTeam();

    const player: Player = {
      id: conn.id,
      name: `Train${this.state.players.size + 1}`,
      x: spawn.x,
      y: spawn.y,
      dir: 1, // Start facing right
      segments: [
        { x: spawn.x, y: spawn.y },
        { x: spawn.x - 1, y: spawn.y },
        { x: spawn.x - 2, y: spawn.y },
        { x: spawn.x - 3, y: spawn.y }
      ],
      color,
      score: 0,
      team,
      invincibleUntil: 0,
      lastUpdate: Date.now()
    };

    this.state.players.set(conn.id, player);
    this.ensureHost();

    // Extract geo info from Cloudflare headers
    const cf = (ctx.request as any).cf;
    const region = cf?.region || cf?.regionCode || 'unknown';
    const country = cf?.country || 'unknown';

    // Track join for metrics and alerts
    this.trackJoin(conn.id, region, country);

    // Send initial state to new player
    conn.send(JSON.stringify({
      type: 'init',
      playerId: conn.id,
      player,
      players: Array.from(this.state.players.values()),
      pickups: this.state.pickups,
      worldSize: this.state.worldSize,
      hostId: this.state.hostId
    }));

    // Notify others of new player
    this.room.broadcast(JSON.stringify({
      type: 'player_joined',
      player
    }), [conn.id]);

    console.log(`Player ${conn.id} joined room ${this.room.id}. Total: ${this.state.players.size}`);
  }

  onClose(conn: Party.Connection) {
    const player = this.state.players.get(conn.id);
    this.state.players.delete(conn.id);

    // Track session end
    this.trackLeave(conn.id);

    // Notify others
    this.room.broadcast(JSON.stringify({
      type: 'player_left',
      playerId: conn.id
    }));

    this.ensureHost();

    console.log(`Player ${conn.id} left. Total: ${this.state.players.size}`);
  }

  onMessage(message: string, sender: Party.Connection) {
    try {
      const data = JSON.parse(message);
      const player = this.state.players.get(sender.id);
      if (!player) return;

      switch (data.type) {
        case 'move':
          // Update player direction
          player.dir = data.dir;
          player.lastUpdate = Date.now();
          break;

        case 'update':
          // Client sends their full state (client-authoritative for now)
          player.x = data.x;
          player.y = data.y;
          player.segments = data.segments;
          player.score = data.score;
          if (data.color) player.color = data.color; // "All Same Gun" - sync gun color
          if (typeof data.invincibleUntil === 'number') {
            player.invincibleUntil = data.invincibleUntil;
          }
          player.lastUpdate = Date.now();
          break;

        case 'name':
          player.name = data.name.slice(0, 12);
          break;

        case 'hit':
          // Validate hit - check proximity before broadcasting
          const attacker = this.state.players.get(sender.id);
          const target = this.state.players.get(data.targetId);
          if (attacker && target && attacker.segments && target.segments) {
            const ax = attacker.x, ay = attacker.y;
            const tx = target.x, ty = target.y;
            const dist = Math.abs(ax - tx) + Math.abs(ay - ty);
            // Only allow hits within reasonable range (15 tiles)
            if (dist < 15) {
              this.room.broadcast(JSON.stringify({
                type: 'hit',
                targetId: data.targetId,
                damage: Math.min(data.damage, 50), // Cap damage
                fromId: sender.id
              }));
            }
          }
          break;

        case 'kill':
          const killFromId = (sender.id === this.state.hostId && typeof data.fromId === 'string')
            ? data.fromId
            : sender.id;
          {
            const target = this.state.players.get(data.targetId);
            if (target) {
              target.segments = [];
            }
          }
          this.room.broadcast(JSON.stringify({
            type: 'kill',
            targetId: data.targetId,
            fromId: killFromId
          }));
          break;

        case 'pickup_collected':
          this.room.broadcast(JSON.stringify({
            type: 'pickup_collected',
            pickupId: data.pickupId,
            x: data.x,
            y: data.y
          }));
          break;

        case 'arena_sync':
          // Forward arena sync to the target player (for syncing AI enemies)
          for (const [id, conn] of this.room.getConnections()) {
            if (id === data.targetId) {
              conn.send(JSON.stringify({
                type: 'arena_sync',
                enemies: data.enemies,
                pickups: data.pickups
              }));
              console.log(`Arena sync forwarded from ${sender.id} to ${data.targetId}`);
              break;
            }
          }
          break;

        case 'request_arena':
          // New player requesting arena state - broadcast to all others
          this.ensureHost();
          this.room.broadcast(JSON.stringify({
            type: 'arena_request',
            fromId: sender.id
          }), [sender.id]); // Exclude requester
          console.log(`Arena request from ${sender.id} broadcast to others`);
          break;

        case 'enemy_state':
          // Host broadcasting enemy state + pickups - relay to all other clients
          player.lastUpdate = Date.now(); // Treat host state broadcast as heartbeat
          this.room.broadcast(JSON.stringify({
            type: 'enemy_state',
            enemies: data.enemies,
            pickups: data.pickups
          }), [sender.id]); // Exclude sender
          break;
      }

      // Broadcast state to all players
      this.room.broadcast(JSON.stringify({
        type: 'state',
        players: Array.from(this.state.players.values()),
        pickups: this.state.pickups
      }));

    } catch (e) {
      console.error('Message parse error:', e);
    }
  }

  // Handle HTTP requests (for training data upload + debug)
  async onRequest(req: Party.Request) {
    // Enable CORS
    const headers = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type'
    };

    if (req.method === 'OPTIONS') {
      return new Response(null, { headers });
    }

    // Check URL path
    const url = new URL(req.url);

    // METRICS PAGE
    if (req.method === 'GET' && url.pathname.endsWith('/metrics')) {
      const metrics = this.getMetrics();
      const html = `<!DOCTYPE html>
<html>
<head>
  <title>LOCOMOT.IO Metrics</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="30">
  <style>
    body { font-family: system-ui, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; max-width: 900px; margin: 0 auto; }
    h1 { color: #0ff; text-align: center; }
    .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin: 20px 0; }
    .stat { background: #16213e; padding: 20px; border-radius: 10px; text-align: center; }
    .stat-value { font-size: 2.5em; color: #0ff; font-weight: bold; }
    .stat-label { color: #888; font-size: 0.9em; }
    table { width: 100%; border-collapse: collapse; margin: 20px 0; }
    th, td { padding: 10px; text-align: left; border-bottom: 1px solid #333; }
    th { color: #0ff; }
    .section { background: #16213e; padding: 20px; border-radius: 10px; margin: 20px 0; }
    h2 { color: #f0f; margin-top: 0; }
    .trend { display: flex; gap: 5px; align-items: flex-end; height: 60px; }
    .bar { background: #0ff; min-width: 20px; border-radius: 3px 3px 0 0; }
    .note { color: #666; font-size: 0.8em; text-align: center; }
  </style>
</head>
<body>
  <h1>🚂 LOCOMOT.IO Metrics</h1>
  <p class="note">Excludes AZ traffic • Auto-refreshes every 30s</p>

  <div class="stats">
    <div class="stat">
      <div class="stat-value">${metrics.currentPlayers}</div>
      <div class="stat-label">Playing Now</div>
    </div>
    <div class="stat">
      <div class="stat-value">${metrics.lastHour}</div>
      <div class="stat-label">Last Hour</div>
    </div>
    <div class="stat">
      <div class="stat-value">${metrics.last24h}</div>
      <div class="stat-label">Last 24h</div>
    </div>
    <div class="stat">
      <div class="stat-value">${metrics.totalSessions}</div>
      <div class="stat-label">Total Sessions</div>
    </div>
    <div class="stat">
      <div class="stat-value">${metrics.avgSessionMinutes}m</div>
      <div class="stat-label">Avg Session</div>
    </div>
  </div>

  <div class="section">
    <h2>📈 Daily Trend</h2>
    ${metrics.byDay.length > 0 ? `
    <div class="trend">
      ${metrics.byDay.slice(-14).map(([date, count]) => {
        const maxCount = Math.max(...metrics.byDay.map(d => d[1] as number));
        const height = Math.max(5, (count as number / maxCount) * 50);
        return `<div class="bar" style="height:${height}px" title="${date}: ${count}"></div>`;
      }).join('')}
    </div>
    <p class="note">Last 14 days</p>
    ` : '<p>No data yet</p>'}
  </div>

  <div class="section">
    <h2>🌍 By Region</h2>
    ${metrics.byRegion.length > 0 ? `
    <table>
      <tr><th>Region</th><th>Sessions</th></tr>
      ${metrics.byRegion.slice(0, 10).map(([region, count]) =>
        `<tr><td>${region}</td><td>${count}</td></tr>`
      ).join('')}
    </table>
    ` : '<p>No data yet</p>'}
  </div>

  <div class="section">
    <h2>🕐 Recent Sessions</h2>
    ${metrics.recentSessions.length > 0 ? `
    <table>
      <tr><th>Time</th><th>Region</th><th>Duration</th></tr>
      ${metrics.recentSessions.map(s =>
        `<tr><td>${new Date(s.joinTime).toLocaleString()}</td><td>${s.region}, ${s.country}</td><td>${s.durationMinutes}${typeof s.durationMinutes === 'number' ? 'm' : ''}</td></tr>`
      ).join('')}
    </table>
    ` : '<p>No sessions yet</p>'}
  </div>
</body>
</html>`;
      return new Response(html, {
        headers: { ...headers, 'Content-Type': 'text/html' }
      });
    }

    // METRICS JSON API
    if (req.method === 'GET' && url.pathname.endsWith('/metrics.json')) {
      return new Response(JSON.stringify(this.getMetrics(), null, 2), {
        headers: { ...headers, 'Content-Type': 'application/json' }
      });
    }

    // DEBUG: GET returns current server state
    if (req.method === 'GET') {
      const state = {
        timestamp: Date.now(),
        playerCount: this.state.players.size,
        enemyCount: (this.state as any).enemies?.length || 0,
        pickupCount: this.state.pickups?.length || 0,
        metrics: this.getMetrics()
      };
      return new Response(JSON.stringify(state, null, 2), {
        headers: { ...headers, 'Content-Type': 'application/json' }
      });
    }

    if (req.method === 'POST') {
      try {
        const data = await req.json() as { type: string; data: unknown };

        if (data.type === 'upload_training') {
          // Store training data (using Partykit storage)
          const key = `training_${Date.now()}_${Math.random().toString(36).slice(2)}`;
          await this.room.storage.put(key, data.data);

          console.log(`Training data uploaded: ${key}`);

          return new Response(JSON.stringify({ success: true, key }), {
            headers: { ...headers, 'Content-Type': 'application/json' }
          });
        }

        if (data.type === 'player_training_data') {
          // Store player behavior data for imitation learning
          const playerData = data.data as { state: number[]; action: number }[];
          const score = (data as any).score || 0;

          // Only store high-quality data (score >= 10)
          if (score >= 10 && Array.isArray(playerData) && playerData.length > 0) {
            const key = `player_${Date.now()}_${score}`;
            await this.room.storage.put(key, playerData);
            console.log(`Player training data: ${playerData.length} frames, score ${score}`);
          }

          return new Response(JSON.stringify({ success: true, frames: playerData?.length || 0 }), {
            headers: { ...headers, 'Content-Type': 'application/json' }
          });
        }

        if (data.type === 'get_player_data') {
          // Retrieve all stored player training data
          const allData: { state: number[]; action: number }[] = [];
          const keys = await this.room.storage.list({ prefix: 'player_' });

          for (const [key, value] of keys) {
            const frames = value as { state: number[]; action: number }[];
            if (Array.isArray(frames)) {
              allData.push(...frames);
            }
          }

          return new Response(JSON.stringify(allData), {
            headers: { ...headers, 'Content-Type': 'application/json' }
          });
        }

        if (data.type === 'get_brain') {
          // Retrieve stored brain weights for training
          const brain = await this.room.storage.get('brain_weights');
          if (brain) {
            return new Response(JSON.stringify(brain), {
              headers: { ...headers, 'Content-Type': 'application/json' }
            });
          }
          return new Response(JSON.stringify(null), {
            headers: { ...headers, 'Content-Type': 'application/json' }
          });
        }

        if (data.type === 'set_brain') {
          // Store brain weights after training
          if (data.brain && data.brain['net.0.weight']) {
            await this.room.storage.put('brain_weights', data.brain);
            return new Response(JSON.stringify({ success: true }), {
              headers: { ...headers, 'Content-Type': 'application/json' }
            });
          }
          return new Response(JSON.stringify({ error: 'Invalid brain data' }), {
            status: 400,
            headers: { ...headers, 'Content-Type': 'application/json' }
          });
        }

        return new Response(JSON.stringify({ error: 'Unknown type' }), {
          status: 400,
          headers: { ...headers, 'Content-Type': 'application/json' }
        });
      } catch (e) {
        console.error('Request error:', e);
        return new Response(JSON.stringify({ error: 'Invalid request' }), {
          status: 400,
          headers: { ...headers, 'Content-Type': 'application/json' }
        });
      }
    }

    return new Response('Method not allowed', { status: 405, headers });
  }
}
