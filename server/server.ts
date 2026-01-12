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
  backupInterval: ReturnType<typeof setInterval> | null = null;
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

    // Backup all data every hour to prevent data loss
    this.backupInterval = setInterval(() => {
      this.saveSessions(); // This now saves to multiple backup keys
      this.backupVisitData();
      console.log('[Backup] Hourly backup completed');
    }, 60 * 60 * 1000);
  }

  // Load sessions from storage on room start - with backup recovery
  async onStart() {
    // Try main storage first
    let stored = await this.room.storage.get<PlayerSession[]>('sessions');

    // If empty, try to recover from backups
    if (!stored || stored.length === 0) {
      console.log('Main sessions empty, checking backups...');

      // Try backup keys in order (most recent first)
      const backupKeys = ['sessions_backup_daily', 'sessions_backup_weekly', 'sessions_backup_permanent'];
      for (const key of backupKeys) {
        const backup = await this.room.storage.get<PlayerSession[]>(key);
        if (backup && backup.length > 0) {
          stored = backup;
          console.log(`Recovered ${backup.length} sessions from ${key}`);
          // Restore to main storage
          await this.room.storage.put('sessions', backup);
          break;
        }
      }
    }

    if (stored && stored.length > 0) {
      this.sessions = stored;
      console.log(`Loaded ${this.sessions.length} sessions from storage`);
    } else {
      console.log('No sessions found in any storage location');
    }

    // Also recover visit_ data if needed
    await this.recoverVisitData();
  }

  // Recover visit data from backup
  async recoverVisitData() {
    const allVisits = await this.room.storage.list({ prefix: 'visit_' });
    if (allVisits.size === 0) {
      console.log('Visit data empty, checking backup...');
      const backup = await this.room.storage.get<any[]>('visits_backup');
      if (backup && backup.length > 0) {
        console.log(`Recovering ${backup.length} visits from backup`);
        for (const visit of backup) {
          const key = `visit_${visit.timestamp}_${Math.random().toString(36).slice(2, 8)}`;
          await this.room.storage.put(key, visit);
        }
      }
    }
  }

  // Save sessions to storage with redundant backups
  async saveSessions() {
    // Save to main storage
    await this.room.storage.put('sessions', this.sessions);

    // Always save to permanent backup (never gets cleared)
    await this.room.storage.put('sessions_backup_permanent', this.sessions);

    // Daily backup (overwritten each day)
    const today = new Date().toISOString().split('T')[0];
    await this.room.storage.put('sessions_backup_daily', this.sessions);
    await this.room.storage.put(`sessions_backup_${today}`, this.sessions);

    // Weekly backup (keeps 7 days)
    const dayOfWeek = new Date().getDay();
    await this.room.storage.put('sessions_backup_weekly', this.sessions);
  }

  // Backup all visit data periodically
  async backupVisitData() {
    const allVisits = await this.room.storage.list({ prefix: 'visit_' });
    const visits: any[] = [];
    allVisits.forEach((value) => visits.push(value));
    if (visits.length > 0) {
      await this.room.storage.put('visits_backup', visits);
      console.log(`Backed up ${visits.length} visits`);
    }
  }

  // Send notification to IrisHub (pasted to Iris terminal)
  // NEVER hardcode tokens - use env vars! (learned this the hard way 2026-01-12)
  async sendIrisNotification(message: string) {
    const BOT_TOKEN = (this.room as any).env?.IRISHUB_BOT_TOKEN;
    const CHAT_ID = 6248804784;

    if (!BOT_TOKEN) {
      console.log('[IrisNotify] No IRISHUB_BOT_TOKEN set, skipping');
      return;
    }

    try {
      await fetch(`https://api.telegram.org/bot${BOT_TOKEN}/sendMessage`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ chat_id: CHAT_ID, text: message })
      });
      console.log(`[IrisNotify] Sent: ${message.slice(0, 50)}...`);
    } catch (e) {
      console.error('[IrisNotify] Failed:', e);
    }
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
          if (data.direction !== undefined) player.direction = data.direction; // For AFK detection
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
          {
            const target = this.state.players.get(data.targetId);
            if (!target) break;
            if (sender.id === this.state.hostId && data.proxy) {
              this.room.broadcast(JSON.stringify({
                type: 'hit',
                targetId: data.targetId,
                damage: Math.min(data.damage, 50), // Cap damage
                fromId: sender.id
              }));
              break;
            }
            const attacker = this.state.players.get(sender.id);
            if (attacker && attacker.segments && target.segments) {
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

        case 'kick_afk':
          // Host is kicking an AFK player - broadcast removal to all clients
          if (sender.id === this.state.hostId) {
            console.log(`[SERVER] Host kicking AFK player: ${data.playerId}`);
            this.state.players.delete(data.playerId);
            this.room.broadcast(JSON.stringify({
              type: 'player_left',
              playerId: data.playerId,
              reason: 'AFK'
            }));
          }
          break;

        case 'arena_sync':
          // Forward arena sync to the target player (for syncing AI enemies)
          const targetConn = [...this.room.getConnections()].find(c => c.id === data.targetId);
          if (targetConn) {
            targetConn.send(JSON.stringify({
              type: 'arena_sync',
              enemies: data.enemies,
              pickups: data.pickups
            }));
            console.log(`Arena sync forwarded from ${sender.id} to ${data.targetId}`);
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

        case 'enemy_hit':
        case 'enemy_aoe':
          console.log(`[SERVER] ${data.type} from ${sender.id}, hostId=${this.state.hostId}`);
          if (!this.state.hostId) {
            console.log('[SERVER] No host - dropping');
            break;
          }
          if (sender.id === this.state.hostId) {
            console.log('[SERVER] Sender is host - dropping');
            break;
          }
          // Relay to host
          const hostConn = [...this.room.getConnections()].find(c => c.id === this.state.hostId);
          if (hostConn) {
            console.log('[SERVER] Relaying to host');
            hostConn.send(JSON.stringify(data));
          } else {
            console.log('[SERVER] Host connection not found!');
          }
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
      const playerCount = this.state.players.size;
      if (playerCount > 1) {
        console.log('[SERVER] Broadcasting state to', playerCount, 'players');
      }
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

    // BACKUP STATUS - check all backup keys
    if (req.method === 'GET' && url.pathname.endsWith('/backup-status')) {
      const backupKeys = ['sessions', 'sessions_backup_permanent', 'sessions_backup_daily', 'sessions_backup_weekly', 'visits_backup'];
      const status: Record<string, any> = {};

      for (const key of backupKeys) {
        const data = await this.room.storage.get(key);
        if (Array.isArray(data)) {
          status[key] = { count: data.length, exists: true };
        } else if (data) {
          status[key] = { exists: true, type: typeof data };
        } else {
          status[key] = { exists: false };
        }
      }

      // Also check visit_ prefix count
      const allVisits = await this.room.storage.list({ prefix: 'visit_' });
      status['visit_entries'] = { count: allVisits.size };

      return new Response(JSON.stringify(status, null, 2), {
        headers: { ...headers, 'Content-Type': 'application/json' }
      });
    }

    // MANUAL BACKUP TRIGGER
    if (req.method === 'POST' && url.pathname.endsWith('/backup-now')) {
      await this.saveSessions();
      await this.backupVisitData();
      return new Response(JSON.stringify({ ok: true, message: 'Backup completed', sessions: this.sessions.length }), {
        headers: { ...headers, 'Content-Type': 'application/json' }
      });
    }

    // VISITORS DASHBOARD - who is playing?
    if (req.method === 'GET' && url.pathname.endsWith('/visitors')) {
      // Fetch visit data
      const allVisits = await this.room.storage.list({ prefix: 'visit_' });
      const visits: any[] = [];
      for (const [, value] of allVisits) {
        visits.push(value);
      }
      visits.sort((a, b) => b.timestamp - a.timestamp);

      // Fetch session data for duration info
      const allSessions = await this.room.storage.list({ prefix: 'session_' });
      const sessions: any[] = [];
      for (const [, value] of allSessions) {
        sessions.push(value);
      }

      const now = Date.now();
      const hour = 60 * 60 * 1000;
      const day = 24 * hour;

      const lastHour = visits.filter(v => now - v.timestamp < hour).length;
      const last24h = visits.filter(v => now - v.timestamp < day).length;

      // Calculate total play time (completed sessions only)
      const completedSessions = sessions.filter(s => s.leaveTime && s.region !== 'Arizona');
      const totalPlayTimeMs = completedSessions.reduce((sum, s) => sum + (s.leaveTime - s.joinTime), 0);
      const avgSessionMins = completedSessions.length > 0
        ? Math.round(totalPlayTimeMs / completedSessions.length / 60000)
        : 0;

      // Group by location
      const byLocation: Record<string, number> = {};
      for (const v of visits) {
        const key = v.country !== 'unknown' ? `${v.city}, ${v.region}, ${v.country}` : 'Unknown';
        byLocation[key] = (byLocation[key] || 0) + 1;
      }
      const locationRows = Object.entries(byLocation)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 15)
        .map(([loc, count]) => `<tr><td>${loc}</td><td>${count}</td></tr>`)
        .join('');

      // Group by username
      const byUser: Record<string, number> = {};
      for (const v of visits) {
        const key = v.username || 'anonymous';
        byUser[key] = (byUser[key] || 0) + 1;
      }
      const userRows = Object.entries(byUser)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 15)
        .map(([user, count]) => `<tr><td>${user === 'anonymous' ? '<i>anonymous</i>' : user}</td><td>${count}</td></tr>`)
        .join('');

      // Recent visits
      const recentRows = visits.slice(0, 20).map(v => {
        const time = new Date(v.timestamp).toLocaleString();
        const user = v.username || '<i>anon</i>';
        const loc = v.country !== 'unknown' ? `${v.city}, ${v.country}` : 'Unknown';
        return `<tr><td>${time}</td><td>${user}</td><td>${loc}</td></tr>`;
      }).join('');

      // Recent sessions with duration (exclude Arizona/dev sessions)
      const recentSessions = sessions
        .filter(s => s.region !== 'Arizona')
        .sort((a, b) => b.joinTime - a.joinTime)
        .slice(0, 15);
      const sessionRows = recentSessions.map(s => {
        const time = new Date(s.joinTime).toLocaleString();
        const loc = s.country !== 'unknown' ? `${s.city}, ${s.country}` : 'Unknown';
        const durationMs = s.leaveTime ? s.leaveTime - s.joinTime : now - s.joinTime;
        const mins = Math.floor(durationMs / 60000);
        const secs = Math.floor((durationMs % 60000) / 1000);
        const duration = s.leaveTime ? `${mins}m ${secs}s` : `<span style="color:#0f0">${mins}m ${secs}s (active)</span>`;
        return `<tr><td>${time}</td><td>${loc}</td><td>${duration}</td></tr>`;
      }).join('');

      const html = `<!DOCTYPE html>
<html>
<head>
  <title>LOCOMOT.IO Visitors</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="30">
  <style>
    body { font-family: system-ui, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; max-width: 900px; margin: 0 auto; }
    h1 { color: #0ff; text-align: center; }
    .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 15px; margin: 20px 0; }
    .stat { background: #16213e; padding: 20px; border-radius: 10px; text-align: center; }
    .stat-value { font-size: 2.5em; color: #0ff; font-weight: bold; }
    .stat-label { color: #888; font-size: 0.9em; }
    table { width: 100%; border-collapse: collapse; margin: 20px 0; }
    th, td { padding: 10px; text-align: left; border-bottom: 1px solid #333; }
    th { color: #0ff; }
    .section { background: #16213e; padding: 20px; border-radius: 10px; margin: 20px 0; }
    h2 { color: #f0f; margin-top: 0; }
    .note { color: #666; font-size: 0.8em; text-align: center; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    @media (max-width: 600px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <h1>🚂 LOCOMOT.IO Visitors</h1>
  <p class="note">Auto-refreshes every 30s</p>

  <div class="stats">
    <div class="stat">
      <div class="stat-value">${lastHour}</div>
      <div class="stat-label">Last Hour</div>
    </div>
    <div class="stat">
      <div class="stat-value">${last24h}</div>
      <div class="stat-label">Last 24h</div>
    </div>
    <div class="stat">
      <div class="stat-value">${visits.length}</div>
      <div class="stat-label">Total Visits</div>
    </div>
    <div class="stat">
      <div class="stat-value">${Object.keys(byUser).filter(u => u !== 'anonymous').length}</div>
      <div class="stat-label">Named Players</div>
    </div>
    <div class="stat">
      <div class="stat-value">${avgSessionMins}m</div>
      <div class="stat-label">Avg Session</div>
    </div>
    <div class="stat">
      <div class="stat-value">${Math.round(totalPlayTimeMs / 3600000)}h</div>
      <div class="stat-label">Total Play Time</div>
    </div>
  </div>

  <div class="grid">
    <div class="section">
      <h2>👤 Players</h2>
      ${userRows ? `<table><tr><th>Username</th><th>Visits</th></tr>${userRows}</table>` : '<p>No players yet</p>'}
    </div>
    <div class="section">
      <h2>🌍 Locations</h2>
      ${locationRows ? `<table><tr><th>Location</th><th>Visits</th></tr>${locationRows}</table>` : '<p>No data yet</p>'}
    </div>
  </div>

  <div class="section">
    <h2>🕐 Recent Visits</h2>
    ${recentRows ? `<table><tr><th>Time</th><th>Player</th><th>Location</th></tr>${recentRows}</table>` : '<p>No visits yet</p>'}
  </div>

  <div class="section">
    <h2>⏱️ Session Durations</h2>
    ${sessionRows ? `<table><tr><th>Time</th><th>Location</th><th>Duration</th></tr>${sessionRows}</table>` : '<p>No sessions yet</p>'}
  </div>
</body>
</html>`;
      return new Response(html, {
        headers: { ...headers, 'Content-Type': 'text/html' }
      });
    }

    // ICE SKATER ANALYTICS DASHBOARD
    if (req.method === 'GET' && url.pathname.endsWith('/ice-analytics')) {
      // Get player data
      const players: any[] = [];
      const playerKeys = await this.room.storage.list({ prefix: 'ice_player_' });
      for (const [key, value] of playerKeys) {
        players.push(value);
      }

      // Get level completion counts
      const levelCompletions: { [level: number]: number } = {};
      const levelKeys = await this.room.storage.list({ prefix: 'ice_level_completions_' });
      for (const [key, value] of levelKeys) {
        const levelNum = parseInt(key.replace('ice_level_completions_', ''));
        levelCompletions[levelNum] = value as number;
      }

      const now = Date.now();
      const day = 24 * 60 * 60 * 1000;

      // Stats
      const uniquePlayers = players.length;
      const todayPlayers = players.filter(p => now - p.lastSeen < day).length;
      const totalCompletions = players.reduce((sum, p) => sum + (p.completedLevels?.length || 0), 0);

      // Furthest level reached
      const maxLevel = players.reduce((max, p) => Math.max(max, p.highestLevel || 0), 0);

      // Level distribution
      const levelDist: { [level: number]: number } = {};
      for (const p of players) {
        const lvl = p.highestLevel || 0;
        levelDist[lvl] = (levelDist[lvl] || 0) + 1;
      }

      // Recent players
      const recentPlayers = players
        .sort((a, b) => b.lastSeen - a.lastSeen)
        .slice(0, 30);

      // Build bar chart HTML for level distribution
      const distBars = Object.entries(levelDist)
        .sort((a, b) => parseInt(a[0]) - parseInt(b[0]))
        .slice(0, 50)
        .map(([lvl, count]) => {
          const maxCount = Math.max(...Object.values(levelDist), 1);
          const height = Math.max(5, (count / maxCount) * 90);
          return '<div class="bar" style="height:' + height + 'px" title="Level ' + lvl + ': ' + count + ' players"></div>';
        }).join('');

      // Build bar chart HTML for completions
      const compBars = Object.entries(levelCompletions)
        .sort((a, b) => parseInt(a[0]) - parseInt(b[0]))
        .slice(0, 50)
        .map(([lvl, count]) => {
          const maxCount = Math.max(...Object.values(levelCompletions), 1);
          const height = Math.max(5, (count / maxCount) * 90);
          return '<div class="bar" style="height:' + height + 'px" title="Level ' + lvl + ': ' + count + ' completions"></div>';
        }).join('');

      // Build recent players table rows
      const playerRows = recentPlayers.map(p => {
        const ago = Math.round((now - p.lastSeen) / 60000);
        const timeStr = ago < 60 ? ago + 'm ago' : ago < 1440 ? Math.round(ago/60) + 'h ago' : Math.round(ago/1440) + 'd ago';
        return '<tr><td>' + (p.id?.slice(0, 8) || '?') + '...</td><td>' + (p.highestLevel || 0) +
          '</td><td>' + (p.completedLevels?.length || 0) + '</td><td>' + (p.visits || 1) +
          '</td><td>' + timeStr + '</td><td>' + (p.country || '?') + '</td></tr>';
      }).join('');

      const html = '<!DOCTYPE html>' +
'<html>' +
'<head>' +
'  <title>FIGURE Analytics</title>' +
'  <meta name="viewport" content="width=device-width, initial-scale=1">' +
'  <meta http-equiv="refresh" content="60">' +
'  <style>' +
'    body { font-family: system-ui, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; max-width: 1100px; margin: 0 auto; }' +
'    h1 { color: #0ff; text-align: center; }' +
'    .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 15px; margin: 20px 0; }' +
'    .stat { background: #16213e; padding: 20px; border-radius: 10px; text-align: center; }' +
'    .stat-value { font-size: 2.5em; color: #0ff; font-weight: bold; }' +
'    .stat-label { color: #888; font-size: 0.9em; }' +
'    table { width: 100%; border-collapse: collapse; margin: 20px 0; }' +
'    th, td { padding: 10px; text-align: left; border-bottom: 1px solid #333; }' +
'    th { color: #0ff; }' +
'    .section { background: #16213e; padding: 20px; border-radius: 10px; margin: 20px 0; }' +
'    h2 { color: #f0f; margin-top: 0; }' +
'    .bar-chart { display: flex; align-items: flex-end; height: 100px; gap: 2px; }' +
'    .bar { background: linear-gradient(to top, #0ff, #f0f); min-width: 8px; border-radius: 2px 2px 0 0; }' +
'    .bar:hover { opacity: 0.8; }' +
'    .note { color: #666; font-size: 0.8em; text-align: center; }' +
'  </style>' +
'</head>' +
'<body>' +
'  <h1>FIGURE Analytics</h1>' +
'  <p class="note">Auto-refreshes every 60s</p>' +
'  <div class="stats">' +
'    <div class="stat"><div class="stat-value">' + uniquePlayers + '</div><div class="stat-label">Unique Players</div></div>' +
'    <div class="stat"><div class="stat-value">' + todayPlayers + '</div><div class="stat-label">Active Today</div></div>' +
'    <div class="stat"><div class="stat-value">' + totalCompletions + '</div><div class="stat-label">Total Completions</div></div>' +
'    <div class="stat"><div class="stat-value">' + maxLevel + '</div><div class="stat-label">Furthest Level</div></div>' +
'  </div>' +
'  <div class="section">' +
'    <h2>Player Progress Distribution</h2>' +
'    <p class="note">How many players reached each level</p>' +
'    <div class="bar-chart">' + distBars + '</div>' +
'    <p class="note">Levels 0-' + Math.min(49, maxLevel) + ' (hover for details)</p>' +
'  </div>' +
'  <div class="section">' +
'    <h2>Recent Players</h2>' +
'    <table>' +
'      <tr><th>Player</th><th>Highest Level</th><th>Completions</th><th>Visits</th><th>Last Seen</th><th>Country</th></tr>' +
'      ' + playerRows +
'    </table>' +
'  </div>' +
'  <div class="section">' +
'    <h2>Level Completion Counts</h2>' +
'    <p class="note">How many times each level was completed</p>' +
'    <div class="bar-chart">' + compBars + '</div>' +
'    <p class="note">Levels completed (hover for details)</p>' +
'  </div>' +
'</body></html>';
      return new Response(html, {
        headers: { ...headers, 'Content-Type': 'text/html' }
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

        if (data.type === 'behavioral_session') {
          // Store behavioral cloning session data
          const session = (data as any).session;
          if (session && session.frames && session.frames.length > 10) {
            const key = `behavior_${Date.now()}_${session.id?.slice(0,8) || 'anon'}`;
            await this.room.storage.put(key, session);
            console.log(`Behavioral session: ${session.frames.length} frames, quality: ${session.quality?.toFixed(2) || '?'}`);

            // Auto-prune old sessions if over limit (keep max 500)
            const MAX_SESSIONS = 500;
            try {
              const allKeys = await this.room.storage.list({ prefix: 'behavior_', limit: MAX_SESSIONS + 50 });
              if (allKeys.size > MAX_SESSIONS) {
                // Keys are sorted by name (which includes timestamp), so oldest come first
                const keysArray = Array.from(allKeys.keys());
                const toDelete = keysArray.slice(0, allKeys.size - MAX_SESSIONS);
                for (const oldKey of toDelete) {
                  await this.room.storage.delete(oldKey);
                }
                console.log(`[Behavioral] Auto-pruned ${toDelete.length} old sessions`);
              }
            } catch (e) {
              console.log(`[Behavioral] Prune check failed: ${e}`);
            }

            return new Response(JSON.stringify({ success: true, key, frames: session.frames.length }), {
              headers: { ...headers, 'Content-Type': 'application/json' }
            });
          }
          return new Response(JSON.stringify({ success: false, reason: 'insufficient data' }), {
            headers: { ...headers, 'Content-Type': 'application/json' }
          });
        }

        if (data.type === 'list_behavioral') {
          // List stored behavioral sessions
          const keys = await this.room.storage.list({ prefix: 'behavior_' });
          const sessions = [];
          for (const [key, value] of keys) {
            const s = value as any;
            sessions.push({
              key,
              id: s.id,
              playerId: s.playerId,
              mode: s.mode,
              frames: s.frames?.length || 0,
              quality: s.quality,
              timestamp: s.timestamp
            });
          }
          return new Response(JSON.stringify({ count: sessions.length, sessions }), {
            headers: { ...headers, 'Content-Type': 'application/json' }
          });
        }

        if (data.type === 'get_behavioral_data') {
          // Get full behavioral session data for training
          const opts = data as { minQuality?: number; since?: number; limit?: number };
          const minQuality = opts.minQuality || 0;
          const since = opts.since || 0;
          const limit = opts.limit || 100;

          const keys = await this.room.storage.list({ prefix: 'behavior_' });
          const sessions = [];
          let totalFrames = 0;

          for (const [key, value] of keys) {
            const s = value as any;
            // Filter by quality and timestamp
            if ((s.quality || 0) < minQuality) continue;
            if ((s.timestamp || 0) < since) continue;

            sessions.push({
              key,
              id: s.id,
              playerId: s.playerId,
              mode: s.mode,
              quality: s.quality,
              timestamp: s.timestamp,
              frames: s.frames || []  // Include full frame data
            });
            totalFrames += s.frames?.length || 0;

            if (sessions.length >= limit) break;
          }

          return new Response(JSON.stringify({
            count: sessions.length,
            totalFrames,
            sessions
          }), {
            headers: { ...headers, 'Content-Type': 'application/json' }
          });
        }

        if (data.type === 'delete_behavioral_sessions') {
          // Delete behavioral sessions by key
          const req = data as { keys: string[] };
          if (!req.keys || !Array.isArray(req.keys)) {
            return new Response(JSON.stringify({ error: 'Missing keys array' }), {
              status: 400, headers: { ...headers, 'Content-Type': 'application/json' }
            });
          }

          let deleted = 0;
          for (const key of req.keys) {
            if (key.startsWith('behavior_')) {
              await this.room.storage.delete(key);
              deleted++;
            }
          }

          console.log(`[Behavioral] Deleted ${deleted} sessions`);

          return new Response(JSON.stringify({ deleted }), {
            headers: { ...headers, 'Content-Type': 'application/json' }
          });
        }

        if (data.type === 'purge_behavioral_batch') {
          // Delete a batch of behavioral sessions without listing all
          // Uses storage.list with a small limit to avoid memory overflow
          const req = data as { batch_size?: number, keep_player?: string };
          const batchSize = Math.min(req.batch_size || 50, 100);

          try {
            // List a small batch of behavioral keys
            const keys = await this.room.storage.list({ prefix: 'behavior_', limit: batchSize });

            let deleted = 0;
            const keepPlayer = req.keep_player;

            for (const [key, value] of keys) {
              // Optionally keep sessions from a specific player
              if (keepPlayer && typeof value === 'object' && (value as any).playerId === keepPlayer) {
                continue;
              }
              await this.room.storage.delete(key);
              deleted++;
            }

            console.log(`[Behavioral] Purged ${deleted} sessions (batch of ${batchSize})`);

            return new Response(JSON.stringify({
              deleted,
              remaining: keys.size - deleted,
              message: deleted > 0 ? 'Call again to delete more' : 'No more sessions to delete'
            }), {
              headers: { ...headers, 'Content-Type': 'application/json' }
            });
          } catch (e) {
            return new Response(JSON.stringify({ error: String(e) }), {
              status: 500, headers: { ...headers, 'Content-Type': 'application/json' }
            });
          }
        }

        if (data.type === 'track_visit') {
          // Track page visit for analytics
          const visit = data as {
            username?: string;
            userAgent?: string;
            screenSize?: string;
            referrer?: string;
          };

          // Get geo from Cloudflare headers
          const cf = (req as any).cf || {};
          const region = cf.region || cf.regionCode || 'unknown';
          const country = cf.country || 'unknown';
          const city = cf.city || 'unknown';

          const visitData = {
            timestamp: Date.now(),
            username: visit.username || null,
            region,
            country,
            city,
            userAgent: visit.userAgent || null,
            screenSize: visit.screenSize || null,
            referrer: visit.referrer || null
          };

          // Store visit
          const visitKey = `visit_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
          await this.room.storage.put(visitKey, visitData);

          // Keep only last 1000 visits (cleanup old ones)
          const allVisits = await this.room.storage.list({ prefix: 'visit_' });
          if (allVisits.size > 1000) {
            const visitKeys = Array.from(allVisits.keys()).sort();
            const toDelete = visitKeys.slice(0, visitKeys.length - 1000);
            for (const key of toDelete) {
              await this.room.storage.delete(key);
            }
          }

          console.log(`[Visit] ${visit.username || 'anon'} from ${city}, ${region}, ${country}`);

          // Backup visit data after every 10 visits
          if (allVisits.size % 10 === 0) {
            await this.backupVisitData();
          }

          return new Response(JSON.stringify({ ok: true }), {
            headers: { ...headers, 'Content-Type': 'application/json' }
          });
        }

        if (data.type === 'get_visits') {
          // Get visit analytics
          const opts = data as { limit?: number; since?: number };
          const limit = opts.limit || 100;
          const since = opts.since || 0;

          const allVisits = await this.room.storage.list({ prefix: 'visit_' });
          const visits: any[] = [];

          for (const [key, value] of allVisits) {
            const v = value as any;
            if (v.timestamp > since) {
              visits.push(v);
            }
          }

          // Sort by timestamp descending (newest first)
          visits.sort((a, b) => b.timestamp - a.timestamp);

          // Compute stats
          const now = Date.now();
          const hour = 60 * 60 * 1000;
          const day = 24 * hour;

          const lastHour = visits.filter(v => now - v.timestamp < hour).length;
          const last24h = visits.filter(v => now - v.timestamp < day).length;

          // Group by region
          const byRegion: Record<string, number> = {};
          for (const v of visits) {
            const key = `${v.city}, ${v.region}`;
            byRegion[key] = (byRegion[key] || 0) + 1;
          }

          // Group by username
          const byUser: Record<string, number> = {};
          for (const v of visits) {
            const key = v.username || 'anonymous';
            byUser[key] = (byUser[key] || 0) + 1;
          }

          return new Response(JSON.stringify({
            total: visits.length,
            lastHour,
            last24h,
            byRegion: Object.entries(byRegion).sort((a, b) => b[1] - a[1]),
            byUser: Object.entries(byUser).sort((a, b) => b[1] - a[1]),
            recent: visits.slice(0, limit)
          }), {
            headers: { ...headers, 'Content-Type': 'application/json' }
          });
        }

        if (data.type === 'get_new_behavioral') {
          // Get sessions newer than a timestamp (for continuous training)
          const opts = data as { since: number; limit?: number };
          const since = opts.since || 0;
          const limit = opts.limit || 50;

          const keys = await this.room.storage.list({ prefix: 'behavior_' });
          const sessions = [];

          for (const [key, value] of keys) {
            const s = value as any;
            const ts = s.timestamp || 0;
            if (ts > since) {
              sessions.push({
                key,
                id: s.id,
                playerId: s.playerId,
                mode: s.mode,
                quality: s.quality,
                timestamp: ts,
                frames: s.frames || []
              });
            }
          }

          // Sort by timestamp ascending (oldest first for replay buffer)
          sessions.sort((a, b) => a.timestamp - b.timestamp);

          return new Response(JSON.stringify({
            count: sessions.length,
            sessions: sessions.slice(0, limit)
          }), {
            headers: { ...headers, 'Content-Type': 'application/json' }
          });
        }

        if (data.type === 'run_summary') {
          // Store run summary for engagement-aware training
          const run = data as {
            runId: string;
            playerId: string;
            mode: string;
            startTime: number;
            durationMs: number;
            finalScore: number;
            finalLength: number;
            peakLength: number;
            peakScore: number;
            turns: number;
            pickups: number;
            kills: number;
            intensity: number;
            focusLostCount: number;
            deathReason: string;
            engaged: number;
            events: unknown[];
          };

          const key = `run_${run.startTime}_${run.runId.slice(0, 8)}`;
          await this.room.storage.put(key, {
            ...run,
            events: run.events?.slice(0, 100) // Cap stored events
          });

          // Update player return tracking
          const playerKey = `player_returns_${run.playerId}`;
          const playerReturns = await this.room.storage.get(playerKey) as number[] || [];
          playerReturns.push(run.startTime);
          // Keep last 30 days of return data
          const thirtyDaysAgo = Date.now() - 30 * 24 * 60 * 60 * 1000;
          const recentReturns = playerReturns.filter(t => t > thirtyDaysAgo);
          await this.room.storage.put(playerKey, recentReturns);

          console.log(`Run stored: ${run.mode} ${run.durationMs}ms engaged:${run.engaged} player:${run.playerId.slice(0, 8)}`);

          return new Response(JSON.stringify({ success: true, runId: run.runId }), {
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

        if (data.type === 'get_runs') {
          // Retrieve run summaries for training
          const opts = data as { days?: number; mode?: string; engagedOnly?: boolean };
          const days = opts.days || 7;
          const cutoff = Date.now() - days * 24 * 60 * 60 * 1000;

          const runs: unknown[] = [];
          const keys = await this.room.storage.list({ prefix: 'run_' });

          for (const [key, value] of keys) {
            const run = value as { startTime: number; mode: string; engaged: number };
            if (run.startTime < cutoff) continue;
            if (opts.mode && run.mode !== opts.mode) continue;
            if (opts.engagedOnly && run.engaged !== 1) continue;
            runs.push(value);
          }

          // Sort by startTime descending
          runs.sort((a: any, b: any) => b.startTime - a.startTime);

          return new Response(JSON.stringify({
            count: runs.length,
            runs: runs.slice(0, 1000) // Cap at 1000 runs
          }), {
            headers: { ...headers, 'Content-Type': 'application/json' }
          });
        }

        if (data.type === 'get_run_stats') {
          // Get aggregate stats for dashboard
          const keys = await this.room.storage.list({ prefix: 'run_' });
          let total = 0, engaged = 0, ffaCount = 0, teamCount = 0;
          let totalDuration = 0, totalIntensity = 0;
          const last24h = Date.now() - 24 * 60 * 60 * 1000;
          let last24hCount = 0;

          for (const [key, value] of keys) {
            const run = value as { startTime: number; mode: string; engaged: number; durationMs: number; intensity: number };
            total++;
            if (run.engaged === 1) engaged++;
            if (run.mode === 'ffa') ffaCount++;
            if (run.mode === 'team') teamCount++;
            totalDuration += run.durationMs || 0;
            totalIntensity += run.intensity || 0;
            if (run.startTime > last24h) last24hCount++;
          }

          return new Response(JSON.stringify({
            totalRuns: total,
            engagedRuns: engaged,
            engagementRate: total > 0 ? Math.round(engaged / total * 100) : 0,
            ffaRuns: ffaCount,
            teamRuns: teamCount,
            avgDurationMs: total > 0 ? Math.round(totalDuration / total) : 0,
            avgIntensity: total > 0 ? Math.round(totalIntensity / total * 10) / 10 : 0,
            last24h: last24hCount
          }), {
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


        // ==================== GENOME POOL SYSTEM ====================

        if (data.type === 'get_genome_manifest') {
          // Return manifest with all genome metadata
          let manifest = await this.room.storage.get('genome_manifest') as any;

          if (!manifest) {
            // Initialize manifest with default genomes
            manifest = {
              genomes: [
                {
                  id: 'ffa',
                  type: 'rl',
                  name: 'DefaultFFA',
                  source: 'embedded',
                  created: new Date().toISOString(),
                  performance: { games: 0, avg_score: 0, avg_survival: 0, win_rate: 0, fitness: 100, kills: 0, player_kills: 0, deaths: 0 },
                  active: true
                },
                {
                  id: 'team',
                  type: 'rl',
                  name: 'DefaultTeam',
                  source: 'embedded',
                  created: new Date().toISOString(),
                  performance: { games: 0, avg_score: 0, avg_survival: 0, win_rate: 0, fitness: 100, kills: 0, player_kills: 0, deaths: 0 },
                  active: true
                }
              ],
              max_active: 15,
              last_tournament: null,
              version: 1
            };
            await this.room.storage.put('genome_manifest', manifest);
          }

          // Apply fitness decay (0.995 per hour) to prevent stagnation
          const now = Date.now();
          const lastDecay = manifest.lastFitnessDecay || now;
          const hoursPassed = (now - lastDecay) / (1000 * 60 * 60);

          if (hoursPassed >= 1) {
            const decayFactor = Math.pow(0.995, hoursPassed);
            for (const g of manifest.genomes) {
              if (g.performance?.fitness) {
                g.performance.fitness = Math.max(10, Math.round(g.performance.fitness * decayFactor));
              }
            }
            manifest.lastFitnessDecay = now;
            await this.room.storage.put('genome_manifest', manifest);
            console.log(`[GenomePool] Applied fitness decay (${hoursPassed.toFixed(1)}h, factor ${decayFactor.toFixed(4)})`);
          }

          return new Response(JSON.stringify(manifest), {
            headers: { ...headers, 'Content-Type': 'application/json' }
          });
        }

        if (data.type === 'get_genome') {
          // Return full genome weights by ID
          const genomeId = (data as any).id;
          if (!genomeId) {
            return new Response(JSON.stringify({ error: 'Missing genome id' }), {
              status: 400, headers: { ...headers, 'Content-Type': 'application/json' }
            });
          }

          const genome = await this.room.storage.get(`genome_${genomeId}`);
          if (!genome) {
            return new Response(JSON.stringify({ error: 'Genome not found' }), {
              status: 404, headers: { ...headers, 'Content-Type': 'application/json' }
            });
          }

          return new Response(JSON.stringify(genome), {
            headers: { ...headers, 'Content-Type': 'application/json' }
          });
        }



        if (data.type === 'store_genome_part') {
          // Store a single weight array for a genome
          const part = data as any;
          if (!part.genome_id || !part.key || !part.data) {
            return new Response(JSON.stringify({ error: 'Invalid part' }), {
              status: 400, headers: { ...headers, 'Content-Type': 'application/json' }
            });
          }

          // Store the part directly
          const partKey = `genome_part_${part.genome_id}_${part.key.replace(/\./g, '_')}`;
          await this.room.storage.put(partKey, part.data);

          console.log(`[GenomePool] Stored part ${part.key} for ${part.genome_id}`);

          return new Response(JSON.stringify({
            success: true,
            key: part.key
          }), {
            headers: { ...headers, 'Content-Type': 'application/json' }
          });
        }

        if (data.type === 'register_genome') {
          // Register a genome in the manifest
          // Supports two modes:
          // 1. Self-hosted: weightsUrl provided, weights stored externally
          // 2. PartyKit-hosted: keys provided, weights stored in parts
          const reg = data as any;
          const genomeId = reg.genome_id || reg.genomeId;

          if (!genomeId) {
            return new Response(JSON.stringify({ error: 'Missing genome id' }), {
              status: 400, headers: { ...headers, 'Content-Type': 'application/json' }
            });
          }

          // If no weightsUrl, require keys and verify parts exist
          if (!reg.weightsUrl && reg.keys) {
            for (const key of reg.keys) {
              const partKey = `genome_part_${genomeId}_${key.replace(/\./g, '_')}`;
              const exists = await this.room.storage.get(partKey);
              if (!exists) {
                return new Response(JSON.stringify({ error: `Missing part: ${key}` }), {
                  status: 400, headers: { ...headers, 'Content-Type': 'application/json' }
                });
              }
            }
          }

          // Update manifest
          let manifest = await this.room.storage.get('genome_manifest') as any;
          if (!manifest) manifest = { genomes: [], max_active: 15, version: 1 };

          const existing = manifest.genomes.findIndex((g: any) => g.id === genomeId);
          const genomeEntry = {
            id: genomeId,
            type: reg.genome_type || 'behavioral',
            name: reg.name || genomeId,
            source: reg.source || 'unknown',
            weightsUrl: reg.weightsUrl || null,  // URL to external weights (self-hosted)
            keys: reg.keys || null,  // Keys for PartyKit-hosted parts
            parents: reg.parents || [],
            generation: reg.generation || 0,
            trained: reg.trained || 'server',  // 'client' or 'server'
            created: new Date().toISOString(),
            performance: { games: 0, avg_score: 0, avg_survival: 0, win_rate: 0, fitness: 100, kills: 0, player_kills: 0, deaths: 0 },
            active: true
          };

          if (existing >= 0) {
            manifest.genomes[existing] = { ...manifest.genomes[existing], ...genomeEntry };
          } else {
            manifest.genomes.push(genomeEntry);
          }

          manifest.version++;
          await this.room.storage.put('genome_manifest', manifest);

          console.log(`[GenomePool] Registered genome: ${genomeId} (${reg.weightsUrl ? 'self-hosted' : 'partykit'})`);

          return new Response(JSON.stringify({
            success: true,
            id: genomeId,
            active: genomeEntry.active
          }), {
            headers: { ...headers, 'Content-Type': 'application/json' }
          });
        }

        if (data.type === 'get_genome_part') {
          // Get a single part of a genome
          const req = data as any;
          if (!req.genome_id || !req.key) {
            return new Response(JSON.stringify({ error: 'Missing genome_id or key' }), {
              status: 400, headers: { ...headers, 'Content-Type': 'application/json' }
            });
          }

          const partKey = `genome_part_${req.genome_id}_${req.key.replace(/\./g, '_')}`;
          const part = await this.room.storage.get(partKey);

          if (!part) {
            return new Response(JSON.stringify({ error: 'Part not found' }), {
              status: 404, headers: { ...headers, 'Content-Type': 'application/json' }
            });
          }

          return new Response(JSON.stringify({ data: part }), {
            headers: { ...headers, 'Content-Type': 'application/json' }
          });
        }
        if (data.type === 'submit_genome_chunk') {
          // Upload a chunk of genome weights
          const chunk = data as any;
          if (!chunk.genome_id || !chunk.chunk_index === undefined || !chunk.key || !chunk.data) {
            return new Response(JSON.stringify({ error: 'Invalid chunk' }), {
              status: 400, headers: { ...headers, 'Content-Type': 'application/json' }
            });
          }

          // Store chunk
          const chunkKey = `genome_chunk_${chunk.genome_id}_${chunk.key}_${chunk.chunk_index}`;
          await this.room.storage.put(chunkKey, {
            data: chunk.data,
            total_chunks: chunk.total_chunks
          });

          console.log(`[GenomePool] Chunk ${chunk.chunk_index + 1}/${chunk.total_chunks} for ${chunk.genome_id}.${chunk.key}`);

          return new Response(JSON.stringify({
            success: true,
            chunk_index: chunk.chunk_index
          }), {
            headers: { ...headers, 'Content-Type': 'application/json' }
          });
        }

        if (data.type === 'finalize_genome') {
          // Assemble chunks into complete genome
          const finalize = data as any;
          if (!finalize.genome_id || !finalize.keys) {
            return new Response(JSON.stringify({ error: 'Invalid finalize request' }), {
              status: 400, headers: { ...headers, 'Content-Type': 'application/json' }
            });
          }

          const weights: any = {};

          // Reassemble each weight key
          for (const keyInfo of finalize.keys) {
            const { key, total_chunks } = keyInfo;
            const chunks: any[] = [];

            for (let i = 0; i < total_chunks; i++) {
              const chunkKey = `genome_chunk_${finalize.genome_id}_${key}_${i}`;
              const chunk = await this.room.storage.get(chunkKey) as any;
              if (!chunk) {
                return new Response(JSON.stringify({ error: `Missing chunk ${i} for ${key}` }), {
                  status: 400, headers: { ...headers, 'Content-Type': 'application/json' }
                });
              }
              chunks.push(chunk.data);
              // Clean up chunk
              await this.room.storage.delete(chunkKey);
            }

            // Flatten chunks back into array
            weights[key] = chunks.flat();
          }

          // Store assembled genome
          await this.room.storage.put(`genome_${finalize.genome_id}`, weights);

          // Update manifest
          let manifest = await this.room.storage.get('genome_manifest') as any;
          if (!manifest) manifest = { genomes: [], max_active: 15, version: 1 };

          const existing = manifest.genomes.findIndex((g: any) => g.id === finalize.genome_id);
          const genomeEntry = {
            id: finalize.genome_id,
            type: finalize.genome_type || 'behavioral',
            name: finalize.name || finalize.genome_id,
            source: finalize.source || 'unknown',
            created: new Date().toISOString(),
            performance: { games: 0, avg_score: 0, avg_survival: 0, win_rate: 0, fitness: 100, kills: 0, player_kills: 0, deaths: 0 },
            active: true  // Always active - culling handles cleanup
          };

          if (existing >= 0) {
            manifest.genomes[existing] = { ...manifest.genomes[existing], ...genomeEntry };
          } else {
            manifest.genomes.push(genomeEntry);
          }

          manifest.version++;
          await this.room.storage.put('genome_manifest', manifest);

          console.log(`[GenomePool] Finalized genome: ${finalize.genome_id}`);

          return new Response(JSON.stringify({
            success: true,
            id: finalize.genome_id,
            active: genomeEntry.active
          }), {
            headers: { ...headers, 'Content-Type': 'application/json' }
          });
        }
        if (data.type === 'submit_genome') {
          // Upload new genome for consideration
          const submission = data as any;
          if (!submission.id || !submission.weights || !submission.weights['net.0.weight']) {
            return new Response(JSON.stringify({ error: 'Invalid genome submission' }), {
              status: 400, headers: { ...headers, 'Content-Type': 'application/json' }
            });
          }

          // Store the genome weights
          await this.room.storage.put(`genome_${submission.id}`, submission.weights);

          // Update manifest
          let manifest = await this.room.storage.get('genome_manifest') as any;
          if (!manifest) manifest = { genomes: [], max_active: 15, version: 1 };

          // Check if genome already exists
          const existing = manifest.genomes.findIndex((g: any) => g.id === submission.id);
          const genomeEntry = {
            id: submission.id,
            type: submission.type || 'behavioral',
            name: submission.name || submission.id,
            source: submission.source || 'unknown',
            created: new Date().toISOString(),
            performance: { games: 0, avg_score: 0, avg_survival: 0, win_rate: 0, fitness: 100, kills: 0, player_kills: 0, deaths: 0 },
            active: true  // Always active - culling handles cleanup
          };

          if (existing >= 0) {
            manifest.genomes[existing] = { ...manifest.genomes[existing], ...genomeEntry };
          } else {
            manifest.genomes.push(genomeEntry);
          }

          manifest.version++;
          await this.room.storage.put('genome_manifest', manifest);

          console.log(`[GenomePool] Submitted genome: ${submission.id} (${submission.type || 'behavioral'})`);

          return new Response(JSON.stringify({
            success: true,
            id: submission.id,
            active: genomeEntry.active
          }), {
            headers: { ...headers, 'Content-Type': 'application/json' }
          });
        }

        if (data.type === 'update_genome_stats') {
          // Record game performance for a genome
          const stats = data as any;
          if (!stats.genome_id) {
            return new Response(JSON.stringify({ error: 'Missing genome_id' }), {
              status: 400, headers: { ...headers, 'Content-Type': 'application/json' }
            });
          }

          let manifest = await this.room.storage.get('genome_manifest') as any;
          if (!manifest) {
            return new Response(JSON.stringify({ error: 'Manifest not found' }), {
              status: 404, headers: { ...headers, 'Content-Type': 'application/json' }
            });
          }

          const genome = manifest.genomes.find((g: any) => g.id === stats.genome_id);
          if (!genome) {
            return new Response(JSON.stringify({ error: 'Genome not found' }), {
              status: 404, headers: { ...headers, 'Content-Type': 'application/json' }
            });
          }

          // Update rolling averages
          const p = genome.performance;
          const n = p.games + 1;
          p.avg_score = ((p.avg_score * p.games) + (stats.score || 0)) / n;
          p.avg_survival = ((p.avg_survival * p.games) + (stats.survival || 0)) / n;
          if (stats.won !== undefined) {
            p.win_rate = ((p.win_rate * p.games) + (stats.won ? 1 : 0)) / n;
          }
          p.games = n;

          await this.room.storage.put('genome_manifest', manifest);

          return new Response(JSON.stringify({ success: true, performance: p }), {
            headers: { ...headers, 'Content-Type': 'application/json' }
          });
        }

        if (data.type === 'report_fitness_events') {
          // Report kills/deaths from live gameplay for fitness tracking
          const req = data as any;
          if (!req.events || !Array.isArray(req.events)) {
            return new Response(JSON.stringify({ error: 'Invalid events array' }), {
              status: 400, headers: { ...headers, 'Content-Type': 'application/json' }
            });
          }

          let manifest = await this.room.storage.get('genome_manifest') as any;
          if (!manifest) {
            return new Response(JSON.stringify({ error: 'Manifest not found' }), {
              status: 404, headers: { ...headers, 'Content-Type': 'application/json' }
            });
          }

          // Fitness values for each event type
          const fitnessValues: { [key: string]: number } = {
            'kill_player': 50,       // Big reward for killing a human
            'kill_bot': 5,           // Small reward for killing another bot
            'killed_by_player': -20, // Penalty for dying to human
            'killed_by_bot': -10,    // Smaller penalty for dying to bot
            'damage_player': 1,      // Minor reward for landing a hit
            'damaged_by_player': -1, // Minor penalty for taking a hit
            'destroy_segment': 3,    // Good reward for destroying a segment
            'lost_segment': -3       // Bigger penalty for losing a segment
          };

          let updated = 0;
          for (const event of req.events) {
            const { genome_id, event_type } = event;
            if (!genome_id || !event_type) continue;

            const genome = manifest.genomes.find((g: any) => g.id === genome_id);
            if (!genome) continue;

            // Ensure performance has fitness fields
            if (!genome.performance) {
              genome.performance = { games: 0, avg_score: 0, avg_survival: 0, win_rate: 0, fitness: 100, kills: 0, player_kills: 0, deaths: 0 };
            }
            if (genome.performance.fitness === undefined) genome.performance.fitness = 100;
            if (genome.performance.kills === undefined) genome.performance.kills = 0;
            if (genome.performance.player_kills === undefined) genome.performance.player_kills = 0;
            if (genome.performance.deaths === undefined) genome.performance.deaths = 0;

            // Apply fitness change
            const delta = fitnessValues[event_type] || 0;
            genome.performance.fitness = Math.max(0, Math.min(1000, genome.performance.fitness + delta));

            // Update counters
            if (event_type === 'kill_player') {
              genome.performance.kills++;
              genome.performance.player_kills++;
            } else if (event_type === 'kill_bot') {
              genome.performance.kills++;
            } else if (event_type === 'killed_by_player' || event_type === 'killed_by_bot') {
              genome.performance.deaths++;
            }

            updated++;
          }

          if (updated > 0) {
            await this.room.storage.put('genome_manifest', manifest);
            console.log(`[GenomePool] Applied ${updated} fitness events`);
          }

          return new Response(JSON.stringify({ success: true, updated }), {
            headers: { ...headers, 'Content-Type': 'application/json' }
          });
        }

        if (data.type === 'update_genome_status') {
          // Update genome active status (for culling)
          const req = data as any;
          if (!req.genome_id) {
            return new Response(JSON.stringify({ error: 'Missing genome_id' }), {
              status: 400, headers: { ...headers, 'Content-Type': 'application/json' }
            });
          }

          let manifest = await this.room.storage.get('genome_manifest') as any;
          if (!manifest) {
            return new Response(JSON.stringify({ error: 'Manifest not found' }), {
              status: 404, headers: { ...headers, 'Content-Type': 'application/json' }
            });
          }

          const genome = manifest.genomes.find((g: any) => g.id === req.genome_id);
          if (!genome) {
            return new Response(JSON.stringify({ error: 'Genome not found' }), {
              status: 404, headers: { ...headers, 'Content-Type': 'application/json' }
            });
          }

          // Update active status
          if (req.active !== undefined) {
            genome.active = req.active;
          }

          // Optionally update other status fields
          if (req.stats) {
            genome.arena_stats = { ...genome.arena_stats, ...req.stats };
          }

          await this.room.storage.put('genome_manifest', manifest);

          console.log(`[GenomePool] Updated status for ${req.genome_id}: active=${genome.active}`);

          return new Response(JSON.stringify({ success: true, genome }), {
            headers: { ...headers, 'Content-Type': 'application/json' }
          });
        }

        if (data.type === 'run_tournament') {
          // Evaluate genomes and update rankings
          let manifest = await this.room.storage.get('genome_manifest') as any;
          if (!manifest) {
            return new Response(JSON.stringify({ error: 'Manifest not found' }), {
              status: 404, headers: { ...headers, 'Content-Type': 'application/json' }
            });
          }

          // Calculate fitness scores
          for (const genome of manifest.genomes) {
            const p = genome.performance;
            if (p.games < 5) {
              genome.fitness = 0; // Not enough data
            } else {
              // Weighted fitness: score + survival + win_rate
              genome.fitness = (
                0.4 * (p.avg_score / 100) +  // Normalize score
                0.3 * (p.avg_survival / 60) + // Normalize survival (60s = good)
                0.3 * p.win_rate
              );
            }
          }

          // Sort by fitness
          manifest.genomes.sort((a: any, b: any) => (b.fitness || 0) - (a.fitness || 0));

          // Mark top N as active, rest as inactive
          manifest.genomes.forEach((g: any, i: number) => {
            g.active = i < manifest.max_active;
            g.rank = i + 1;
          });

          manifest.last_tournament = new Date().toISOString();
          await this.room.storage.put('genome_manifest', manifest);

          console.log(`[GenomePool] Tournament complete. Top genome: ${manifest.genomes[0]?.id}`);

          return new Response(JSON.stringify({
            success: true,
            rankings: manifest.genomes.map((g: any) => ({
              id: g.id,
              fitness: g.fitness,
              rank: g.rank,
              active: g.active
            }))
          }), {
            headers: { ...headers, 'Content-Type': 'application/json' }
          });
        }
        // ==================== ICE SKATER LEADERBOARD ====================

        if (data.type === 'ice_skater_get_best') {
          const levelIndex = (data as any).level;
          if (levelIndex === undefined) {
            return new Response(JSON.stringify({ error: 'Missing level' }), {
              status: 400, headers: { ...headers, 'Content-Type': 'application/json' }
            });
          }
          const best = await this.room.storage.get(`ice_best_${levelIndex}`) as number | null;
          return new Response(JSON.stringify({ best: best || null }), {
            headers: { ...headers, 'Content-Type': 'application/json' }
          });
        }

        if (data.type === 'ice_skater_submit') {
          const { level, moves } = data as any;
          if (level === undefined || moves === undefined) {
            return new Response(JSON.stringify({ error: 'Missing level or moves' }), {
              status: 400, headers: { ...headers, 'Content-Type': 'application/json' }
            });
          }
          const key = `ice_best_${level}`;
          const current = await this.room.storage.get(key);
          const currentBest = typeof current === 'number' ? current : null;
          console.log(`[IceSkater] Level ${level}, moves ${moves}, current best: ${currentBest}`);

          if (currentBest === null || moves < currentBest) {
            await this.room.storage.put(key, moves);
            console.log(`[IceSkater] New record for level ${level}: ${moves} moves`);
            return new Response(JSON.stringify({ best: moves, isRecord: true }), {
              headers: { ...headers, 'Content-Type': 'application/json' }
            });
          }
          return new Response(JSON.stringify({ best: currentBest, isRecord: false }), {
            headers: { ...headers, 'Content-Type': 'application/json' }
          });
        }

        if (data.type === 'ice_skater_get_all_bests') {
          // Get all best times (for preloading)
          const allBests: { [level: number]: number } = {};
          const keys = await this.room.storage.list({ prefix: 'ice_best_' });
          for (const [key, value] of keys) {
            const levelNum = parseInt(key.replace('ice_best_', ''));
            allBests[levelNum] = value as number;
          }
          return new Response(JSON.stringify({ bests: allBests }), {
            headers: { ...headers, 'Content-Type': 'application/json' }
          });
        }

        // ==================== ICE SKATER SPEED RECORDS (TIME-BASED) ====================

        if (data.type === 'ice_skater_get_best_time') {
          const levelIndex = (data as any).level;
          if (levelIndex === undefined) {
            return new Response(JSON.stringify({ error: 'Missing level' }), {
              status: 400, headers: { ...headers, 'Content-Type': 'application/json' }
            });
          }
          const bestTime = await this.room.storage.get(`ice_time_${levelIndex}`) as number | null;
          return new Response(JSON.stringify({ best: bestTime || null }), {
            headers: { ...headers, 'Content-Type': 'application/json' }
          });
        }

        if (data.type === 'ice_skater_submit_time') {
          const { level, time } = data as any;
          if (level === undefined || time === undefined) {
            return new Response(JSON.stringify({ error: 'Missing level or time' }), {
              status: 400, headers: { ...headers, 'Content-Type': 'application/json' }
            });
          }
          const key = `ice_time_${level}`;
          const current = await this.room.storage.get(key);
          const currentBest = typeof current === 'number' ? current : null;
          console.log(`[IceSkater] Level ${level}, time ${time}ms, current best: ${currentBest}ms`);

          if (currentBest === null || time < currentBest) {
            await this.room.storage.put(key, time);
            console.log(`[IceSkater] New speed record for level ${level}: ${time}ms`);

            // Notify Iris of new record
            const timeStr = (time / 1000).toFixed(2);
            const prevStr = currentBest ? (currentBest / 1000).toFixed(2) + 's' : 'none';
            this.sendIrisNotification(`⛸️ FIGURE ∞ NEW RECORD!\nLevel ${level}: ${timeStr}s\nPrevious: ${prevStr}`);

            return new Response(JSON.stringify({ best: time, isRecord: true }), {
              headers: { ...headers, 'Content-Type': 'application/json' }
            });
          }
          return new Response(JSON.stringify({ best: currentBest, isRecord: false }), {
            headers: { ...headers, 'Content-Type': 'application/json' }
          });
        }

        if (data.type === 'ice_skater_get_all_times') {
          // Get all best times (for preloading speed records)
          const allTimes: { [level: number]: number } = {};
          const keys = await this.room.storage.list({ prefix: 'ice_time_' });
          for (const [key, value] of keys) {
            const levelNum = parseInt(key.replace('ice_time_', ''));
            allTimes[levelNum] = value as number;
          }
          return new Response(JSON.stringify({ times: allTimes }), {
            headers: { ...headers, 'Content-Type': 'application/json' }
          });
        }

        // ==================== ICE SKATER PLAYER ANALYTICS ====================

        if (data.type === 'ice_skater_track_player') {
          // Track player visit with their highest level reached
          // Retroactive: if playing level 76, count as completing 1-75
          const req = data as any;
          const playerId = req.player_id;
          const highestLevel = req.highest_level || 0;

          if (!playerId) {
            return new Response(JSON.stringify({ error: 'Missing player_id' }), {
              status: 400, headers: { ...headers, 'Content-Type': 'application/json' }
            });
          }

          // Get geo from Cloudflare headers
          const cf = (req as any).cf || {};
          const country = cf?.country || 'unknown';
          const region = cf?.region || cf?.regionCode || 'unknown';

          // Get or create player record
          const playerKey = `ice_player_${playerId}`;
          let player = await this.room.storage.get(playerKey) as any;

          if (!player) {
            player = {
              id: playerId,
              firstSeen: Date.now(),
              lastSeen: Date.now(),
              visits: 1,
              highestLevel: highestLevel,
              completedLevels: new Array(highestLevel).fill(0).map((_, i) => i), // Retroactive
              country,
              region
            };
          } else {
            player.lastSeen = Date.now();
            player.visits++;
            // Update highest level if higher (retroactive tracking)
            if (highestLevel > player.highestLevel) {
              // Add all levels between old highest and new highest
              for (let i = player.highestLevel; i < highestLevel; i++) {
                if (!player.completedLevels.includes(i)) {
                  player.completedLevels.push(i);
                }
              }
              player.highestLevel = highestLevel;
            }
          }

          await this.room.storage.put(playerKey, player);
          console.log(`[IceSkater] Player ${playerId.slice(0,8)} tracked, highest: ${player.highestLevel}`);

          return new Response(JSON.stringify({ success: true, player }), {
            headers: { ...headers, 'Content-Type': 'application/json' }
          });
        }

        if (data.type === 'ice_skater_track_completion') {
          // Track puzzle completion
          const req = data as any;
          const playerId = req.player_id;
          const level = req.level;
          const moves = req.moves;
          const time = req.time;

          if (!playerId || level === undefined) {
            return new Response(JSON.stringify({ error: 'Missing player_id or level' }), {
              status: 400, headers: { ...headers, 'Content-Type': 'application/json' }
            });
          }

          // Update player record
          const playerKey = `ice_player_${playerId}`;
          let player = await this.room.storage.get(playerKey) as any;

          if (!player) {
            player = {
              id: playerId,
              firstSeen: Date.now(),
              lastSeen: Date.now(),
              visits: 1,
              highestLevel: level + 1,
              completedLevels: [],
              completions: []
            };
          }

          // Add this level to completed if not already
          if (!player.completedLevels) player.completedLevels = [];
          if (!player.completedLevels.includes(level)) {
            player.completedLevels.push(level);
          }

          // Track completion details
          if (!player.completions) player.completions = [];
          player.completions.push({
            level,
            moves,
            time,
            timestamp: Date.now()
          });
          // Keep last 100 completions per player
          if (player.completions.length > 100) {
            player.completions = player.completions.slice(-100);
          }

          // Update highest level (retroactive)
          if (level + 1 > player.highestLevel) {
            player.highestLevel = level + 1;
            // Retroactively add all prior levels as completed
            for (let i = 0; i < level; i++) {
              if (!player.completedLevels.includes(i)) {
                player.completedLevels.push(i);
              }
            }
          }

          player.lastSeen = Date.now();
          await this.room.storage.put(playerKey, player);

          // Also track global level completion count
          const levelKey = `ice_level_completions_${level}`;
          const levelCount = (await this.room.storage.get(levelKey) as number) || 0;
          await this.room.storage.put(levelKey, levelCount + 1);

          console.log(`[IceSkater] Player ${playerId.slice(0,8)} completed level ${level}`);

          return new Response(JSON.stringify({ success: true }), {
            headers: { ...headers, 'Content-Type': 'application/json' }
          });
        }

        if (data.type === 'ice_skater_get_analytics') {
          // Get analytics summary
          const players: any[] = [];
          const playerKeys = await this.room.storage.list({ prefix: 'ice_player_' });

          for (const [key, value] of playerKeys) {
            players.push(value);
          }

          // Get level completion counts
          const levelCompletions: { [level: number]: number } = {};
          const levelKeys = await this.room.storage.list({ prefix: 'ice_level_completions_' });
          for (const [key, value] of levelKeys) {
            const levelNum = parseInt(key.replace('ice_level_completions_', ''));
            levelCompletions[levelNum] = value as number;
          }

          const now = Date.now();
          const day = 24 * 60 * 60 * 1000;

          // Calculate stats
          const uniquePlayers = players.length;
          const todayPlayers = players.filter(p => now - p.lastSeen < day).length;
          const totalCompletions = players.reduce((sum, p) => sum + (p.completedLevels?.length || 0), 0);

          // Distribution of highest levels reached
          const levelDistribution: { [level: number]: number } = {};
          for (const p of players) {
            const lvl = p.highestLevel || 0;
            levelDistribution[lvl] = (levelDistribution[lvl] || 0) + 1;
          }

          // Recent players (last 20)
          const recentPlayers = players
            .sort((a, b) => b.lastSeen - a.lastSeen)
            .slice(0, 20)
            .map(p => ({
              id: p.id.slice(0, 8) + '...',
              highestLevel: p.highestLevel,
              completions: p.completedLevels?.length || 0,
              visits: p.visits,
              lastSeen: new Date(p.lastSeen).toISOString(),
              country: p.country
            }));

          return new Response(JSON.stringify({
            uniquePlayers,
            todayPlayers,
            totalCompletions,
            levelDistribution: Object.entries(levelDistribution).sort((a, b) => parseInt(a[0]) - parseInt(b[0])),
            levelCompletions: Object.entries(levelCompletions).sort((a, b) => parseInt(a[0]) - parseInt(b[0])),
            recentPlayers
          }), {
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
