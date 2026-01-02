import type * as Party from "partykit/server";

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

export default class LocomotServer implements Party.Server {
  state: GameState;
  cleanupInterval: ReturnType<typeof setInterval> | null = null;

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

  onConnect(conn: Party.Connection, ctx: Party.ConnectionContext) {
    const spawn = this.getSpawnPosition();
    const color = COLORS[this.state.players.size % COLORS.length];

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
      lastUpdate: Date.now()
    };

    this.state.players.set(conn.id, player);
    if (!this.state.hostId) {
      this.assignHost(conn.id);
    }

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

    // Notify others
    this.room.broadcast(JSON.stringify({
      type: 'player_left',
      playerId: conn.id
    }));

    if (this.state.hostId === conn.id) {
      const remainingIds = Array.from(this.state.players.keys()).sort();
      const nextHost = remainingIds.length > 0 ? remainingIds[0] : null;
      this.assignHost(nextHost);
    }

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
          this.room.broadcast(JSON.stringify({
            type: 'arena_request',
            fromId: sender.id
          }), [sender.id]); // Exclude requester
          console.log(`Arena request from ${sender.id} broadcast to others`);
          break;

        case 'enemy_state':
          // Host broadcasting enemy state + pickups - relay to all other clients
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

    // DEBUG: GET returns current server state
    if (req.method === 'GET') {
      const state = {
        timestamp: Date.now(),
        playerCount: this.state.players.size,
        enemyCount: this.state.enemies?.length || 0,
        pickupCount: this.state.pickups?.length || 0,
        enemies: (this.state.enemies || []).slice(0, 3).map(e => ({
          id: e.id,
          name: e.name,
          pos: e.segments?.[0] ? { x: e.segments[0].x, y: e.segments[0].y } : null,
          len: e.segments?.length || 0
        }))
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
