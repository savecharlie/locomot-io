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
}

const COLORS = ['#f0f', '#0ff', '#ff0', '#f80', '#8f0', '#08f', '#f08', '#80f'];
const WORLD_COLS = 200;
const WORLD_ROWS = 150;

export default class LocomotServer implements Party.Server {
  state: GameState;

  constructor(readonly room: Party.Room) {
    this.state = {
      players: new Map(),
      pickups: this.generatePickups(50),
      worldSize: { cols: WORLD_COLS, rows: WORLD_ROWS }
    };
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

    // Send initial state to new player
    conn.send(JSON.stringify({
      type: 'init',
      playerId: conn.id,
      player,
      players: Array.from(this.state.players.values()),
      pickups: this.state.pickups,
      worldSize: this.state.worldSize
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
          // Broadcast hit to the target player
          this.room.broadcast(JSON.stringify({
            type: 'hit',
            targetId: data.targetId,
            damage: data.damage,
            fromId: sender.id
          }));
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

  // Handle HTTP requests (for training data upload)
  async onRequest(req: Party.Request) {
    // Enable CORS
    const headers = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type'
    };

    if (req.method === 'OPTIONS') {
      return new Response(null, { headers });
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
