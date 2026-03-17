import type * as Party from "partykit/server";

import {
  LIVE_BOT_COUNT,
  TEAM_SPECS,
  addBotPlayer,
  addHumanPlayer,
  botCount,
  createSnapshot,
  createSimulation,
  getActor,
  hasCapacityForHuman,
  humanCount,
  removeActor,
  resetMatch,
  setActorName,
  setPlayerInput,
  stepSimulation,
  type GameMode,
  type SimulationState
} from "../../../packages/core/src/index";
import {
  createInitMessage,
  createUpdateMessage,
  encodeMessage,
  parseClientMessage
} from "../../../packages/protocol/src/index";

function modeForRoom(roomId: string): GameMode {
  return roomId.toLowerCase().startsWith("team-") ? "teams" : "ffa";
}

function defaultPlayerName(index: number): string {
  return `squid-${index + 1}`;
}

/** Parse hex color to hue (0-360) */
function hexToHue(hex: string): number {
  const r = parseInt(hex.slice(1, 3), 16) / 255;
  const g = parseInt(hex.slice(3, 5), 16) / 255;
  const b = parseInt(hex.slice(5, 7), 16) / 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const d = max - min;
  if (d === 0) return 0;
  let h = 0;
  if (max === r) h = ((g - b) / d + 6) % 6;
  else if (max === g) h = (b - r) / d + 2;
  else h = (r - g) / d + 4;
  return h * 60;
}

/** Hue distance (circular, 0-180) */
function hueDist(a: number, b: number): number {
  const d = Math.abs(a - b) % 360;
  return d > 180 ? 360 - d : d;
}

/** Reassign bot colors to be maximally distinct from all humans and each other */
function deconflictColors(sim: SimulationState, protectId: string) {
  const actors = Array.from(sim.actors.values());
  const humans = actors.filter(a => !a.isBot || a.id === protectId);
  const bots = actors.filter(a => a.isBot && a.id !== protectId);

  const takenHues = humans.map(a => hexToHue(a.color));

  for (const bot of bots) {
    const botHue = hexToHue(bot.color);
    const tooClose = takenHues.some(h => hueDist(h, botHue) < 30);
    if (!tooClose) {
      takenHues.push(botHue);
      continue;
    }
    // Find the hue maximally distant from all taken hues
    let bestHue = 0, bestMinDist = 0;
    for (let h = 0; h < 360; h += 5) {
      const minDist = Math.min(...takenHues.map(t => hueDist(t, h)));
      if (minDist > bestMinDist) {
        bestMinDist = minDist;
        bestHue = h;
      }
    }
    // Convert hue to hex (same saturation/lightness as game palette)
    const s = 0.80, l = 0.62;
    const a = s * Math.min(l, 1 - l);
    const f = (n: number) => {
      const k = (n + (bestHue / 360) * 12) % 12;
      const color = l - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
      return Math.round(255 * color).toString(16).padStart(2, "0");
    };
    bot.color = `#${f(0)}${f(8)}${f(4)}`;
    takenHues.push(bestHue);
  }
}

export default class RecoilLoveRoom implements Party.Server {
  private readonly room: Party.Room;
  private readonly simulation;
  private readonly tickHandle: ReturnType<typeof setInterval>;

  constructor(room: Party.Room) {
    this.room = room;
    this.simulation = createSimulation({
      roomId: room.id,
      mode: modeForRoom(room.id),
      localPlayerId: "__server__",
      localPlayerName: "server",
      botCount: 0
    });
    removeActor(this.simulation, "__server__");
    this.reconcileBots();

    let prevWinnerId: string | null = null;
    this.tickHandle = setInterval(() => {
      stepSimulation(this.simulation, 1);

      // Detect match reset — send full init with new maze to all clients
      if (prevWinnerId && !this.simulation.winnerId) {
        for (const conn of this.room.getConnections()) {
          this.sendInit(conn);
        }
      }
      prevWinnerId = this.simulation.winnerId;

      this.broadcastUpdate();
    }, 1000 / 60);
  }

  onConnect(connection: Party.Connection): void {
    if (!hasCapacityForHuman(this.simulation)) {
      connection.send(
        encodeMessage({
          type: "event",
          event: {
            id: -1,
            text: "room is full",
            atTick: this.simulation.tick
          }
        })
      );
      connection.close();
      return;
    }

    const humanIndex = humanCount(this.simulation);
    addHumanPlayer(this.simulation, connection.id, defaultPlayerName(humanIndex));
    this.reconcileBots();
    this.sendInit(connection);
    this.broadcastUpdate();
  }

  onMessage(raw: string | ArrayBuffer, sender: Party.Connection): void {
    const message = parseClientMessage(String(raw));
    if (!message) {
      return;
    }

    if (message.type === "ping") {
      sender.send(
        encodeMessage({
          type: "pong",
          sentAt: message.sentAt
        })
      );
      return;
    }

    if (message.type === "join") {
      const trimmed = message.name.replace(/[^a-z]/gi, "").toLowerCase().slice(0, 3);
      if (trimmed) {
        setActorName(this.simulation, sender.id, trimmed);
      }
      // Apply player's chosen color (FFA only — teams forces team color)
      if (message.color && this.simulation.config.mode !== "teams") {
        const actor = getActor(this.simulation, sender.id);
        if (actor) {
          actor.color = message.color;
          deconflictColors(this.simulation, sender.id);
        }
      }
      this.sendInit(sender);
      this.broadcastUpdate();
      return;
    }

    if (message.type === "restart" && this.simulation.winnerId) {
      resetMatch(this.simulation);
      // Send fresh init with new maze to all clients
      for (const conn of this.room.getConnections()) {
        this.sendInit(conn);
      }
      this.broadcastUpdate();
      return;
    }

    setPlayerInput(this.simulation, sender.id, message.input);
  }

  onClose(connection: Party.Connection): void {
    removeActor(this.simulation, connection.id);
    this.reconcileBots();
    this.broadcastUpdate();
  }

  onError(connection: Party.Connection, error: Error): void {
    connection.send(
      encodeMessage({
        type: "event",
        event: {
          id: -1,
          text: `room error: ${error.message}`,
          atTick: this.simulation.tick
        }
      })
    );
  }

  onRequest(_request: Party.Request): Response {
    return new Response("Inkjet room is alive.", {
      status: 200,
      headers: {
        "content-type": "text/plain; charset=utf-8"
      }
    });
  }

  private sendInit(connection: Party.Connection) {
    const snapshot = createSnapshot(this.simulation, {
      localPlayerId: connection.id
    });
    connection.send(encodeMessage(createInitMessage(snapshot)));
  }

  private broadcastUpdate() {
    const snapshot = createSnapshot(this.simulation);
    this.room.broadcast(encodeMessage(createUpdateMessage(snapshot)));
  }

  private reconcileBots() {
    const desiredBots = Math.max(0, LIVE_BOT_COUNT - humanCount(this.simulation));

    while (botCount(this.simulation) < desiredBots) {
      addBotPlayer(this.simulation);
    }

    while (botCount(this.simulation) > desiredBots) {
      const bot = Array.from(this.simulation.actors.values()).find((actor) => actor.isBot);
      if (!bot) {
        break;
      }
      removeActor(this.simulation, bot.id);
    }

    // Force team colors in teams mode
    if (this.simulation.config.mode === "teams") {
      for (const actor of this.simulation.actors.values()) {
        if (actor.teamId !== null) {
          const spec = TEAM_SPECS.find(t => t.teamId === actor.teamId);
          if (spec) actor.color = spec.color;
        }
      }
    }
  }
}
