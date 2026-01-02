# Locomot.io Handoff

## What changed
- Multiplayer pickups now stay in sync across clients (IDs + collection events + host-only spawning).
- PartyKit server assigns and migrates host ownership when the current host leaves, ensuring the arena stays alive.
- `player_history.json` is ignored and removed from git (runtime data only).

## Code changes
- `/home/ivy/locomot-io/index.html`
  - Added pickup IDs + helper (`addPickup`, `serializePickup`, `removePickupById`, `syncPickupIdCounter`).
  - Pickup syncing: full pickup payload in `arena_sync` and `enemy_state` broadcasts.
  - Clients send `pickup_collected` and remove pickups on receipt.
  - Host-only pickup spawning in `updateWaves` (clients no longer generate their own).
  - Magnet in multiplayer uses a larger pickup range (no pickup position drift).
  - Host selection now comes from server (`host_assigned`) instead of client-side lowest-id heuristic.

- `/home/ivy/locomot-io/server/server.ts`
  - Added `hostId` to game state.
  - Server assigns host on connect and reassigns when host leaves.
  - Broadcasts `host_assigned` to all clients.
  - Relays `pickup_collected` events to all clients.

- `/home/ivy/locomot-io/.gitignore`
  - Added `player_history.json`.

## Commits
- `ba5edc1` Fix multiplayer pickup sync
- `8cc8d03` Ignore player history data
- `e7da472` Add server-assigned host migration

## Deployments
- PartyKit deployed: `https://locomot-io.savecharlie.partykit.dev`

## How to verify
1) Open two clients in the same room.
2) Pickups should appear in the same positions.
3) When one player picks up an item, it disappears for the other.
4) Close the host tab; remaining player should become host and arena should continue.
5) Hard refresh clients (`Ctrl+Shift+R`) to bypass SW cache if behavior looks stale.

## Notes
- Pickup collection is still client-authoritative; server only relays events. If you want anti-cheat, move pickup validation to the server and broadcast authoritative pickup lists.
