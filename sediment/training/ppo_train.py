"""PPO trainer for Sediment.

Trains a neural network agent using Proximal Policy Optimization
with per-step rewards. Communicates with the Node.js step server
which runs the exact game physics.

Usage:
    python3 ppo_train.py                  # Train from scratch
    python3 ppo_train.py --resume ppo.pt  # Resume
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Network ──

class ActorCritic(nn.Module):
    """Shared-backbone actor-critic for PPO."""

    def __init__(self, obs_dim=72, act_dim=5):
        super().__init__()
        # Match browser SmallNN: 72→48→32→5
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 48),
            nn.ReLU(),
            nn.Linear(48, 32),
            nn.ReLU(),
        )
        self.actor = nn.Linear(32, act_dim)
        self.critic = nn.Linear(32, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)

    def get_action(self, obs):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value.squeeze(-1)

    def evaluate(self, obs, actions):
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values.squeeze(-1)

    def export_for_browser(self, path):
        """Export weights as JSON matching the browser SmallNN format."""
        # Both PPO and browser use 72→48→32→5 architecture
        # We export the actor head only (not the critic)
        sd = self.state_dict()
        weights = {
            'w1': sd['shared.0.weight'].tolist(),
            'b1': sd['shared.0.bias'].tolist(),
            'w2': sd['shared.2.weight'].tolist(),
            'b2': sd['shared.2.bias'].tolist(),
            'w3': sd['actor.weight'].tolist(),
            'b3': sd['actor.bias'].tolist(),
        }
        with open(path, 'w') as f:
            json.dump(weights, f)


# ── Environment wrapper ──

class SedimentEnv:
    """Communicates with Node.js step server via stdin/stdout JSON lines."""

    def __init__(self, level_num=2):
        self.level_num = level_num
        self.proc = subprocess.Popen(
            ['node', os.path.join(SCRIPT_DIR, 'step_server.js')],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        # Wait for ready message
        ready = self.proc.stderr.readline()
        if 'ready' not in ready:
            raise RuntimeError(f"Step server failed to start: {ready}")

    def _send(self, msg):
        self.proc.stdin.write(json.dumps(msg) + '\n')
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        return json.loads(line)

    def reset(self):
        result = self._send({'cmd': 'reset', 'levelNum': self.level_num})
        return np.array(result['state'], dtype=np.float32)

    def step(self, action):
        result = self._send({'cmd': 'step', 'action': int(action)})
        state = np.array(result['state'], dtype=np.float32)
        return state, result['reward'], result['done'], result.get('info', {})

    def close(self):
        self.proc.terminate()
        self.proc.wait()


# ── PPO ──

class PPOTrainer:
    def __init__(
        self,
        env,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        epochs_per_update=4,
        batch_size=64,
        rollout_steps=2048,
        entropy_coef=0.05,
        value_coef=0.5,
        max_grad_norm=0.5,
    ):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.epochs_per_update = epochs_per_update
        self.batch_size = batch_size
        self.rollout_steps = rollout_steps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        self.model = ActorCritic(obs_dim=72, act_dim=5)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.total_steps = 0
        self.episode_count = 0
        self.best_distance = 0

    def collect_rollout(self):
        """Collect rollout_steps of experience."""
        obs_buf = []
        act_buf = []
        rew_buf = []
        done_buf = []
        logp_buf = []
        val_buf = []

        obs = self.env.reset()
        ep_reward = 0
        ep_distance = 0
        ep_count = 0

        for _ in range(self.rollout_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action, logp, value = self.model.get_action(obs_t)

            next_obs, reward, done, info = self.env.step(action)

            obs_buf.append(obs)
            act_buf.append(action)
            rew_buf.append(reward)
            done_buf.append(done)
            logp_buf.append(logp.item())
            val_buf.append(value.item())

            ep_reward += reward
            self.total_steps += 1

            if done:
                ep_distance = info.get('distance', 0)
                ep_count += 1
                self.episode_count += 1
                if ep_distance > self.best_distance:
                    self.best_distance = ep_distance
                obs = self.env.reset()
                ep_reward = 0
            else:
                obs = next_obs

        # Compute GAE advantages
        with torch.no_grad():
            last_val = self.model(torch.FloatTensor(obs).unsqueeze(0))[1].item()

        advantages = np.zeros(self.rollout_steps, dtype=np.float32)
        returns = np.zeros(self.rollout_steps, dtype=np.float32)
        gae = 0
        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                next_val = last_val
                next_done = 0
            else:
                next_val = val_buf[t + 1]
                next_done = done_buf[t + 1]

            delta = rew_buf[t] + self.gamma * next_val * (1 - done_buf[t]) - val_buf[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - done_buf[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + val_buf[t]

        return {
            'obs': torch.FloatTensor(np.array(obs_buf)),
            'actions': torch.LongTensor(act_buf),
            'logprobs': torch.FloatTensor(logp_buf),
            'returns': torch.FloatTensor(returns),
            'advantages': torch.FloatTensor(advantages),
            'ep_count': ep_count,
            'ep_distance': ep_distance,
        }

    def update(self, rollout):
        obs = rollout['obs']
        actions = rollout['actions']
        old_logprobs = rollout['logprobs']
        returns = rollout['returns']
        advantages = rollout['advantages']

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0
        n_updates = 0

        for _ in range(self.epochs_per_update):
            # Shuffle indices
            indices = torch.randperm(len(obs))

            for start in range(0, len(obs), self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]

                new_logprobs, entropy, new_values = self.model.evaluate(
                    obs[idx], actions[idx]
                )

                # Policy loss (clipped)
                ratio = torch.exp(new_logprobs - old_logprobs[idx])
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(new_values, returns[idx])

                # Entropy bonus
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                n_updates += 1

        return total_loss / max(1, n_updates)

    def save(self, path):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'episode_count': self.episode_count,
            'best_distance': self.best_distance,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.total_steps = ckpt['total_steps']
        self.episode_count = ckpt['episode_count']
        self.best_distance = ckpt.get('best_distance', 0)

    def train(self, total_updates=500, checkpoint_dir='.', export_every=10):
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"PPO Training | Rollout: {self.rollout_steps} steps | Level: {self.env.level_num}")
        print("-" * 70)

        for update_i in range(total_updates):
            t0 = time.time()

            rollout = self.collect_rollout()
            loss = self.update(rollout)

            elapsed = time.time() - t0
            eps = rollout['ep_count']
            dist = rollout['ep_distance']

            # Progress bar
            TILE = 12; SEGMENT_W = 40; segs = 4
            goal_x = (segs * SEGMENT_W - 4) * TILE
            pct = min(100, self.best_distance * TILE / goal_x * 100)
            bar_len = 30
            filled = int(bar_len * pct / 100)
            bar = '█' * filled + '░' * (bar_len - filled)

            print(
                f"Update {update_i+1:4d} |{bar}| "
                f"best:{self.best_distance:4.0f} last:{dist:4.0f} | "
                f"eps:{eps:3d} steps:{self.total_steps:>8d} | "
                f"loss:{loss:.3f} | {elapsed:.1f}s"
            )
            sys.stdout.flush()

            # Export weights for browser preview
            if (update_i + 1) % export_every == 0:
                export_path = os.path.join(checkpoint_dir, 'ppo_weights.json')
                self.model.export_for_browser(export_path)

            # Checkpoint
            if (update_i + 1) % 50 == 0:
                self.save(os.path.join(checkpoint_dir, 'ppo.pt'))

        # Final save
        self.save(os.path.join(checkpoint_dir, 'ppo.pt'))
        self.model.export_for_browser(os.path.join(checkpoint_dir, 'ppo_weights.json'))
        print(f"\nDone. Best distance: {self.best_distance}")


def main():
    parser = argparse.ArgumentParser(description='PPO trainer for Sediment')
    parser.add_argument('--level', type=int, default=2)
    parser.add_argument('--updates', type=int, default=500)
    parser.add_argument('--rollout-steps', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--checkpoint-dir', type=str, default='.')
    args = parser.parse_args()

    env = SedimentEnv(level_num=args.level)

    trainer = PPOTrainer(
        env,
        lr=args.lr,
        rollout_steps=args.rollout_steps,
    )

    if args.resume:
        trainer.load(args.resume)
        print(f"Resumed from {args.resume} (step {trainer.total_steps})")

    try:
        trainer.train(
            total_updates=args.updates,
            checkpoint_dir=args.checkpoint_dir,
        )
    finally:
        env.close()


if __name__ == '__main__':
    main()
