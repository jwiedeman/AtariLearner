"""Ensemble of per-game Double DQN agents for Atari learning.

This module implements an ensemble architecture where each game has its own
independent Double DQN model. This design enables:

1. **Specialisation**: Each model focuses entirely on one game
2. **Independent checkpointing**: Best models saved per-game
3. **Scalability**: Models can be trained/deployed independently
4. **N64 deployment**: Small individual models suitable for embedded systems

**Algorithm (per-game):**
    - Double DQN to reduce overestimation bias (van Hasselt et al., 2016)
    - Experience replay with uniform sampling (Mnih et al., 2015)
    - Reward clipping to [-1, 1] for stable gradients

**Architecture (per-game):**
    - Convolutional encoder (84x84 RGB input)
    - Single output head (18 actions max)
    - Huber loss for robust TD updates

References:
    Mnih, V., et al. (2015). Human-level control through deep reinforcement
        learning. Nature, 518(7540), 529-533.
    van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning
        with double Q-learning. AAAI Conference on Artificial Intelligence.
"""

from __future__ import annotations

import datetime as _dt
import os
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

# Optional TensorBoard support - graceful degradation if not installed.
try:
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    _SummaryWriter = None  # type: ignore[misc,assignment]
    TENSORBOARD_AVAILABLE = False

__all__ = ["Agent", "AgentState"]


def _get_device(preference: str = "auto") -> torch.device:
    """Select the best available device (CUDA > MPS > CPU)."""
    if preference == "cuda" or (preference == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    if preference == "mps" or (preference == "auto" and torch.backends.mps.is_available()):
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Per-game state tracking
# ---------------------------------------------------------------------------


@dataclass
class GameState:
    """Training state for a single game."""

    total_steps: int = 0
    episodes_seen: int = 0
    updates: int = 0
    best_episode_reward: float = float("-inf")
    last_checkpoint_step: int = 0


@dataclass
class AgentState:
    """Aggregate state across all games (for compatibility)."""

    total_steps: int = 0
    episodes_seen: int = 0
    updates: int = 0
    last_checkpoint_step: int = 0


# ---------------------------------------------------------------------------
# Replay Buffer (per-game, simpler without head_indices)
# ---------------------------------------------------------------------------


class ReplayBuffer:
    """Fixed-size replay buffer for a single game."""

    def __init__(self, capacity: int, obs_shape: Sequence[int], device: torch.device) -> None:
        self.capacity = int(capacity)
        self.sample_device = torch.device(device)
        # Store on CPU if using GPU to save VRAM
        self.storage_device = torch.device("cpu") if self.sample_device.type in ("cuda", "mps") else self.sample_device
        self.obs_shape = tuple(int(x) for x in obs_shape)

        self.states = torch.empty((self.capacity, *self.obs_shape), dtype=torch.uint8, device=self.storage_device)
        self.next_states = torch.empty_like(self.states)
        self.actions = torch.empty(self.capacity, dtype=torch.int64, device=self.storage_device)
        self.rewards = torch.empty(self.capacity, dtype=torch.float32, device=self.storage_device)
        self.dones = torch.empty(self.capacity, dtype=torch.float32, device=self.storage_device)

        self.size = 0
        self.cursor = 0

    def add(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor, done: bool) -> None:
        idx = self.cursor
        self.states[idx].copy_(
            torch.clamp(state, 0.0, 1.0).mul(255.0).to(dtype=torch.uint8, device=self.storage_device, non_blocking=True)
        )
        self.next_states[idx].copy_(
            torch.clamp(next_state, 0.0, 1.0)
            .mul(255.0)
            .to(dtype=torch.uint8, device=self.storage_device, non_blocking=True)
        )
        self.actions[idx] = int(action)
        self.rewards[idx] = float(reward)
        self.dones[idx] = 1.0 if done else 0.0

        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: torch.Generator, device: torch.device):
        indices = torch.randint(0, self.size, (batch_size,), generator=rng, device="cpu")
        states = self.states[indices].to(device=device, dtype=torch.float32).div(255.0)
        next_states = self.next_states[indices].to(device=device, dtype=torch.float32).div(255.0)
        actions = self.actions[indices].to(device=device)
        rewards = self.rewards[indices].to(device=device)
        dones = self.dones[indices].to(device=device)
        return states, actions, rewards, next_states, dones

    def state_dict(self, limit: Optional[int] = None) -> dict:
        if self.size == 0:
            return {"capacity": self.capacity, "obs_shape": self.obs_shape, "size": 0, "cursor": 0}

        count = min(self.size, limit) if limit else self.size
        start = (self.cursor - count) % self.capacity
        order = (torch.arange(count, device=self.storage_device) + start) % self.capacity

        return {
            "capacity": self.capacity,
            "obs_shape": self.obs_shape,
            "size": count,
            "cursor": count % self.capacity,
            "states": self.states.index_select(0, order).to("cpu"),
            "next_states": self.next_states.index_select(0, order).to("cpu"),
            "actions": self.actions.index_select(0, order).to("cpu"),
            "rewards": self.rewards.index_select(0, order).to("cpu"),
            "dones": self.dones.index_select(0, order).to("cpu"),
        }

    def load_state_dict(self, payload: dict, device: torch.device) -> None:
        size = int(payload.get("size", 0))
        if size <= 0:
            self.size = 0
            self.cursor = 0
            return

        states_src = payload.get("states")
        if states_src is not None:
            self.states[:size].copy_(states_src.to(device=self.storage_device, dtype=torch.uint8))

        next_src = payload.get("next_states")
        if next_src is not None:
            self.next_states[:size].copy_(next_src.to(device=self.storage_device, dtype=torch.uint8))

        self.actions[:size].copy_(payload["actions"].to(device=self.storage_device, dtype=torch.int64))
        self.rewards[:size].copy_(payload["rewards"].to(device=self.storage_device, dtype=torch.float32))
        self.dones[:size].copy_(payload["dones"].to(device=self.storage_device, dtype=torch.float32))

        self.size = size
        self.cursor = int(payload.get("cursor", size % self.capacity)) % self.capacity


# ---------------------------------------------------------------------------
# Single-game DQN Network
# ---------------------------------------------------------------------------


class SingleGameDQN(nn.Module):
    """Complete DQN network for a single game (encoder + output head)."""

    def __init__(self, num_actions: int = 18, input_channels: int = 3, latent_dim: int = 512) -> None:
        super().__init__()
        self.num_actions = num_actions

        # Convolutional encoder
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

        # Calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 84, 84)
            conv_out = self.conv(dummy)
            flat_dim = conv_out.view(1, -1).shape[1]

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


# ---------------------------------------------------------------------------
# Per-Game Agent
# ---------------------------------------------------------------------------


class PerGameAgent:
    """Independent DQN agent for a single Atari game."""

    # Hyperparameters (per-game, can be smaller than multi-game setup)
    LEARNING_RATE = 2.5e-4
    REPLAY_CAPACITY = 100_000  # 100K per game (vs 500K shared)
    REPLAY_SNAPSHOT_LIMIT = 5_000
    BATCH_SIZE = 32
    GAMMA = 0.99
    LEARNING_STARTS = 5_000
    TARGET_UPDATE_INTERVAL = 5_000
    MAX_GRAD_NORM = 10.0
    EPSILON_START = 1.0
    EPSILON_FINAL = 0.01
    EPSILON_DECAY = 500_000  # 500K per game (faster than multi-game)

    OBS_SHAPE = (3, 84, 84)
    MAX_ACTIONS = 18

    def __init__(self, game_id: str, device: torch.device) -> None:
        self.game_id = game_id
        self.device = device
        self.state = GameState()

        # Random generators
        base_seed = hash(game_id) % (2**32)
        self._rng = torch.Generator(device=device if device.type == "cuda" else "cpu")
        self._rng.manual_seed(base_seed)
        self._cpu_rng = torch.Generator(device="cpu")
        self._cpu_rng.manual_seed(base_seed)

        # Networks
        self.policy = SingleGameDQN(self.MAX_ACTIONS).to(device)
        self.target = SingleGameDQN(self.MAX_ACTIONS).to(device)
        self.target.load_state_dict(self.policy.state_dict())

        # Optimizer and replay
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.LEARNING_RATE)
        self.replay = ReplayBuffer(self.REPLAY_CAPACITY, self.OBS_SHAPE, device)

        # Episode tracking
        self._prev_obs: Optional[torch.Tensor] = None
        self._prev_action: int = 0
        self._last_frame_count: float = 0.0
        self._last_reward: float = 0.0
        self._last_done: bool = False
        self._episode_reward: float = 0.0

    def compute_epsilon(self) -> float:
        progress = min(1.0, self.state.total_steps / self.EPSILON_DECAY)
        return self.EPSILON_START + (self.EPSILON_FINAL - self.EPSILON_START) * progress

    def select_action(self, obs: torch.Tensor) -> int:
        epsilon = self.compute_epsilon()
        if torch.rand(1, generator=self._cpu_rng).item() < epsilon:
            return int(torch.randint(0, self.MAX_ACTIONS, (1,), generator=self._cpu_rng).item())

        with torch.no_grad():
            q_values = self.policy(obs.unsqueeze(0))
            return int(q_values.argmax(dim=1).item())

    def step(self, obs: torch.Tensor, reward: float, frame_count: float, done: bool) -> int:
        """Process one environment step, return action to take."""
        # Detect episode reset
        if frame_count < self._last_frame_count:
            self._last_frame_count = 0.0
            self._last_reward = 0.0
            self._episode_reward = 0.0

        # Store transition
        delta_reward = reward - self._last_reward
        clipped_reward = max(-1.0, min(1.0, delta_reward))

        if self._prev_obs is not None and frame_count > self._last_frame_count:
            self.replay.add(self._prev_obs, self._prev_action, clipped_reward, obs, done)
            self.state.total_steps += 1

        # Track episode reward (unclipped for evaluation)
        self._episode_reward += delta_reward

        if done:
            self.state.episodes_seen += 1
            # Track best episode for checkpointing
            if self._episode_reward > self.state.best_episode_reward:
                self.state.best_episode_reward = self._episode_reward
            self._episode_reward = 0.0

        # Select action
        action = self.select_action(obs)

        # Learn
        self._learn()

        # Update state
        self._prev_obs = obs.detach()
        self._prev_action = action
        self._last_frame_count = frame_count
        self._last_reward = reward
        self._last_done = done

        return action

    def _learn(self) -> None:
        if self.replay.size < max(self.BATCH_SIZE, self.LEARNING_STARTS):
            return

        batch = self.replay.sample(self.BATCH_SIZE, self._cpu_rng, self.device)
        states, actions, rewards, next_states, dones = batch

        # Current Q-values
        q_values = self.policy(states).gather(1, actions.view(-1, 1)).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            next_actions = self.policy(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target(next_states).gather(1, next_actions).squeeze(1)
            target = rewards + (1.0 - dones) * self.GAMMA * next_q

        loss = F.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.MAX_GRAD_NORM)
        self.optimizer.step()

        self.state.updates += 1
        if self.state.updates % self.TARGET_UPDATE_INTERVAL == 0:
            self.target.load_state_dict(self.policy.state_dict())

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "game_id": self.game_id,
            "state": asdict(self.state),
            "policy": self.policy.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "replay": self.replay.state_dict(limit=self.REPLAY_SNAPSHOT_LIMIT),
        }
        torch.save(payload, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)

        # Restore state
        for key, value in payload.get("state", {}).items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)

        # Restore networks
        self.policy.load_state_dict(payload["policy"])
        self.target.load_state_dict(payload["target"])

        # Restore optimizer (move to device)
        self.optimizer.load_state_dict(payload["optimizer"])
        for state in self.optimizer.state.values():
            for k, v in list(state.items()):
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        # Restore replay
        if "replay" in payload:
            self.replay.load_state_dict(payload["replay"], self.device)


# ---------------------------------------------------------------------------
# Ensemble Agent (manages all per-game agents)
# ---------------------------------------------------------------------------


class Agent:
    """Ensemble of per-game DQN agents.

    This class maintains independent models for each game and provides
    the same API as the original multi-head agent for compatibility with
    atari_learner.py.
    """

    OBS_SHAPE = (3, 84, 84)
    MAX_ACTIONS = 18

    def __init__(
        self,
        game_ids: Optional[Sequence[str]] = None,
        max_snapshots: int = 5,
        log_dir: Optional[str] = None,
        checkpoint_dir: str = "checkpoints",
        device: str = "auto",
    ) -> None:
        self.device = _get_device(device)
        print(f"[Agent] Using device: {self.device}")

        self.checkpoint_dir = checkpoint_dir
        self.max_snapshots = max_snapshots
        self.state = AgentState()

        # Per-game agents
        self._agents: Dict[str, PerGameAgent] = {}
        self._env_to_game: List[str] = []  # Maps env index to game_id

        # TensorBoard logging
        self._writer: Optional["SummaryWriter"] = None
        if TENSORBOARD_AVAILABLE and log_dir is not None:
            self._writer = _SummaryWriter(log_dir=log_dir)
            print(f"[Agent] TensorBoard logging enabled: {log_dir}")

        if game_ids:
            self.configure_envs(game_ids)

    def _get_or_create_agent(self, game_id: str) -> PerGameAgent:
        """Get existing agent or create new one for the game."""
        if game_id not in self._agents:
            agent = PerGameAgent(game_id, self.device)

            # Try to load existing checkpoint
            best_path = self._checkpoint_path(game_id, "best")
            latest_path = self._checkpoint_path(game_id, "latest")

            if os.path.exists(best_path):
                try:
                    agent.load(best_path)
                    print(f"[{game_id}] Loaded best checkpoint (reward={agent.state.best_episode_reward:.1f})")
                except Exception as e:
                    print(f"[{game_id}] Failed to load best checkpoint: {e}")
            elif os.path.exists(latest_path):
                try:
                    agent.load(latest_path)
                    print(f"[{game_id}] Loaded latest checkpoint")
                except Exception as e:
                    print(f"[{game_id}] Failed to load latest checkpoint: {e}")
            else:
                print(f"[{game_id}] Starting fresh agent")

            self._agents[game_id] = agent

        return self._agents[game_id]

    def _checkpoint_path(self, game_id: str, variant: str = "latest") -> str:
        """Generate checkpoint path for a game."""
        # Sanitize game_id for filesystem
        safe_name = game_id.replace("/", "_").replace(":", "_")
        return os.path.join(self.checkpoint_dir, safe_name, f"{variant}.pt")

    def configure_envs(self, game_ids: Sequence[str]) -> None:
        """Configure the agent to handle the specified environment list."""
        self._env_to_game = list(game_ids)
        for game_id in set(game_ids):
            self._get_or_create_agent(game_id)

    def _preprocess_observations(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """Resize and normalise observations."""
        obs = obs_tensor.to(self.device, dtype=torch.float32)
        if obs.dim() == 4 and obs.shape[-1] == 3:
            obs = obs.permute(0, 3, 1, 2)
        obs = obs / 255.0
        obs = F.interpolate(obs, size=(84, 84), mode="bilinear", align_corners=False)
        return obs

    def act_and_learn(
        self, obs_tensor: torch.Tensor, info_tensor: torch.Tensor, action_tensor: torch.Tensor
    ) -> None:
        """Process observations and select actions for all environments."""
        num_envs = obs_tensor.shape[0]
        if num_envs == 0:
            return

        # Preprocess all observations
        obs_processed = self._preprocess_observations(obs_tensor)
        info_cpu = info_tensor.to("cpu")

        # Process each environment
        for env_idx in range(num_envs):
            game_id = self._env_to_game[env_idx]
            agent = self._agents[game_id]

            obs = obs_processed[env_idx]
            reward = float(info_cpu[env_idx, 0].item())
            frame_count = float(info_cpu[env_idx, 1].item())
            terminated = info_cpu[env_idx, 2].item() > 0.5
            truncated = info_cpu[env_idx, 3].item() > 0.5
            done = terminated or truncated

            action = agent.step(obs, reward, frame_count, done)
            action_tensor[env_idx] = action

            # Log to TensorBoard on episode end
            if done and self._writer is not None:
                self._writer.add_scalar(
                    f"episode/{game_id}/reward",
                    agent._episode_reward if agent._episode_reward != 0 else reward,
                    agent.state.episodes_seen,
                )
                self._writer.add_scalar(f"episode/{game_id}/length", int(frame_count), agent.state.episodes_seen)

        # Update aggregate state
        self.state.total_steps = sum(a.state.total_steps for a in self._agents.values())
        self.state.episodes_seen = sum(a.state.episodes_seen for a in self._agents.values())
        self.state.updates = sum(a.state.updates for a in self._agents.values())

    def save(self, path: str) -> None:
        """Save all per-game agents."""
        for game_id, agent in self._agents.items():
            # Always save latest
            latest_path = self._checkpoint_path(game_id, "latest")
            agent.save(latest_path)

            # Save as best if this is the best episode
            best_path = self._checkpoint_path(game_id, "best")
            if not os.path.exists(best_path):
                agent.save(best_path)
            else:
                # Check if current is better
                try:
                    existing = torch.load(best_path, map_location="cpu", weights_only=False)
                    existing_best = existing.get("state", {}).get("best_episode_reward", float("-inf"))
                    if agent.state.best_episode_reward > existing_best:
                        agent.save(best_path)
                        print(f"[{game_id}] New best: {agent.state.best_episode_reward:.1f}")
                except Exception:
                    agent.save(best_path)

        # Save aggregate state for compatibility
        self.state.last_checkpoint_step = self.state.total_steps
        aggregate_path = os.path.join(self.checkpoint_dir, "ensemble_state.pt")
        os.makedirs(os.path.dirname(aggregate_path), exist_ok=True)
        torch.save(
            {
                "state": asdict(self.state),
                "games": list(self._agents.keys()),
                "timestamp": _dt.datetime.utcnow().isoformat(timespec="seconds"),
            },
            aggregate_path,
        )

    def load(self, path: str) -> None:
        """Load all per-game agents from checkpoint directory."""
        # The path argument is ignored; we load from checkpoint_dir
        # This maintains API compatibility
        for game_id in self._env_to_game:
            self._get_or_create_agent(game_id)  # Will auto-load best/latest

    def close(self) -> None:
        """Clean up resources."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
