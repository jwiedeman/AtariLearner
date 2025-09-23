"""Neural agent used by :mod:`atari_learner`.

The original repository shipped a tiny random-action scaffold whose main
purpose was to exercise the multiprocessing/data sharing machinery.  The module
now provides a fully functional learner that:

* Trains an Atari-style convolutional Q-network in place while the runner is
  executing.
* Maintains independent output heads for every game so that the policy can
  specialise per title (multiple instances of the same game reuse the same
  weights).
* Streams experience into a replay buffer and performs batched TD updates.
* Periodically checkpoints model/optimiser/replay state so that training can be
  resumed without losing context.

The implementation favours readability and robustness over raw performance – it
is intentionally compact so that it can serve as a good starting point for more
advanced agents.
"""

from __future__ import annotations

import datetime as _dt
import os
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Agent", "AgentState"]


# ---------------------------------------------------------------------------
# Utility structures
# ---------------------------------------------------------------------------


@dataclass
class AgentState:
    """Small collection of counters persisted alongside the model."""

    total_steps: int = 0
    episodes_seen: int = 0
    updates: int = 0
    last_checkpoint_step: int = 0


class ReplayBuffer:
    """Fixed-size replay buffer stored on CPU-friendly tensors."""

    def __init__(self, capacity: int, obs_shape: Sequence[int], device: torch.device) -> None:
        self.capacity = int(capacity)
        self.sample_device = torch.device(device)
        self.storage_device = torch.device("cpu") if self.sample_device.type == "cuda" else self.sample_device
        self.obs_shape = tuple(int(x) for x in obs_shape)

        self.states = torch.empty(
            (self.capacity, *self.obs_shape), dtype=torch.uint8, device=self.storage_device
        )
        self.next_states = torch.empty_like(self.states)
        self.actions = torch.empty(self.capacity, dtype=torch.int64, device=self.storage_device)
        self.rewards = torch.empty(self.capacity, dtype=torch.float32, device=self.storage_device)
        self.dones = torch.empty(self.capacity, dtype=torch.float32, device=self.storage_device)
        self.head_indices = torch.empty(self.capacity, dtype=torch.int64, device=self.storage_device)

        self.size = 0
        self.cursor = 0

    # ------------------------------------------------------------------
    def add(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        head_index: int,
    ) -> None:
        idx = self.cursor
        self.states[idx].copy_(
            torch.clamp(state, 0.0, 1.0).mul(255.0).to(
                dtype=torch.uint8, device=self.storage_device, non_blocking=True
            )
        )
        self.next_states[idx].copy_(
            torch.clamp(next_state, 0.0, 1.0).mul(255.0).to(
                dtype=torch.uint8, device=self.storage_device, non_blocking=True
            )
        )
        self.actions[idx] = int(action)
        self.rewards[idx] = float(reward)
        self.dones[idx] = 1.0 if done else 0.0
        self.head_indices[idx] = int(head_index)

        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # ------------------------------------------------------------------
    def sample(
        self, batch_size: int, generator: torch.Generator, to_device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        if self.size < batch_size:
            raise ValueError("Not enough samples to draw a batch")

        indices = torch.randint(
            0, self.size, (batch_size,), generator=generator, device=self.storage_device
        )
        return (
            self.states.index_select(0, indices)
            .to(to_device, dtype=torch.float32, non_blocking=True)
            .div_(255.0),
            self.actions.index_select(0, indices).to(to_device, dtype=torch.int64, non_blocking=True),
            self.rewards.index_select(0, indices).to(to_device, non_blocking=True),
            self.next_states.index_select(0, indices)
            .to(to_device, dtype=torch.float32, non_blocking=True)
            .div_(255.0),
            self.dones.index_select(0, indices).to(to_device, non_blocking=True),
            self.head_indices.index_select(0, indices).to(to_device, dtype=torch.int64, non_blocking=True),
        )

    # ------------------------------------------------------------------
    def state_dict(self, limit: Optional[int] = None) -> dict:
        """Return a serialisable snapshot of the buffer."""

        if self.size == 0:
            return {
                "capacity": self.capacity,
                "obs_shape": self.obs_shape,
                "size": 0,
                "cursor": 0,
            }

        if limit is not None:
            limit = max(1, int(limit))
            count = min(self.size, limit)
        else:
            count = self.size

        start = (self.cursor - count) % self.capacity
        order = (torch.arange(count, device=self.storage_device) + start) % self.capacity

        payload = {
            "capacity": self.capacity,
            "obs_shape": self.obs_shape,
            "size": count,
            "cursor": count % self.capacity,
            "states": self.states.index_select(0, order).to("cpu"),
            "next_states": self.next_states.index_select(0, order).to("cpu"),
            "actions": self.actions.index_select(0, order).to("cpu"),
            "rewards": self.rewards.index_select(0, order).to("cpu"),
            "dones": self.dones.index_select(0, order).to("cpu"),
            "head_indices": self.head_indices.index_select(0, order).to("cpu"),
        }
        return payload

    # ------------------------------------------------------------------
    def load_state_dict(self, payload: dict, device: torch.device) -> None:
        """Restore buffer contents from :meth:`state_dict` output."""

        capacity = int(payload.get("capacity", self.capacity))
        obs_shape = tuple(int(x) for x in payload.get("obs_shape", self.obs_shape))

        if capacity != self.capacity or obs_shape != self.obs_shape or torch.device(device) != self.sample_device:
            self.__init__(capacity, obs_shape, device)

        size = int(payload.get("size", 0))
        if size <= 0:
            self.size = 0
            self.cursor = 0
            return

        states_src = payload.get("states")
        if states_src is not None:
            states_cpu = states_src.to(device=self.storage_device)
            if states_cpu.dtype == torch.uint8:
                self.states[:size].copy_(states_cpu)
            else:
                self.states[:size].copy_(
                    torch.clamp(states_cpu.to(dtype=torch.float32), 0.0, 1.0).mul(255.0).to(dtype=torch.uint8)
                )

        next_states_src = payload.get("next_states")
        if next_states_src is not None:
            next_states_cpu = next_states_src.to(device=self.storage_device)
            if next_states_cpu.dtype == torch.uint8:
                self.next_states[:size].copy_(next_states_cpu)
            else:
                self.next_states[:size].copy_(
                    torch.clamp(next_states_cpu.to(dtype=torch.float32), 0.0, 1.0)
                    .mul(255.0)
                    .to(dtype=torch.uint8)
                )
        self.actions[:size].copy_(payload["actions"].to(device=self.storage_device, dtype=torch.int64))
        self.rewards[:size].copy_(payload["rewards"].to(device=self.storage_device, dtype=torch.float32))
        self.dones[:size].copy_(payload["dones"].to(device=self.storage_device, dtype=torch.float32))
        self.head_indices[:size].copy_(payload["head_indices"].to(device=self.storage_device, dtype=torch.int64))

        self.size = size
        self.cursor = int(payload.get("cursor", size % self.capacity)) % self.capacity


class _Encoder(nn.Module):
    """Shared convolutional torso used by all heads."""

    def __init__(self, input_channels: int = 3, latent_dim: int = 512) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 84, 84)
            conv_out = self.conv(dummy)
            flat_dim = conv_out.view(1, -1).shape[1]

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, latent_dim),
            nn.ReLU(inplace=True),
        )
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 - torch docs style
        return self.proj(self.conv(x))


class _MultiHeadQNetwork(nn.Module):
    """Feature extractor shared across per-game output heads."""

    def __init__(self, num_actions: int) -> None:
        super().__init__()
        self.encoder = _Encoder()
        self.num_actions = int(num_actions)
        self.heads = nn.ModuleDict()

    # ------------------------------------------------------------------
    def ensure_heads(self, keys: Iterable[str]) -> List[str]:
        """Make sure heads for the requested game identifiers exist."""

        created: List[str] = []
        device = next(self.parameters()).device
        for key in keys:
            if key in self.heads:
                continue
            head = nn.Sequential(
                nn.Linear(self.encoder.latent_dim, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, self.num_actions),
            ).to(device)
            self.heads[key] = head
            created.append(key)
        return created

    # ------------------------------------------------------------------
    def forward(self, inputs: torch.Tensor, head_keys: Sequence[str]) -> torch.Tensor:
        if inputs.shape[0] != len(head_keys):
            raise ValueError("Batch size and head key list must match")

        if inputs.shape[0] == 0:
            return torch.empty((0, self.num_actions), device=inputs.device, dtype=inputs.dtype)

        unique_keys = list(dict.fromkeys(head_keys))
        self.ensure_heads(unique_keys)

        features = self.encoder(inputs)
        outputs = torch.empty((inputs.shape[0], self.num_actions), device=inputs.device, dtype=features.dtype)

        for key in unique_keys:
            indices = [i for i, candidate in enumerate(head_keys) if candidate == key]
            index_tensor = torch.tensor(indices, dtype=torch.long, device=inputs.device)
            outputs.index_copy_(0, index_tensor, self.heads[key](features.index_select(0, index_tensor)))

        return outputs


# ---------------------------------------------------------------------------
# Main agent implementation
# ---------------------------------------------------------------------------


class Agent:
    """Neural agent that learns directly from the runner's shared tensors."""

    MAX_ACTIONS: int = 18
    OBS_SHAPE = (3, 84, 84)

    # Training hyper-parameters (kept intentionally conservative).
    LEARNING_RATE = 1e-4
    REPLAY_CAPACITY = 50_000
    REPLAY_SNAPSHOT_LIMIT = 5_000
    BATCH_SIZE = 64
    GAMMA = 0.99
    LEARNING_STARTS = 2_000
    TARGET_UPDATE_INTERVAL = 2_000
    MAX_GRAD_NORM = 10.0
    EPSILON_START = 1.0
    EPSILON_FINAL = 0.05
    EPSILON_DECAY = 150_000

    def __init__(self, game_ids: Optional[Sequence[str]] = None, max_snapshots: int = 5) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state = AgentState()
        self.max_snapshots = max(0, int(max_snapshots))

        # Torch random generator dedicated to reproducible sampling.
        base_seed = torch.initial_seed()
        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(base_seed)
        self._cpu_rng = torch.Generator(device="cpu")
        self._cpu_rng.manual_seed(base_seed)

        self.policy = _MultiHeadQNetwork(self.MAX_ACTIONS).to(self.device)
        self.target_policy = _MultiHeadQNetwork(self.MAX_ACTIONS).to(self.device)

        self._game_to_head_index: Dict[str, int] = {}
        self._head_index_to_game: List[str] = []
        self.env_game_ids: List[str] = []
        self._env_head_indices: List[int] = []
        self._last_frame_counts: List[float] = []
        self._last_cumulative_rewards: List[float] = []
        self._last_done_flags: List[bool] = []
        self._previous_obs: List[Optional[torch.Tensor]] = []
        self._previous_actions: List[int] = []

        initial_games = list(dict.fromkeys(game_ids or ["global"]))
        self._register_games(initial_games)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.LEARNING_RATE)
        self.replay = ReplayBuffer(self.REPLAY_CAPACITY, self.OBS_SHAPE, self.device)

        self._sync_target_network()

        if game_ids:
            self.configure_envs(game_ids)

    # ------------------------------------------------------------------
    # Public API expected by ``atari_learner``
    # ------------------------------------------------------------------
    def act_and_learn(
        self, obs_tensor: torch.Tensor, info_tensor: torch.Tensor, action_tensor: torch.Tensor
    ) -> None:
        num_envs = obs_tensor.shape[0]
        if num_envs == 0:
            return

        if not self.env_game_ids or len(self.env_game_ids) != num_envs:
            self.configure_envs(self.env_game_ids or [self._head_index_to_game[0]] * num_envs)

        # Move observations to the model device and preprocess them.
        obs_processed = self._preprocess_observations(obs_tensor)

        head_keys = [self._head_index_to_game[idx] for idx in self._env_head_indices]

        epsilon = self._compute_epsilon()
        with torch.no_grad():
            q_values = self.policy(obs_processed, head_keys)
            greedy_actions = q_values.argmax(dim=1)
            random_actions = torch.randint(
                0,
                self.MAX_ACTIONS,
                (num_envs,),
                generator=self._rng,
                device=self.device,
            )
            exploration_mask = torch.rand((num_envs,), generator=self._rng, device=self.device) < epsilon
            actions = torch.where(exploration_mask, random_actions, greedy_actions).to(torch.int64)

        # Update replay buffer using the info tensor (kept on CPU for bookkeeping).
        self._ingest_transitions(obs_processed, info_tensor.to("cpu"), actions)

        self._learn()

        action_tensor.copy_(actions.to(action_tensor.device, dtype=action_tensor.dtype))

    def save(self, path: str) -> None:
        directory = os.path.dirname(os.path.abspath(path))
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        self.state.last_checkpoint_step = self.state.total_steps
        payload = {
            "agent_state": asdict(self.state),
            "policy": self.policy.state_dict(),
            "target_policy": self.target_policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "replay": self.replay.state_dict(limit=self.REPLAY_SNAPSHOT_LIMIT),
            "metadata": {
                "env_game_ids": list(self.env_game_ids),
                "head_index_to_game": list(self._head_index_to_game),
                "obs_shape": list(self.OBS_SHAPE),
                "replay_capacity": self.REPLAY_CAPACITY,
                "timestamp": _dt.datetime.utcnow().isoformat(timespec="seconds"),
                "version": 2,
            },
        }

        torch.save(payload, path)

        if self.max_snapshots > 0:
            base = os.path.splitext(os.path.basename(path))[0]
            snapshot = os.path.join(directory, f"{base}-{_dt.datetime.utcnow():%Y%m%d-%H%M%S}.pt")
            torch.save(payload, snapshot)
            self._prune_snapshots(directory, base)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        payload = torch.load(path, map_location="cpu")

        if "policy" not in payload or "optimizer" not in payload:
            raise ValueError(f"Checkpoint {path!r} is missing required keys")

        metadata = payload.get("metadata", {})
        head_keys = metadata.get("head_index_to_game") or []
        if head_keys:
            self._register_games(head_keys)

        self.policy.load_state_dict(payload["policy"])
        self.target_policy.load_state_dict(payload.get("target_policy", payload["policy"]))
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.LEARNING_RATE)
        self.optimizer.load_state_dict(payload["optimizer"])
        self._move_optimizer_state_to_device()

        replay_state = payload.get("replay")
        if isinstance(replay_state, dict):
            self.replay.load_state_dict(replay_state, self.device)

        agent_state = payload.get("agent_state")
        if isinstance(agent_state, dict):
            self.state = AgentState(**agent_state)

        env_ids = metadata.get("env_game_ids")
        if env_ids:
            self.configure_envs(env_ids)

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def configure_envs(self, game_ids: Sequence[str]) -> None:
        if not game_ids:
            raise ValueError("At least one environment must be provided")

        self._register_games(game_ids)

        self.env_game_ids = list(game_ids)
        self._env_head_indices = [self._game_to_head_index[g] for g in self.env_game_ids]

        num_envs = len(self.env_game_ids)
        self._last_frame_counts = [0.0 for _ in range(num_envs)]
        self._last_cumulative_rewards = [0.0 for _ in range(num_envs)]
        self._last_done_flags = [False for _ in range(num_envs)]
        self._previous_obs = [None for _ in range(num_envs)]
        self._previous_actions = [0 for _ in range(num_envs)]

    def _register_games(self, game_ids: Iterable[str]) -> None:
        new_keys = []
        for gid in game_ids:
            if gid not in self._game_to_head_index:
                self._game_to_head_index[gid] = len(self._head_index_to_game)
                self._head_index_to_game.append(gid)
                new_keys.append(gid)

        created = self.policy.ensure_heads(new_keys)
        if created:
            self.target_policy.ensure_heads(created)
            for key in created:
                self.target_policy.heads[key].load_state_dict(self.policy.heads[key].state_dict())

    # ------------------------------------------------------------------
    # Core learning logic
    # ------------------------------------------------------------------
    def _preprocess_observations(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        obs = obs_tensor.to(self.device, dtype=torch.float32, non_blocking=True) / 255.0
        obs = obs.permute(0, 3, 1, 2).contiguous()
        obs = F.interpolate(obs, size=self.OBS_SHAPE[1:], mode="bilinear", align_corners=False)
        return obs

    def _compute_epsilon(self) -> float:
        decay_progress = min(1.0, self.state.total_steps / float(self.EPSILON_DECAY))
        return float(self.EPSILON_FINAL + (self.EPSILON_START - self.EPSILON_FINAL) * (1.0 - decay_progress))

    def _ingest_transitions(
        self,
        current_obs: torch.Tensor,
        info_tensor_cpu: torch.Tensor,
        actions: torch.Tensor,
    ) -> None:
        num_envs = current_obs.shape[0]
        rewards = info_tensor_cpu[:, 0].tolist()
        frame_counts = info_tensor_cpu[:, 1].tolist()
        terminated = info_tensor_cpu[:, 2].tolist()
        truncated = info_tensor_cpu[:, 3].tolist()

        for env_index in range(num_envs):
            head_index = self._env_head_indices[env_index]
            prev_obs = self._previous_obs[env_index]
            prev_action = self._previous_actions[env_index]
            last_frames = self._last_frame_counts[env_index]
            last_reward = self._last_cumulative_rewards[env_index]
            last_done = self._last_done_flags[env_index]

            frame_count = frame_counts[env_index]
            reward_total = rewards[env_index]
            done_flag = bool(terminated[env_index] > 0.5 or truncated[env_index] > 0.5)

            if frame_count < last_frames:
                # Environment restarted outside of normal termination bookkeeping.
                last_frames = 0.0
                last_reward = 0.0

            new_frame = frame_count > last_frames
            new_done = done_flag and not last_done

            if prev_obs is not None and (new_frame or new_done):
                delta_reward = reward_total - last_reward
                self.replay.add(
                    prev_obs,
                    prev_action,
                    float(delta_reward),
                    current_obs[env_index].detach(),
                    done_flag,
                    head_index,
                )
                step_increase = max(1, int(frame_count - last_frames)) if new_frame else 1
                self.state.total_steps += step_increase
                if done_flag:
                    self.state.episodes_seen += 1

            self._previous_obs[env_index] = current_obs[env_index].detach()
            self._previous_actions[env_index] = int(actions[env_index].item())
            self._last_frame_counts[env_index] = float(frame_count)
            self._last_cumulative_rewards[env_index] = float(reward_total)
            self._last_done_flags[env_index] = done_flag

    def _learn(self) -> None:
        if self.replay.size < max(self.BATCH_SIZE, self.LEARNING_STARTS):
            return

        batch = self.replay.sample(self.BATCH_SIZE, self._cpu_rng, self.device)
        states, actions, rewards, next_states, dones, head_indices = batch

        head_keys = [self._head_index_to_game[int(idx)] for idx in head_indices.tolist()]

        q_values = self.policy(states, head_keys).gather(1, actions.view(-1, 1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_policy(next_states, head_keys).max(dim=1).values
            target = rewards + (1.0 - dones) * self.GAMMA * next_q

        loss = F.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.MAX_GRAD_NORM)
        self.optimizer.step()

        self.state.updates += 1
        if self.state.updates % self.TARGET_UPDATE_INTERVAL == 0:
            self._sync_target_network()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _sync_target_network(self) -> None:
        self.target_policy.ensure_heads(self.policy.heads.keys())
        self.target_policy.load_state_dict(self.policy.state_dict())

    def _move_optimizer_state_to_device(self) -> None:
        for state in self.optimizer.state.values():
            for key, value in list(state.items()):
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(self.device)

    def _prune_snapshots(self, directory: str, base_name: str) -> None:
        pattern = f"{base_name}-"
        snapshots = [
            os.path.join(directory, entry)
            for entry in os.listdir(directory)
            if entry.startswith(pattern) and entry.endswith(".pt")
        ]
        snapshots.sort(reverse=True)
        for old_path in snapshots[self.max_snapshots :]:
            try:
                os.remove(old_path)
            except OSError:
                pass

