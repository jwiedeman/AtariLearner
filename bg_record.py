"""Utility hooks used by :mod:`atari_learner` for lightweight monitoring.

The original demo streamed video/metric dashboards by delegating to an external
``bg_record`` module.  To keep this repository self-contained we provide a very
small implementation that focuses on metrics:

* Environment threads call :func:`bind_logger`, :func:`log_step` and
  :func:`log_close` to update a shared statistics tensor and to emit the
  occasional textual summary.
* A dedicated background process launched by ``atari_learner.py`` periodically
  polls the tensors and prints aggregated diagnostics.

The goal is to offer sensible default behaviour without pulling in heavy video
dependencies.  Users wishing to capture actual gameplay footage can replace
this module with their own recorder.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

__all__ = [
    "bind_logger",
    "log_step",
    "log_close",
    "bg_record_proc",
]


@dataclass
class _ThreadRecorderState:
    """Per-thread state updated by :func:`log_step`.

    ``info_tensor`` is a shared tensor (CPU or CUDA) with four statistics per
    environment: cumulative reward, frame count, terminated flag, and truncated
    flag.  The tensor is written to directly so that the learner can consume the
    data without any additional synchronisation primitives.
    """

    game_id: str
    env_index: int
    info_tensor: torch.Tensor
    episode_reward: float = 0.0
    frame_count: int = 0
    episodes_completed: int = 0
    last_print_ts: float = field(default_factory=time.time)

    def tensor_row(self) -> torch.Tensor:
        return self.info_tensor[self.env_index]


_THREAD_STATE = threading.local()


def bind_logger(game_id: str, env_index: int, info_tensor: torch.Tensor) -> None:
    """Attach the global logging helpers to an environment worker thread."""

    _THREAD_STATE.state = _ThreadRecorderState(game_id=game_id, env_index=env_index, info_tensor=info_tensor)


def log_step(action: int, obs, reward: float, terminated: bool, truncated: bool) -> None:
    """Update counters after every environment step.

    ``obs`` is accepted for API compatibility, but we do not process it in the
    reference implementation.  The helper focuses on maintaining ``info_tensor``
    so that the agent can react to termination signals and observe rewards.
    """

    state: Optional[_ThreadRecorderState] = getattr(_THREAD_STATE, "state", None)
    if state is None:
        return

    state.episode_reward += float(reward)
    state.frame_count += 1

    # ``info_tensor`` can live on either the CPU or GPU.  Assigning to
    # individual elements is safe from CPU threads and keeps everything
    # device-resident.
    row = state.tensor_row()
    row[0] = state.episode_reward
    row[1] = float(state.frame_count)
    row[2] = 1.0 if terminated else 0.0
    row[3] = 1.0 if truncated else 0.0

    if terminated or truncated:
        state.episodes_completed += 1
        now = time.time()
        if now - state.last_print_ts > 15.0:
            print(
                f"[{state.game_id} | env {state.env_index}] episodes={state.episodes_completed} "
                f"last_reward={state.episode_reward:.1f} last_length={state.frame_count}"
            )
            state.last_print_ts = now
        state.episode_reward = 0.0
        state.frame_count = 0


def log_close() -> None:
    """Clean up thread-local state when an environment is shutting down."""

    if hasattr(_THREAD_STATE, "state"):
        delattr(_THREAD_STATE, "state")


def bg_record_proc(obs_tensor, info_tensor, shutdown_event, game_list, start_time):
    """Background process that periodically prints aggregate statistics."""

    print("[bg_record] background recorder online – collecting metrics")
    num_envs = info_tensor.shape[0]
    last_report = 0.0

    # Store the best (highest) cumulative reward seen for each environment.
    best_rewards: Dict[int, float] = {i: float("-inf") for i in range(num_envs)}

    while not shutdown_event.is_set():
        time.sleep(10.0)
        snapshot = info_tensor.clone().detach().to("cpu")

        cumulative_rewards = snapshot[:, 0]
        frame_counts = snapshot[:, 1]
        terminated_flags = snapshot[:, 2]
        truncated_flags = snapshot[:, 3]

        # Track best reward per environment.
        for idx, reward in enumerate(cumulative_rewards.tolist()):
            if reward > best_rewards[idx]:
                best_rewards[idx] = reward

        elapsed = time.time() - start_time
        if elapsed - last_report < 10.0:
            continue

        last_report = elapsed
        running_envs = int(((terminated_flags + truncated_flags) == 0).sum().item())
        avg_reward = float(cumulative_rewards.mean().item()) if num_envs else 0.0
        avg_length = float(frame_counts.mean().item()) if num_envs else 0.0
        print(
            f"[bg_record] t={elapsed:7.1f}s | envs={num_envs} | running={running_envs} "
            f"| avg_reward={avg_reward:7.2f} | avg_length={avg_length:7.1f}"
        )

    print("[bg_record] shutdown requested – final metrics:")
    for idx, best in best_rewards.items():
        game_name = game_list[idx] if idx < len(game_list) else f"env_{idx}"
        if best == float("-inf"):
            continue
        print(f"  • {game_name}: best_cumulative_reward={best:.1f}")

