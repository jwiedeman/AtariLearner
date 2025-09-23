"""Reference implementation of the :class:`Agent` API expected by ``atari_learner``.

This module provides a very small, self-contained agent that keeps track of a
handful of statistics and issues random actions.  The goal is not to provide a
competitive learner, but rather to ship a ready-to-run scaffold that exercises
the multiprocessing/data sharing paths in ``atari_learner.py``.  Advanced users
are encouraged to replace this file with their own agent implementation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class AgentState:
    """Container for the lightweight state that we persist to disk."""

    total_steps: int = 0
    episodes_seen: int = 0


class Agent:
    """Minimal agent used for bootstrapping the Atari learner runner.

    The runner shares three CUDA tensors with this class:

    ``obs_tensor``
        ``(num_envs, H, W, C)`` tensor containing the latest RGB observation for
        every tracked environment.

    ``info_tensor``
        ``(num_envs, 4)`` tensor with per-environment statistics formatted as
        ``(cumulative_reward, frame_count, terminated_flag, truncated_flag)``.

    ``action_tensor``
        ``(num_envs,)`` tensor where the agent must write the next action to
        execute.  All environments are created with ``full_action_space=True`` so
        the valid action range is ``[0, 18)``.

    The reference implementation keeps the tensors on the GPU (matching the
    expectations of ``atari_learner.py``) and simply issues random actions.  The
    bookkeeping makes it easy to swap in a proper learner at a later point.
    """

    #: Maximum number of actions in the Atari full action space.
    MAX_ACTIONS: int = 18

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state = AgentState()
        # Torch random generator dedicated to action sampling.  Using a manual
        # generator keeps the behaviour deterministic with respect to seeding in
        # ``atari_learner.py``.
        self._rng = torch.Generator(device=self.device)

        # ``atari_learner`` seeds the global PyTorch RNG, therefore copying the
        # current seed keeps everything reproducible.
        self._rng.manual_seed(torch.initial_seed())

        # Buffers that will be lazily initialised on the correct device on first
        # use.  They are reused on every call to avoid repeated allocations.
        self._action_buffer: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Public API expected by ``atari_learner``
    # ------------------------------------------------------------------
    def act_and_learn(
        self, obs_tensor: torch.Tensor, info_tensor: torch.Tensor, action_tensor: torch.Tensor
    ) -> None:
        """Read the latest environment data and emit actions.

        The default implementation samples uniformly from the full action space
        and keeps track of a few diagnostic counters.  The tensors already live
        on the GPU so all operations happen in-place without device transfers.
        """

        num_envs = obs_tensor.shape[0]

        if self._action_buffer is None or self._action_buffer.shape[0] != num_envs:
            self._action_buffer = torch.empty(num_envs, device=self.device, dtype=torch.float32)

        # Sample a random action for every environment.  ``torch.randint`` is not
        # generator-aware prior to PyTorch 2.3, therefore we sample uniform random
        # numbers and quantise them to the [0, MAX_ACTIONS) range.
        self._action_buffer.uniform_(0.0, float(self.MAX_ACTIONS), generator=self._rng)
        actions = self._action_buffer.to(dtype=torch.int64)

        # Update basic statistics from ``info_tensor``.  Keeping them on the
        # agent side makes it trivial to extend the class into a real learner.
        with torch.no_grad():
            info_view = info_tensor[:, :2]
            if info_view.numel() > 0:
                frame_counts = info_view[:, 1]
                self.state.total_steps += int(frame_counts.sum().item())

        action_tensor.copy_(actions)

    def save(self, path: str) -> None:
        """Persist lightweight agent state to ``path``.

        Only a few counters are stored; the expectation is that real agents will
        extend this to include neural network weights, optimiser states, etc.
        """

        directory = os.path.dirname(os.path.abspath(path))
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        torch.save(self.state.__dict__, path)

    def load(self, path: str) -> None:
        """Load previously persisted state from ``path`` if it exists."""

        if not os.path.exists(path):
            raise FileNotFoundError(path)

        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected checkpoint format in {path!r}")

        self.state = AgentState(**payload)

