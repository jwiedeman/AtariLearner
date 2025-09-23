# AtariLearner

This repository contains the multi-process runner used in the viral demo by **@actualhog** ("one NN learns every Atari game at once in realtime from scratch in one hour on a 4090").  The goal of this document is to outline everything that is required to boot the runner locally so that you can plug in your own agent implementation and try to reproduce the experiment.

> **TL;DR** – this repo only ships the orchestration script.  You must provide your own `myagent.py` and `bg_record.py` modules (described below) and an Atari-capable build of Gymnasium/ALE.  Once the prerequisites are met you can launch the full suite of 57 Atari environments with:
>
> ```bash
> export myseed=1
> export RUNDURATIONSECONDS=$((60*60))    # run for one hour
> python atari_learner.py
> ```

---

## 1. Repository Layout

```
.
├── atari_learner.py   # Main orchestrator (spawns env workers + single learner process)
└── README.md           # This file
```

`atari_learner.py` is responsible for:

* Spawning one learner process (`agent_proc`) that continuously calls `Agent.act_and_learn(...)`.
* Launching 16 environment worker processes (`env_proc`) that each host several Gymnasium Atari environments driven by background threads.
* Maintaining shared CUDA tensors that hold RGB observations, selected actions, and auxiliary episode statistics for every game in the list of 57 Atari titles shipped in Gymnasium.
* Periodically checkpointing the agent by calling its `save`/`load` methods.
* (Optionally) Streaming video/metrics by delegating to a background recorder process (see `bg_record.py`).

The actual learning algorithm is **not** in this repository—you are expected to provide your own implementation of the `Agent` class.

---

## 2. Hardware Expectations

The original tweet referenced training on a single NVIDIA RTX 4090 at 60 FPS per environment.  To keep up with the aggregate frame throughput you will need:

* A CUDA-capable GPU with at least 24 GB of memory (the default tensor allocation reserves ~3.5 GB for observations alone, leaving room for the model and optimizer state).
* A multi-core CPU (the runner launches 16 processes × multiple threads each).
* Fast storage only matters if you record video—checkpoints are small.

You can scale `NUM_PROCS` or trim the game list in `atari_learner.py` if you are experimenting on smaller hardware.

---

## 3. Software Requirements

| Component | Recommended Version | Notes |
|-----------|--------------------|-------|
| Operating system | Linux (Ubuntu 22.04 tested) | Uses `forkserver` multiprocessing start method. |
| Python | 3.10+ | Matches current Gymnasium / PyTorch releases. |
| CUDA toolkit & drivers | 12.x (or 11.8+) | Needed for the PyTorch GPU build. |
| PyTorch | Latest stable with CUDA support | Install from [pytorch.org](https://pytorch.org/get-started/locally/). |
| Gymnasium | `gymnasium[atari,accept-rom-license]` ≥ 0.29 | Provides the ALE Atari environments with RGB observations. |
| ALE-Py | Installed automatically via Gymnasium extra | Required backend for Atari ROMs. |
| NumPy | Latest | Array manipulation. |
| FFmpeg | Optional but recommended | Needed by the sample `bg_record.py` implementation for video capture. |

Other utilities (`bg_record.py`, logging stack, etc.) depend on how you decide to implement them.  If you use the same recorder module as in the original gist you will also need `torchvision`, `Pillow`, and `av`.

Create an isolated virtual environment and install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
# Install the CUDA build of PyTorch – replace cu121 with the wheel that matches your driver.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install Gymnasium with Atari extras (downloads the ROMs on first run; accept the license prompt).
pip install "gymnasium[atari,accept-rom-license]"

pip install numpy
# Optional extras for video logging
pip install av pillow
```

On Debian/Ubuntu you can obtain FFmpeg via `sudo apt-get install ffmpeg`.

---

## 4. Required Local Modules

Two helper modules are imported by `atari_learner.py`.  Provide implementations for both in the repository root (alongside `atari_learner.py`).

### 4.1 `myagent.py`

Define an `Agent` class with the following API:

```python
class Agent:
    def __init__(self):
        ...  # build networks, buffers, etc.

    def act_and_learn(self, obs_tensor: torch.Tensor, info_tensor: torch.Tensor, action_tensor: torch.Tensor) -> None:
        """Read the latest RGB observations + episode stats, write actions back to `action_tensor`, and update learner state."""

    def save(self, path: str) -> None:
        """Checkpoint model weights and optimizer state to disk."""

    def load(self, path: str) -> None:
        """Restore a previously saved checkpoint."""
```

The runner keeps `obs_tensor` and `action_tensor` on the GPU, so your agent should also operate on the GPU to avoid device transfers.  The helper tensor `info_tensor` stores per-environment statistics in the format `(cumulative_reward, frame_count, terminated_flag, truncated_flag)`.

### 4.2 `bg_record.py`

This module is optional but highly recommended for debugging.  It is expected to expose:

```python
def bind_logger(game_id: str, env_index: int, info_tensor: torch.Tensor) -> None: ...
def log_step(action: int, obs: np.ndarray, reward: float, terminated: bool, truncated: bool) -> None: ...
def log_close() -> None: ...

def bg_record_proc(obs_tensor, info_tensor, shutdown_event, game_list, start_time):
    """Background process that consumes shared tensors and produces a video/metrics file."""
```

You can stub these functions out (no-ops) if you only want the core training loop.

---

## 5. Environment Variables

Before launching the runner set the following environment variables:

* `myseed` – integer seed used to seed Python, NumPy, PyTorch, and the Atari environments.  Each environment thread derives a unique seed by adding its index.
* `RUNDURATIONSECONDS` – wall-clock runtime for the entire experiment.  The original demo used one hour (`3600`).  The main loop exits when this duration is reached (or on `Ctrl+C`).

Example:

```bash
export myseed=20240925
export RUNDURATIONSECONDS=$((45*60))  # 45 minute session
python atari_learner.py
```

---

## 6. Running the Suite

1. Ensure that your CUDA device is visible (`CUDA_VISIBLE_DEVICES` if needed).
2. Start the Python virtual environment and set the environment variables above.
3. Optionally edit `NUM_PROCS`, `games`, `FPS`, or `MAX_EPISODE_STEPS` in `atari_learner.py` to match your hardware constraints.
4. Launch the runner:

   ```bash
   python atari_learner.py
   ```

5. The script prints the number of environments, seeds per game, and periodically checks that all child processes are still alive.  When the duration elapses (or you interrupt execution) the shutdown flag is broadcast so every process exits cleanly.

Checkpoint files (`agent.pt` by default) are written to the current working directory every ~29 minutes.  You can change the path or cadence inside `agent_proc`.

---

## 7. Troubleshooting & Tips

* **First-run ROM download:** The `gymnasium[atari]` extra will prompt you to accept the ROM license the first time you create an Atari environment.  If you are running headless, pre-download the ROMs by running `python -c "import gymnasium as gym; gym.make('ALE/Pong-v5')"` once interactively.
* **CUDA out of memory:** Reduce `NUM_PROCS`, lower the resolution of the shared observation tensor, or decrease the number of simultaneous games (`games` list) while prototyping.
* **Slow agent loop:** The runner does not throttle the learner.  If your `Agent.act_and_learn` is slow the environments will effectively step at a lower frame-rate.  Profile with PyTorch's `torch.autograd.profiler` or wrap the call in timers.
* **Stability on Mac/Windows:** The script assumes a Unix-like environment and uses `forkserver`.  Windows users should run under WSL2 with CUDA passthrough.
* **Video capture disabled:** If you do not need the video/metric recorder simply provide a dummy `bg_record.py` that defines the expected functions but does nothing.

---

## 8. Credits

* Original concept and demo by [@actualhog](https://twitter.com/actualhog).
* Inspired by John Carmack's discussion on real-time learning.

Happy experimenting!
