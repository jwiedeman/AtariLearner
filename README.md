# AtariLearner

This repository contains the multi-process runner used in the viral demo by **@actualhog** ("one NN learns every Atari game at once in realtime from scratch in one hour on a 4090").  The goal of this document is to outline everything that is required to boot the runner locally so that you can plug in your own agent implementation and try to reproduce the experiment.

> **TL;DR** – this repo ships the orchestration script **and** lightweight reference implementations of `myagent.py` and `bg_record.py`.  Feel free to replace them with your own learner/recorder, but the included versions make it possible to boot the runner immediately once you install an Atari-capable build of Gymnasium/ALE.  Launch the full suite of 57 Atari environments with:
>
> ```bash
> export myseed=1
> export RUNDURATIONSECONDS=$((60*60))    # run for one hour
> python atari_learner.py
> ```
>
> On more modest hardware (for example an Apple Silicon laptop) you can now run a single environment on the CPU by selecting it on the command line:
>
> ```bash
> export myseed=1
> export RUNDURATIONSECONDS=$((5*60))     # run for five minutes
> python atari_learner.py --game ALE/Pong-v5 --device cpu
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
* Launching a configurable number of environment worker processes (`env_proc`) that each host several Gymnasium Atari environments driven by background threads.
* Maintaining shared tensors (CPU or CUDA, depending on how you launch the runner) that hold RGB observations, selected actions, and auxiliary episode statistics for every game in the list of 57 Atari titles shipped in Gymnasium.
* Periodically checkpointing the agent by calling its `save`/`load` methods.
* (Optionally) Streaming video/metrics by delegating to a background recorder process (see `bg_record.py`).

The included reference agent is a compact deep Q-learner that trains directly
from the shared tensors.  It batches Atari RGB observations through a
convolutional encoder, maintains per-game output heads so that multiple
environments can specialise, streams data into a replay buffer, and periodically
writes checkpoints that include optimiser/replay state for resuming runs.  You
can still replace `myagent.py` with your own implementation, but the shipped
version is now a practical starting point for experimentation rather than a
random-action scaffold.

---

## 2. Hardware Expectations

The original tweet referenced training on a single NVIDIA RTX 4090 at 60 FPS per environment.  To keep up with the aggregate frame throughput you will need:

* A CUDA-capable GPU with at least 24 GB of memory (the default tensor allocation reserves ~3.5 GB for observations alone, leaving room for the model and optimizer state).
* A multi-core CPU (the runner launches 16 processes × multiple threads each).
* Fast storage only matters if you record video—checkpoints are small.

You can scale `NUM_PROCS` or trim the game list in `atari_learner.py` if you are experimenting on smaller hardware.

As of this update the orchestrator can also run on CPU-only systems.  Use `--game` to target a single title (or provide a shorter list via `--games`) and pass `--device cpu` to allocate the shared tensors on host memory.  The runner automatically switches to the safer `spawn` multiprocessing start method on macOS.

---

## 3. Software Requirements

| Component | Recommended Version | Notes |
|-----------|--------------------|-------|
| Operating system | Linux (Ubuntu 22.04 tested), macOS 14+ | Uses `forkserver` on Linux and automatically falls back to `spawn` on macOS. |
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

### macOS (Apple Silicon or Intel) setup

The commands above also work on macOS, but a couple of mac-specific notes can
save you some friction:

1. Use the system `python3` binary (or a Homebrew install) when creating the
   virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   ```

2. Install the universal (CPU + Metal/MPS) build of PyTorch that ships on PyPI.
   It runs on Apple Silicon out of the box and also supports Intel Macs:

   ```bash
   pip install torch torchvision
   ```

   If you prefer the Intel-only wheel, replace the previous line with
   `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.

3. Install the Atari extras and supporting libraries exactly as shown earlier:

   ```bash
   pip install "gymnasium[atari,accept-rom-license]"
   pip install numpy
   pip install av pillow
   ```

4. Optional: install FFmpeg for video capture via Homebrew with
   `brew install ffmpeg`.

When running under the default `zsh` shell, paste each command individually (or
remove the leading `#` comments) to avoid "parse error near `)`" messages.

---

## 4. Required Local Modules

Two helper modules are imported by `atari_learner.py`.  Provide implementations for both in the repository root (alongside `atari_learner.py`).

### 4.1 `myagent.py`

The shipped agent is no longer a random-action scaffold – it is a small but
fully functioning deep Q-network learner that:

* Shares a convolutional encoder across all environments and attaches a
  dedicated output head to each game so that policies can specialise.
* Streams the runner's shared tensors into an experience replay buffer,
  performs TD updates in-place, and periodically syncs a target network.
* Supports resumable training by checkpointing network weights, optimiser
  state, replay samples, and bookkeeping counters.

The class still exposes the same API (`act_and_learn`, `save`, `load`), making
it easy to drop in more advanced implementations.  If you prefer a different
algorithm simply replace `myagent.py` with your own agent while keeping the
method signatures intact.

### 4.2 `bg_record.py`

`atari_learner.py` also imports a background recorder module.  The built-in
version focuses on metrics: environment workers call `bind_logger`,
`log_step`, and `log_close` to update a shared tensor, while the
`bg_record_proc` process periodically prints aggregate statistics.  Power users
can swap in their own recorder that writes gameplay video, streams metrics, or
integrates with experiment tracking services.

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

On macOS laptops you will typically run a single training environment at a
time.  Use a shorter duration and explicitly pin the runner to the CPU:

```bash
export myseed=1
export RUNDURATIONSECONDS=$((5*60))
python atari_learner.py --game ALE/Pong-v5 --device cpu --num-procs 1
```

The `--game` flag selects one Atari title, while `--num-procs 1` keeps the
process count low so the run fits comfortably on a MacBook.

Want to saturate a CPU-only machine with multiple copies of the same title?
Add `--instances-per-game 10` (or any positive integer) to replicate each
selected game.  Combined with `--show-viewer` the Tkinter window tiles every
instance so you can watch them side by side:

```bash
python atari_learner.py --game ALE/Pong-v5 --device cpu --instances-per-game 10 --show-viewer
```

By default the runner spawns up to one worker process per environment (capped
at 16 unless you override `--num-procs`), keeping each instance isolated and
matching the viewer layout.

---

## 6. Running the Suite

1. Start the Python virtual environment and set the `myseed`/`RUNDURATIONSECONDS` variables shown above.
2. Decide which environments to launch:
   * **Full suite:** run `python atari_learner.py` with no extra flags.
   * **Single environment:** add `--game ALE/Pong-v5` (or any other ID from the Gymnasium registry).
   * **Custom list:** provide multiple IDs with `--games ALE/Pong-v5 ALE/Breakout-v5`.
   * **Multiple copies per game:** append `--instances-per-game 10` to run ten instances of each selected title in parallel.
3. Pick a device with `--device {auto,cuda,cpu}`.  The default `auto` uses CUDA when available and falls back to CPU otherwise.
4. (Optional) Tune parallelism with `--num-procs` and the frame rate with `--fps`/`--max-episode-steps`.  These replace the manual edits that were previously required.
5. Launch the runner.  For example, to run Pong and Breakout on the CPU:

   ```bash
   python atari_learner.py --games ALE/Pong-v5 ALE/Breakout-v5 --device cpu
   ```

### Watching the run live

Pass `--show-viewer` to open a lightweight Tkinter window that mirrors one of the
environments in real time.  The viewer defaults to the first environment in the
selected list, but you can target a specific game via `--viewer-game
ALE/Pong-v5` or by passing the numeric index with `--viewer-env-index 3`.

Additional knobs let you rescale the window (`--viewer-scale 3`) or reduce the
refresh rate to save CPU time (`--viewer-fps 15`).  The implementation depends on
[`tkinter`](https://docs.python.org/3/library/tkinter.html) (ships with Python)
and [Pillow](https://python-pillow.org/); install Pillow with `pip install
pillow` if you do not already have it in your environment.

The script prints the number of environments, seeds per game, and periodically checks that all child processes are still alive.  When the duration elapses (or you interrupt execution) the shutdown flag is broadcast so every process exits cleanly.

Checkpoint files are written automatically while the run is in progress.  Use
`--checkpoint-dir` to pick the output folder, `--checkpoint-interval` to control
the cadence (in seconds), and `--max-checkpoint-snapshots` to choose how many
timestamped snapshots are retained alongside the rolling `latest.pt` file.

---

## 7. Useful Command-Line Flags

The runner exposes a handful of switches so you can tailor it to the hardware you have on hand:

| Flag | Description |
|------|-------------|
| `--game ALE/Pong-v5` | Launch exactly one environment. |
| `--games ...` | Provide an explicit list of Gymnasium environment IDs to run simultaneously. |
| `--device {auto,cuda,cpu}` | Choose where the shared tensors live.  `auto` prefers CUDA when available. |
| `--num-procs N` | Number of environment worker processes.  Defaults to `min(16, number of environments)`. |
| `--instances-per-game 4` | Replicate each selected title the specified number of times (defaults to 1). |
| `--fps 30` | Target frame rate per environment. |
| `--max-episode-steps 10000` | Override the maximum episode length passed to Gymnasium. |
| `--start-method spawn` | Force a specific multiprocessing start method (Linux defaults to `forkserver`, macOS to `spawn`). |
| `--show-viewer` | Open a Tkinter window with a live feed from one environment. |
| `--viewer-game ALE/Pong-v5` | Pick the environment shown in the viewer by name. |
| `--viewer-env-index 0` | Choose the environment shown in the viewer by index. |
| `--viewer-scale 2.0` | Scale the viewer window by the provided factor. |
| `--viewer-fps 30` | Target refresh rate for the viewer window (lower values reduce CPU usage). |
| `--checkpoint-dir checkpoints` | Directory used to store `latest.pt` and timestamped snapshots. |
| `--checkpoint-interval 1740` | Seconds between automatic checkpoint saves. |
| `--max-checkpoint-snapshots 5` | Number of rolling timestamped snapshots to retain. |

Run `python atari_learner.py --help` to see the complete list and default values.

## 8. Troubleshooting & Tips

* **First-run ROM download:** The `gymnasium[atari]` extra will prompt you to accept the ROM license the first time you create an Atari environment.  If you are running headless, pre-download the ROMs by running `python -c "import gymnasium as gym; gym.make('ALE/Pong-v5')"` once interactively.
* **CUDA out of memory:** Reduce `--num-procs`, lower the resolution of the shared observation tensor, or decrease the number of simultaneous games by selecting a smaller set with `--games`.
* **Slow agent loop:** The runner does not throttle the learner.  If your `Agent.act_and_learn` is slow the environments will effectively step at a lower frame-rate.  Profile with PyTorch's `torch.autograd.profiler` or wrap the call in timers.
* **Stability on Mac/Windows:** The script assumes a Unix-like environment.  Linux defaults to the `forkserver` start method, while macOS automatically switches to `spawn`.  Windows users should run under WSL2 with CUDA passthrough.
* **Video capture disabled:** If you do not need the video/metric recorder simply provide a dummy `bg_record.py` that defines the expected functions but does nothing.

---

## 9. Credits

* Original concept and demo by [@actualhog](https://twitter.com/actualhog).
* Inspired by John Carmack's discussion on real-time learning.

Happy experimenting!
