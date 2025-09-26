# AtariLearner on macOS

This guide describes how to run AtariLearner on macOS Ventura/Sonoma for both Apple Silicon (M1/M2/M3) and Intel Macs.  Because macOS lacks native CUDA, the runner falls back to CPU or Apple's Metal Performance Shaders (MPS) backend when available.

## Prerequisites

### Hardware

* **Apple Silicon:** M1 Pro/Max, M2 Pro/Max, or M3-class chips perform best thanks to the integrated GPU and unified memory.
* **Intel:** Preferably with a discrete AMD GPU (Metal-enabled) and at least 32 GB RAM.  CPU-only runs work but will be slower.
* **Memory:** 16 GB RAM minimum; unified memory benefits from higher capacity if you increase `--instances-per-game`.

### Software

* macOS 13 (Ventura) or newer.
* Xcode command-line tools (`xcode-select --install`).
* Homebrew (recommended) for installing FFmpeg.
* Python 3.10 or 3.11 from the system, Homebrew, or `pyenv`.

Create and activate a virtual environment:

```bash
cd /path/to/AtariLearner
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
```

Install the universal PyTorch build (includes CPU + Metal support), Gymnasium, and dependencies:

```bash
pip install torch torchvision
pip install "gymnasium[atari,accept-rom-license]" numpy av pillow
```

Optional: install FFmpeg via Homebrew for the sample recorder module:

```bash
brew install ffmpeg
```

Accept the ROM licence prompt once with:

```bash
python -c "import gymnasium as gym; gym.make('ALE/Pong-v5')"
```

## Environment variables

```bash
export myseed=7
export RUNDURATIONSECONDS=$((20*60))  # 20-minute experiment for laptops
```

## Single-game run recipes

macOS defaults to the safe `spawn` multiprocessing start method.  The commands below balance throughput with thermals on Apple Silicon laptops.  Substitute your preferred game ID for Pong.

### 1. Apple Silicon with MPS acceleration

```bash
python atari_learner.py \
  --game ALE/Pong-v5 \
  --device auto \
  --num-procs 4 \
  --instances-per-game 4 \
  --fps 45
```

* `--device auto` selects the Metal backend when available.
* Four workers keep the SoC busy without overwhelming cooling.

### 2. CPU-only run (Intel Mac or thermally constrained laptop)

```bash
python atari_learner.py \
  --game ALE/Pong-v5 \
  --device cpu \
  --num-procs 2 \
  --instances-per-game 2 \
  --fps 30
```

* Keeps the process count low to avoid fan spikes.

### 3. High-visibility training with viewer

```bash
python atari_learner.py \
  --game ALE/Pong-v5 \
  --device auto \
  --num-procs 3 \
  --instances-per-game 3 \
  --show-viewer \
  --viewer-scale 2 \
  --viewer-fps 20
```

* Reduces viewer FPS to minimise UI overhead.

### 4. Resume from checkpoint stored on external drive

```bash
python atari_learner.py \
  --game ALE/Pong-v5 \
  --device auto \
  --num-procs 4 \
  --instances-per-game 4 \
  --checkpoint-dir /Volumes/External/checkpoints \
  --max-checkpoint-snapshots 3
```

### 5. Fresh run bypassing checkpoints

```bash
python atari_learner.py \
  --game ALE/Pong-v5 \
  --device auto \
  --num-procs 4 \
  --instances-per-game 4 \
  --fresh-agent
```

## Tips

* macOS Ventura+ prompts for microphone/camera access when Tkinter windows capture input—deny safely.
* Use `Activity Monitor` or `powermetrics --samplers smc` to monitor GPU usage and thermals.
* If you see "NotImplementedError: MPS backend is not available", reinstall PyTorch (ensure you are on Python 3.10/3.11) and update macOS.
* The first run after waking from sleep can be sluggish; restart the terminal session to reset the MPS context if needed.

Happy training on macOS!
