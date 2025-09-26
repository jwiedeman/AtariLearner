# AtariLearner on Linux

This guide walks through setting up and running the AtariLearner orchestration script on a modern Linux distribution (Ubuntu 22.04+ tested).  It focuses on reproducible installation steps and high-throughput single-game runs that saturate a local GPU.

## Prerequisites

### Hardware

* **GPU:** NVIDIA RTX 3080/4080/4090-class GPU with at least 12 GB of memory.  Training the full-resolution RGB pipeline for one title comfortably fits in 8–10 GB, leaving headroom for replay buffers and optimiser state.
* **CPU:** 8+ cores recommended so that environment workers and the learner can run concurrently without contention.
* **Memory:** 16 GB RAM minimum (32 GB+ suggested if you increase `--instances-per-game`).
* **Storage:** Only a few GB are needed for checkpoints and ROMs.

### Software

* Ubuntu 22.04 LTS (or similar).  Ensure your kernel and NVIDIA driver are up to date.
* Python 3.10 or 3.11.
* CUDA drivers/toolkit matching your GPU (12.1 or 11.8 work with the latest PyTorch wheels).
* System packages: `build-essential`, `python3-venv`, and `ffmpeg` (optional for video capture).

Install the base packages:

```bash
sudo apt update
sudo apt install -y build-essential python3 python3-venv python3-dev ffmpeg
```

Install the CUDA-enabled PyTorch wheel and Python dependencies inside a virtual environment:

```bash
cd /path/to/AtariLearner
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip wheel setuptools
# Replace cu121 with the correct CUDA tag for your driver.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install "gymnasium[atari]" autorom[accept-rom-license] numpy av pillow

# Download the ROM bundle once (AutoROM accepts the licence on your behalf).
AutoROM --accept-license
```

Verify that Gymnasium can locate the ROM set:

```bash
python -c "import gymnasium as gym; gym.make('ALE/Pong-v5')"
```

## Environment variables

Set the two environment variables that control seeding and runtime:

```bash
export myseed=42
export RUNDURATIONSECONDS=$((30*60))  # 30-minute training session
```

## Single-game "optimal" run recipes

The goal of single-game optimisation is to maximise sample throughput for one title while keeping GPU utilisation high.  The commands below assume Pong (`ALE/Pong-v5`), but you can substitute any Gymnasium Atari ID.

### 1. High-throughput GPU run (default viewer disabled)

```bash
python atari_learner.py \
  --game ALE/Pong-v5 \
  --device cuda \
  --num-procs 8 \
  --instances-per-game 8 \
  --fps 60
```

* Uses CUDA tensors for learner + workers.
* Spawns eight worker processes (enough to feed a 4090-class GPU).
* Replicates Pong eight times to generate diverse experience.

### 2. GPU run with live viewer

```bash
python atari_learner.py \
  --game ALE/Pong-v5 \
  --device cuda \
  --num-procs 6 \
  --instances-per-game 6 \
  --show-viewer \
  --viewer-scale 2
```

* Slightly fewer workers to offset viewer overhead.
* Opens a Tkinter window showing the first environment.

### 3. CPU-only fallback (headless server)

```bash
python atari_learner.py \
  --game ALE/Pong-v5 \
  --device cpu \
  --num-procs 4 \
  --instances-per-game 4 \
  --fps 30
```

* Targets systems without a discrete GPU (or during GPU maintenance windows).
* Lower frame rate keeps CPU load manageable.

### 4. Resume from checkpoint

If `checkpoints/latest.pt` exists, resume training and keep only the last five snapshots:

```bash
python atari_learner.py \
  --game ALE/Pong-v5 \
  --device cuda \
  --num-procs 8 \
  --instances-per-game 8 \
  --checkpoint-dir checkpoints \
  --max-checkpoint-snapshots 5
```

### 5. Fresh run ignoring stale checkpoints

```bash
python atari_learner.py \
  --game ALE/Pong-v5 \
  --device cuda \
  --num-procs 8 \
  --instances-per-game 8 \
  --fresh-agent
```

## Monitoring tips

* Use `nvidia-smi dmon` to confirm the GPU is saturated (target 90%+ utilisation).
* Sample CPU usage with `htop`—if workers are starved, reduce `--num-procs` or `--instances-per-game`.
* Check `checkpoints/` for `latest.pt` snapshots every few minutes; adjust `--checkpoint-interval` if needed.
* When running with `--show-viewer`, keep the Tkinter window on the same X server or Wayland session—X forwarding adds latency.

Happy training!
