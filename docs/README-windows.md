# AtariLearner on Windows

This document explains how to run AtariLearner on Windows 11 using either WSL2 (recommended for CUDA workflows) or native Windows Python environments.  WSL2 provides the smoothest experience because the repository targets a Unix-style process model.

## Option A – WSL2 with Ubuntu (recommended)

### Prerequisites

* Windows 11 22H2 or later.
* NVIDIA GPU with drivers that support WSL2 CUDA (version 522+).
* Installed WSL2 with Ubuntu 22.04 or 24.04: `wsl --install -d Ubuntu`.
* Windows Terminal (optional but convenient).

Inside the Ubuntu shell:

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  python3 python3-venv python3-dev python3-pip \
  ffmpeg
```

If `pip` is still missing (older Ubuntu images ship without it), bootstrap it
once:

```bash
python3 -m ensurepip --upgrade
```

Create and activate a Python virtual environment in the cloned repository. The
commands below assume you are using **bash** inside WSL (start it explicitly
with `bash` if your default shell is `/bin/sh`).

```bash
cd /mnt/c/Users/<you>/AtariLearner
python3 -m venv .venv
source .venv/bin/activate   # or: . .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install "gymnasium[atari]" autorom[accept-rom-license] numpy av pillow

# Download the ROM set once so the ALE namespace is registered.
AutoROM --accept-license
```

> **Tip:** If you see `source: not found`, you are in `/bin/sh`. Run `bash`
> first or use `. .venv/bin/activate` to source the activation script with POSIX
> `.`.

Quick sanity check (optional but recommended):

```bash
python -m pip --version
python - <<'PY'
import torch, gymnasium
print('Torch:', torch.__version__)
print('Gymnasium:', gymnasium.__version__)
PY
```

Accept the Atari ROM licence once:

```bash
python -c "import gymnasium as gym; gym.make('ALE/Pong-v5')"
```

Set up environment variables (place them in `~/.bashrc` or export per-session):

```bash
export myseed=101
export RUNDURATIONSECONDS=$((45*60))
```

### Single-game run recipes (WSL2)

1. **Max throughput on a 4090:**

   ```bash
   python atari_learner.py \
     --game ALE/Pong-v5 \
     --device cuda \
     --num-procs 8 \
     --instances-per-game 10 \
     --fps 60
   ```

2. **Viewer-enabled session:**

   ```bash
   python atari_learner.py \
     --game ALE/Pong-v5 \
     --device cuda \
     --num-procs 6 \
     --instances-per-game 6 \
     --show-viewer \
     --viewer-scale 2
   ```

3. **CPU-only fallback (no CUDA pass-through):**

   ```bash
   python atari_learner.py \
     --game ALE/Pong-v5 \
     --device cpu \
     --num-procs 4 \
     --instances-per-game 4 \
     --fps 30
   ```

4. **Resume from checkpoints stored on the Windows filesystem:**

   ```bash
   python atari_learner.py \
     --game ALE/Pong-v5 \
     --device cuda \
     --num-procs 8 \
     --instances-per-game 10 \
     --checkpoint-dir /mnt/c/Users/<you>/AtariLearner/checkpoints \
     --max-checkpoint-snapshots 5
   ```

5. **Fresh start ignoring checkpoints:**

   ```bash
   python atari_learner.py \
     --game ALE/Pong-v5 \
     --device cuda \
     --num-procs 8 \
     --instances-per-game 10 \
     --fresh-agent
   ```

## Option B – Native Windows Python (experimental)

Running the orchestrator directly on Windows is possible but carries caveats because the default multiprocessing start method is `spawn`.  Expect slower start-up times and ensure your agent/background recorder is import-safe.

### Prerequisites

* Install Python 3.10+ from the Microsoft Store or python.org.
* Install the Visual Studio 2022 Build Tools (C++ workload) for compiling dependencies.
* Install the latest NVIDIA drivers.
* Optional: install Chocolatey and use it to fetch FFmpeg (`choco install ffmpeg`).

Open **Windows Terminal → PowerShell** and create a virtual environment:

```powershell
cd C:\path\to\AtariLearner
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel setuptools
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install "gymnasium[atari]" autorom[accept-rom-license] numpy av pillow

# AutoROM downloads the ROMs and records the licence acceptance once.
AutoROM --accept-license
```

Export environment variables for the current PowerShell session:

```powershell
$env:myseed = 2024
$env:RUNDURATIONSECONDS = 3600
```

Verify the install once AutoROM has finished:

```powershell
python -c "import gymnasium as gym; gym.make('ALE/Pong-v5')"
```

### Single-game commands (native Windows)

Use escaped quotes and backticks to wrap long commands if desired.  Below are PowerShell-friendly snippets.

1. **Maximum CUDA throughput:**

   ```powershell
   python atari_learner.py `
     --game ALE/Pong-v5 `
     --device cuda `
     --num-procs 6 `
     --instances-per-game 8 `
     --fps 60
   ```

2. **Viewer-enabled training:**

   ```powershell
   python atari_learner.py `
     --game ALE/Pong-v5 `
     --device cuda `
     --num-procs 4 `
     --instances-per-game 4 `
     --show-viewer `
     --viewer-scale 2
   ```

3. **CPU-only mode (for systems without CUDA):**

   ```powershell
   python atari_learner.py `
     --game ALE/Pong-v5 `
     --device cpu `
     --num-procs 3 `
     --instances-per-game 3 `
     --fps 30
   ```

4. **Resume from checkpoints:**

   ```powershell
   python atari_learner.py `
     --game ALE/Pong-v5 `
     --device cuda `
     --num-procs 6 `
     --instances-per-game 8 `
     --checkpoint-dir checkpoints `
     --max-checkpoint-snapshots 5
   ```

5. **Force fresh agent:**

   ```powershell
   python atari_learner.py `
     --game ALE/Pong-v5 `
     --device cuda `
     --num-procs 6 `
     --instances-per-game 8 `
     --fresh-agent
   ```

## Troubleshooting

* If PowerShell blocks scripts, run `Set-ExecutionPolicy -Scope Process RemoteSigned` before activating the virtual environment.
* When running inside WSL2, ensure the project folder resides on the Linux filesystem (e.g. `~/AtariLearner`) for best disk performance.
* `ModuleNotFoundError: tkinter`: install `sudo apt install python3-tk` inside WSL2 or add the Windows optional feature "Tkinter" when using the Windows Store Python distribution.
* If FFmpeg is missing, either install it via `apt` (WSL2) or Chocolatey (Windows), or provide a no-op `bg_record.py`.

Happy training on Windows!
