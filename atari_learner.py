#!/usr/bin/env python3
"""Entry point for launching the multi-environment Atari runner.

The original viral demo by ``@actualhog`` launched the full suite of 57 Atari
titles simultaneously on a powerful GPU.  This script keeps that ability while
also exposing a couple of quality-of-life flags so that the runner can be used
on more modest hardware (for example, running a single environment on a CPU
only Mac).
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import threading
import time
from typing import List, Optional, Sequence

import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp
from torch import Tensor

from bg_record import bind_logger, log_close, log_step

# torch.set_num_threads(1)

DEFAULT_NUM_PROCS = 16
DEFAULT_FPS = 60.0
MAX_ACTIONS = 18

ALL_GAMES: Sequence[str] = sorted([
    "ALE/Adventure-v5", "ALE/AirRaid-v5", "ALE/Alien-v5", "ALE/Amidar-v5", "ALE/Assault-v5",
    "ALE/Asterix-v5", "ALE/Asteroids-v5", "ALE/Atlantis-v5", "ALE/BankHeist-v5",
    "ALE/BattleZone-v5", "ALE/BeamRider-v5", "ALE/Berzerk-v5", "ALE/Bowling-v5",
    "ALE/Boxing-v5", "ALE/Breakout-v5", "ALE/Carnival-v5", "ALE/Centipede-v5",
    "ALE/ChopperCommand-v5", "ALE/CrazyClimber-v5", "ALE/Defender-v5", "ALE/DemonAttack-v5",
    "ALE/DoubleDunk-v5", "ALE/ElevatorAction-v5", "ALE/Enduro-v5", "ALE/FishingDerby-v5",
    "ALE/Freeway-v5", "ALE/Frostbite-v5", "ALE/Gopher-v5", "ALE/Gravitar-v5", "ALE/Hero-v5",
    "ALE/IceHockey-v5", "ALE/Jamesbond-v5", "ALE/JourneyEscape-v5", "ALE/Kangaroo-v5",
    "ALE/KeystoneKapers-v5", "ALE/KingKong-v5", "ALE/Krull-v5", "ALE/KungFuMaster-v5",
    "ALE/MontezumaRevenge-v5", "ALE/MsPacman-v5", "ALE/NameThisGame-v5", "ALE/Phoenix-v5",
    "ALE/Pitfall-v5", "ALE/Pong-v5", "ALE/Pooyan-v5", "ALE/PrivateEye-v5", "ALE/Qbert-v5",
    "ALE/Riverraid-v5", "ALE/RoadRunner-v5", "ALE/Robotank-v5", "ALE/Seaquest-v5",
    "ALE/Skiing-v5", "ALE/Solaris-v5", "ALE/SpaceInvaders-v5", "ALE/StarGunner-v5",
    "ALE/Tennis-v5", "ALE/TimePilot-v5", "ALE/Tutankham-v5", "ALE/UpNDown-v5",
    "ALE/Venture-v5", "ALE/VideoPinball-v5", "ALE/WizardOfWor-v5", "ALE/YarsRevenge-v5",
    "ALE/Zaxxon-v5"
])


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command line arguments used to customise the runner."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--game",
        help=(
            "Run a single Gymnasium environment instead of the full 57-game suite. "
            "Mutually exclusive with --games."
        ),
    )
    parser.add_argument(
        "--games",
        nargs="+",
        metavar="ENV_ID",
        help="Explicit list of Gymnasium environment IDs to launch.",
    )
    parser.add_argument(
        "--num-procs",
        type=int,
        default=None,
        help=(
            "Number of environment worker processes to spawn.  Defaults to min(16, number of environments)."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help=(
            "Device used for the shared tensors.  'auto' selects CUDA when available "
            "and falls back to CPU otherwise."
        ),
    )
    parser.add_argument(
        "--start-method",
        choices=sorted(mp.get_all_start_methods()),
        default=None,
        help=(
            "Multiprocessing start method.  Defaults to 'forkserver' on Linux and 'spawn' on macOS."
        ),
    )
    parser.add_argument(
        "--instances-per-game",
        type=int,
        default=1,
        help=(
            "Number of parallel instances to launch for each selected game.  "
            "Useful on CPU-only hardware to run multiple copies of the same title."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help="Target frames per second per environment (default: 60).",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help=(
            "Maximum steps per episode passed to Gymnasium.  Defaults to 45 minutes worth of frames "
            "at the selected FPS."
        ),
    )
    parser.add_argument(
        "--show-viewer",
        action="store_true",
        help=(
            "Launch a lightweight Tkinter window that shows live feeds from the selected environments. "
            "Requires Pillow (pip install pillow)."
        ),
    )
    parser.add_argument(
        "--viewer-env-index",
        type=int,
        default=None,
        help=(
            "Index of the environment to display when --show-viewer is set.  "
            "By default the viewer tiles every available environment."
        ),
    )
    parser.add_argument(
        "--viewer-game",
        type=str,
        default=None,
        help=(
            "Name of the environment to display when --show-viewer is set.  "
            "Overrides --viewer-env-index when provided."
        ),
    )
    parser.add_argument(
        "--viewer-scale",
        type=float,
        default=2.0,
        help="Scaling factor applied to the viewer window when --show-viewer is set (default: 2x).",
    )
    parser.add_argument(
        "--viewer-fps",
        type=float,
        default=30.0,
        help=(
            "Approximate refresh rate of the viewer window when --show-viewer is set.  "
            "Lower values reduce CPU usage (default: 30)."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory where agent checkpoints are stored (default: checkpoints).",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=29 * 60,
        help="Seconds between automatic agent checkpoints (default: 1740).",
    )
    parser.add_argument(
        "--max-checkpoint-snapshots",
        type=int,
        default=5,
        help=(
            "Maximum number of rolling checkpoint snapshots retained alongside the latest file "
            "(default: 5)."
        ),
    )
    parser.add_argument(
        "--fresh-agent",
        action="store_true",
        help=(
            "Start training from a freshly initialised agent instead of loading an existing checkpoint. "
            "Useful when you explicitly want to train a new model."
        ),
    )
    args = parser.parse_args(argv)

    if args.game and args.games:
        parser.error("--game and --games are mutually exclusive")

    if args.instances_per_game < 1:
        parser.error("--instances-per-game must be a positive integer")

    if args.checkpoint_interval <= 0:
        parser.error("--checkpoint-interval must be a positive integer")

    if args.max_checkpoint_snapshots < 0:
        parser.error("--max-checkpoint-snapshots must be >= 0")

    return args


def resolve_games(args: argparse.Namespace) -> tuple[List[str], List[str]]:
    """Return the ordered list of environments and display names to launch."""

    if args.game:
        base_games = [args.game]
    elif args.games:
        base_games = list(dict.fromkeys(args.games))  # Preserve order while removing duplicates.
    else:
        base_games = list(ALL_GAMES)

    instances = max(1, int(args.instances_per_game))
    expanded: List[str] = []
    display_names: List[str] = []
    use_suffixes = instances > 1 or len(base_games) != len(set(base_games))
    instance_counters: dict[str, int] = {}

    for game in base_games:
        for _ in range(instances):
            expanded.append(game)
            count = instance_counters.get(game, 0) + 1
            instance_counters[game] = count
            if use_suffixes:
                display_names.append(f"{game} #{count}")
            else:
                display_names.append(game)

    return expanded, display_names


def resolve_device(device_flag: str) -> torch.device:
    """Map a device flag to an actual :class:`torch.device`."""

    if device_flag == "auto":
        device_flag = "cuda" if torch.cuda.is_available() else "cpu"
    if device_flag == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_flag)


def select_start_method(start_method_flag: str | None) -> str:
    """Pick an appropriate multiprocessing start method."""

    if start_method_flag:
        return start_method_flag
    if sys.platform == "darwin":
        # Fork-based start methods are not recommended on macOS when interacting with
        # Cocoa or Metal (common on Apple Silicon laptops).
        return "spawn"
    return "forkserver"


def env_thread_worker(
    first_start_at: float,
    game_id: str,
    display_name: str,
    g_idx: int,
    obs_s: Tensor,
    act_s: Tensor,
    info_s: Tensor,
    shutdown,
    fps: float,
    max_episode_steps: int,
):
    import ale_py # required for atari
    next_frame_due = first_start_at + 15.0 # let all procs start
    env = gym.make(
        game_id,
        obs_type="rgb",
        frameskip=1,
        repeat_action_probability=0.0,
        full_action_space=True,
        max_episode_steps=max_episode_steps,
    )
    envseed = g_idx * 100 + int(os.environ['myseed'])
    print(f"env_index={g_idx} game_id={game_id} display_name={display_name!r} envseed={envseed}")
    obs, _ = env.reset(seed=envseed)
    h, w, _ = obs.shape
    obs_s[g_idx, :h, :w].copy_(torch.from_numpy(obs), non_blocking=True)
    bind_logger(display_name, g_idx, info_s)
    while not shutdown.is_set():
        while time.time() > next_frame_due:
            next_frame_due += 1.0 / fps
        time.sleep(max(0, next_frame_due - time.time()))
        action = act_s[g_idx].item()
        obs, rew, term, trunc, _ = env.step(action)
        log_step(action, obs, rew, term, trunc)
        obs_s[g_idx, :h, :w].copy_(torch.from_numpy(obs), non_blocking=True)
        if term or trunc:
            obs, _ = env.reset()
            obs_s[g_idx, :h, :w].copy_(torch.from_numpy(obs), non_blocking=True)
    log_close()
def seed(prefix, offset:int):
    s = int(os.environ['myseed']) + offset
    print(f'random seed: {prefix}: {s=}')
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(s)
    # torch.backends.cudnn.deterministic = True # Not worth it!
    # torch.backends.cudnn.benchmark = False # Not worth it!


def env_proc(first_start_at, game_chunk, offset, obs_s, act_s, info_s, shutdown, fps, max_episode_steps):
    seed('env', offset + 1)
    threads = [
        threading.Thread(
            target=env_thread_worker,
            args=(
                first_start_at,
                game_id,
                display_name,
                offset + i,
                obs_s,
                act_s,
                info_s,
                shutdown,
                fps,
                max_episode_steps,
            ),
        )
        for i, (game_id, display_name) in enumerate(game_chunk)
    ]
    for t in threads: t.start()
    for t in threads: t.join()

def agent_proc(
    obs_s,
    act_s,
    info_s,
    shutdown,
    game_ids,
    checkpoint_dir,
    checkpoint_interval,
    max_snapshots,
    fresh_agent,
):
    seed('agent', 0)
    from myagent import Agent

    checkpoint_dir = os.path.abspath(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "latest.pt")

    agent = Agent(game_ids=game_ids, max_snapshots=max_snapshots)
    agent.configure_envs(game_ids)

    if not fresh_agent and os.path.exists(checkpoint_path):
        try:
            print(f"loading existing checkpoint from {checkpoint_path}")
            agent.load(checkpoint_path)
            agent.configure_envs(game_ids)
            print("checkpoint load successful")
        except Exception as exc:
            print(f"failed to load checkpoint {checkpoint_path}: {exc}")
    elif fresh_agent and os.path.exists(checkpoint_path):
        print(
            "--fresh-agent specified: existing checkpoint will be ignored and a new model will be initialised"
        )

    try:
        print(f"saving initial checkpoint to {checkpoint_path}")
        agent.save(checkpoint_path)
    except Exception as exc:
        print(f"unable to write initial checkpoint {checkpoint_path}: {exc}")

    last_save_time = time.time()
    while not shutdown.is_set():
        # NOTE: THE AGENT IS CALLED IN A LOOP AS FAST AS POSSIBLE. THERE IS NO SLEEP STATEMENT IN THIS BLOCK.
        # (a very fast agent would do multiple passes per frame. a slow agent would take multiple frames to do a pass.)
        # NOTE: THE act_and_learn ARGUMENTS HAVE CHANGED
        # EACH ROW IN info_s IS LIKE (acc_reward, acc_frames, acc_term, acc_trunc)
        agent.act_and_learn(obs_s, info_s.clone(), act_s)

        if time.time() - last_save_time >= checkpoint_interval:
            try:
                print(f"saving checkpoint to {checkpoint_path}")
                agent.save(checkpoint_path)
                last_save_time = time.time()
            except Exception as exc:
                print(f"failed to save checkpoint {checkpoint_path}: {exc}")


def _trim_black_borders(frame: np.ndarray) -> np.ndarray:
    """Return the slice of ``frame`` that excludes surrounding zero rows/columns."""

    if frame.ndim != 3:
        return frame
    mask = frame.any(axis=2)
    row_mask = mask.any(axis=1)
    col_mask = mask.any(axis=0)
    rows = np.where(row_mask)[0]
    cols = np.where(col_mask)[0]
    if rows.size and cols.size:
        return frame[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    return frame


def viewer_proc(
    obs_s: Tensor,
    shutdown,
    env_indices: Sequence[int],
    game_list: Sequence[str],
    scale: float,
    target_fps: float,
):
    """Display live feeds from ``obs_s`` for one or more environments."""

    try:
        import tkinter as tk
    except Exception as exc:  # pragma: no cover - platform specific
        print(f"[viewer] Tkinter unavailable – viewer disabled ({exc})")
        return

    try:
        from PIL import Image, ImageTk
    except ImportError:  # pragma: no cover - optional dependency
        print("[viewer] Pillow not installed – run 'pip install pillow' to enable the live viewer")
        return

    env_indices = list(env_indices)
    if not env_indices:
        print("[viewer] no environments selected – viewer disabled")
        return

    env_names = [
        game_list[i] if i < len(game_list) else f"env_{i}"
        for i in env_indices
    ]

    root = tk.Tk()
    if len(env_indices) == 1:
        title_env = env_names[0]
        title_suffix = f" – {title_env} (env {env_indices[0]})"
    else:
        title_suffix = f" – {len(env_indices)} environments"
    root.title(f"AtariLearner Viewer{title_suffix}")
    root.resizable(True, True)

    if len(env_indices) == 1:
        print(f"[viewer] launching viewer for {env_names[0]} (env {env_indices[0]})")
    else:
        print(f"[viewer] launching viewer for {len(env_indices)} environments")

    num_tiles = len(env_indices)
    tiles = []
    tile_padding = 4

    for env_index, env_name in zip(env_indices, env_names):
        frame = tk.Frame(root, borderwidth=1, relief="solid")
        image_label = tk.Label(frame, background="black")
        image_label.grid(row=0, column=0, sticky="nsew")
        caption = tk.Label(frame, text=f"{env_name} (env {env_index})")
        caption.grid(row=1, column=0, sticky="ew")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        tiles.append({
            "env_index": env_index,
            "frame": frame,
            "image_label": image_label,
            "caption": caption,
            "target_size": None,
        })

    last_frames: List[Optional[np.ndarray]] = [None] * num_tiles
    interval_ms = int(max(1.0, 1000.0 / max(target_fps, 1e-3)))

    root.update_idletasks()

    try:
        base_height = int(obs_s.shape[1])
        base_width = int(obs_s.shape[2])
    except Exception:
        base_height = base_width = 1

    caption_height = max((tile["caption"].winfo_reqheight() for tile in tiles), default=0)

    initial_cols = max(1, math.ceil(math.sqrt(num_tiles)))
    initial_rows = max(1, math.ceil(num_tiles / initial_cols))
    safe_scale = max(scale, 1e-3)
    base_scaled_width = max(base_width * safe_scale, 1.0)
    base_scaled_height = max(base_height * safe_scale, 1.0)
    initial_width = int(initial_cols * base_scaled_width + (initial_cols * 2 * tile_padding))
    initial_height = int(
        initial_rows * (base_scaled_height + caption_height) + (initial_rows * 2 * tile_padding)
    )

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    if screen_width > 0 and screen_height > 0:
        fit_scale = min(screen_width / max(initial_width, 1), screen_height / max(initial_height, 1), 1.0)
    else:
        fit_scale = 1.0
    if fit_scale < 1.0:
        initial_width = int(initial_width * fit_scale)
        initial_height = int(initial_height * fit_scale)

    root.geometry(f"{max(1, initial_width)}x{max(1, initial_height)}")

    current_layout = {"cols": 0, "rows": 0}
    current_target_size: tuple[int, int] | None = None
    layout_pending = False

    def apply_layout() -> None:
        nonlocal layout_pending, current_target_size
        layout_pending = False
        available_width = max(root.winfo_width(), 1)
        available_height = max(root.winfo_height(), 1)

        best_cols = 1
        best_rows = num_tiles
        best_area = -1.0
        best_size: tuple[int, int] | None = None

        for cols in range(1, num_tiles + 1):
            rows = math.ceil(num_tiles / cols)
            total_pad_x = cols * 2 * tile_padding
            total_pad_y = rows * 2 * tile_padding
            cell_width = (available_width - total_pad_x) / max(cols, 1)
            cell_height = (available_height - total_pad_y) / max(rows, 1)
            if cell_width <= 1 or cell_height <= 1:
                continue
            image_height_available = max(cell_height - caption_height, 1.0)
            width_scale = cell_width / max(base_scaled_width, 1)
            height_scale = image_height_available / max(base_scaled_height, 1)
            scale_factor = min(width_scale, height_scale)
            if scale_factor <= 0:
                continue
            target_width = int(max(1, base_scaled_width * scale_factor))
            target_height = int(max(1, base_scaled_height * scale_factor))
            tile_area = target_width * target_height
            if tile_area > best_area:
                best_area = tile_area
                best_cols = cols
                best_rows = rows
                best_size = (target_width, target_height)

        if best_size is None:
            best_size = (
                int(max(1, base_scaled_width)),
                int(max(1, base_scaled_height)),
            )

        if current_layout["cols"] != best_cols or current_layout["rows"] != best_rows:
            # Clear existing weight configuration.
            for r in range(current_layout["rows"]):
                root.grid_rowconfigure(r, weight=0)
            for c in range(current_layout["cols"]):
                root.grid_columnconfigure(c, weight=0)

            for tile in tiles:
                tile_frame = tile["frame"]
                tile_frame.grid_forget()

            for index, tile in enumerate(tiles):
                row = index // best_cols
                col = index % best_cols
                tile["frame"].grid(
                    row=row,
                    column=col,
                    padx=tile_padding,
                    pady=tile_padding,
                    sticky="nsew",
                )

            for r in range(best_rows):
                root.grid_rowconfigure(r, weight=1)
            for c in range(best_cols):
                root.grid_columnconfigure(c, weight=1)

            current_layout["cols"] = best_cols
            current_layout["rows"] = best_rows

        current_target_size = best_size
        for tile in tiles:
            tile["target_size"] = current_target_size

    def schedule_layout(_event=None) -> None:
        nonlocal layout_pending
        if not layout_pending:
            layout_pending = True
            root.after_idle(apply_layout)

    schedule_layout()
    root.bind("<Configure>", schedule_layout)

    def update_frame() -> None:
        if shutdown.is_set():
            root.destroy()
            return

        for tile_index, tile in enumerate(tiles):
            env_index = tile["env_index"]

            with torch.no_grad():
                frame_tensor = obs_s[env_index].detach()
                if frame_tensor.device.type != "cpu":
                    frame_tensor = frame_tensor.to("cpu")
                else:
                    frame_tensor = frame_tensor.clone()
            frame = frame_tensor.numpy()

            if frame.any():
                trimmed = _trim_black_borders(frame)
                frame_to_show = np.ascontiguousarray(trimmed)
                last_frames[tile_index] = frame_to_show.copy()
            elif last_frames[tile_index] is not None:
                frame_to_show = last_frames[tile_index]
            else:
                frame_to_show = np.zeros((1, 1, 3), dtype=np.uint8)

            image = Image.fromarray(frame_to_show)
            target_size = tile.get("target_size")
            if target_size is not None and all(val > 0 for val in target_size):
                target_width, target_height = target_size
                display_image = Image.new("RGB", (target_width, target_height))
                width_scale = target_width / max(image.width, 1)
                height_scale = target_height / max(image.height, 1)
                scale_factor = min(width_scale, height_scale)
                resized = image.resize(
                    (
                        max(1, int(image.width * scale_factor)),
                        max(1, int(image.height * scale_factor)),
                    ),
                    Image.NEAREST,
                )
                paste_x = max(0, (target_width - resized.width) // 2)
                paste_y = max(0, (target_height - resized.height) // 2)
                display_image.paste(resized, (paste_x, paste_y))
            else:
                display_image = image

            photo = ImageTk.PhotoImage(image=display_image)
            tile_label = tile["image_label"]
            tile_label.configure(image=photo)
            tile_label.image = photo

        root.after(interval_ms, update_frame)

    root.after(0, update_frame)

    def on_close() -> None:
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    try:
        root.mainloop()
    except Exception as exc:  # pragma: no cover - GUI loop exceptions
        print(f"[viewer] encountered an error: {exc}")



def _monitor_processes(
    procs: Sequence[mp.Process],
    shutdown,
    duration_seconds: Optional[int],
    start_time: float,
    failure_flag: Optional[threading.Event] = None,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Watch child processes and trigger shutdown when required."""

    while not shutdown.is_set():
        if stop_event is not None and stop_event.is_set():
            return

        if duration_seconds is not None and time.time() - start_time >= duration_seconds:
            shutdown.set()
            break

        if stop_event is not None:
            if stop_event.wait(timeout=15):
                return
        else:
            shutdown.wait(timeout=15)

        for proc in procs:
            if not proc.is_alive():
                print("RIP SOMEONE CRASHED", file=sys.stderr)
                if failure_flag is not None:
                    failure_flag.set()
                shutdown.set()
                return

        sys.stdout.flush()
        sys.stderr.flush()

def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])

    selected_games, game_display_names = resolve_games(args)
    if not selected_games:
        raise ValueError("No environments selected – specify at least one via --game/--games.")

    max_episode_steps = args.max_episode_steps
    if max_episode_steps is None:
        max_episode_steps = int(45 * 60 * args.fps)

    device = resolve_device(args.device)
    start_method = select_start_method(args.start_method)
    mp.set_start_method(start_method, force=True)

    num_envs = len(selected_games)
    print(f"Selected {num_envs} environment(s): {', '.join(game_display_names)}")
    print(f"Using device={device} | start_method={start_method} | fps={args.fps} | max_episode_steps={max_episode_steps}")

    obs_s = torch.zeros((num_envs, 250, 160, 3), dtype=torch.uint8, device=device).share_memory_()
    act_s = torch.zeros(num_envs, dtype=torch.int64, device=device).share_memory_()
    info_s = torch.zeros((num_envs, 4), dtype=torch.float32, device=device).share_memory_()
    shutdown = mp.Event()

    proc_configs = [
        {
            'target': agent_proc,
            'args': (
                obs_s,
                act_s,
                info_s,
                shutdown,
                selected_games,
                args.checkpoint_dir,
                int(args.checkpoint_interval),
                int(args.max_checkpoint_snapshots),
                bool(args.fresh_agent),
            ),
        }
    ]

    num_procs = args.num_procs or min(DEFAULT_NUM_PROCS, num_envs)
    num_procs = max(1, num_procs)

    game_instances = list(zip(selected_games, game_display_names))
    raw_chunks = np.array_split(np.array(game_instances, dtype=object), num_procs)
    game_chunks: List[List[tuple[str, str]]] = [list(chunk) for chunk in raw_chunks if len(chunk) > 0]

    first_start_at = time.time()
    for i, chunk in enumerate(game_chunks):
        offset = sum(len(c) for c in game_chunks[:i])
        proc_configs.append(
            {
                'target': env_proc,
                'args': (first_start_at, chunk, offset, obs_s, act_s, info_s, shutdown, args.fps, max_episode_steps),
            }
        )

    from bg_record import bg_record_proc

    proc_configs.append({'target': bg_record_proc, 'args': (obs_s, info_s, shutdown, game_display_names, first_start_at)})

    viewer_args: Optional[tuple] = None
    if args.show_viewer and num_envs:
        if args.viewer_game is not None:
            try:
                viewer_index = game_display_names.index(args.viewer_game)
            except ValueError:
                try:
                    viewer_index = selected_games.index(args.viewer_game)
                except ValueError as exc:
                    raise ValueError(
                        f"Requested viewer game '{args.viewer_game}' is not part of the selected game list."
                    ) from exc
            viewer_indices = [viewer_index]
        elif args.viewer_env_index is not None:
            viewer_index = args.viewer_env_index
            if not (0 <= viewer_index < num_envs):
                raise ValueError(
                    f"viewer index {viewer_index} is out of range for {num_envs} environment(s)."
                )
            viewer_indices = [viewer_index]
        else:
            viewer_indices = list(range(num_envs))
        viewer_args = (obs_s, shutdown, viewer_indices, game_display_names, args.viewer_scale, args.viewer_fps)

    procs = [mp.Process(**cfg) for cfg in proc_configs]

    for p in procs:
        p.start()

    failure_flag = threading.Event()
    try:
        duration = int(os.environ["RUNDURATIONSECONDS"])
        monitor_args = (procs, shutdown, duration, first_start_at)
        monitor_kwargs = {"failure_flag": failure_flag}

        if viewer_args is not None:
            stop_monitor = threading.Event()
            monitor_thread = threading.Thread(
                target=_monitor_processes,
                args=monitor_args,
                kwargs={**monitor_kwargs, "stop_event": stop_monitor},
                daemon=True,
            )
            monitor_thread.start()
            viewer_finished_cleanly = False
            try:
                viewer_proc(*viewer_args)
                viewer_finished_cleanly = True
            finally:
                stop_monitor.set()
                while monitor_thread.is_alive():
                    monitor_thread.join(timeout=0.5)
                if viewer_finished_cleanly and not shutdown.is_set():
                    _monitor_processes(*monitor_args, **monitor_kwargs)
        else:
            _monitor_processes(*monitor_args, **monitor_kwargs)
    except KeyboardInterrupt:
        print("\nShutdown signal received...")
    finally:
        shutdown.set()
        for p in procs:
            p.join(timeout=10)
        for p in procs:
            if p.is_alive():
                p.terminate()
        print("All processes terminated.")
        if failure_flag.is_set():
            sys.exit(1)


if __name__ == "__main__":
    main()

# agent ranking code has moved but essentially you want your top few episodes to score within 50% of the all-time record on each and every game. especially adventure, pong, pitfall, and skiing. nobody willing to face the pain with skiing... so much pain in that game lol
