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
import os
import random
import sys
import threading
import time
from typing import List, Sequence

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
    args = parser.parse_args(argv)

    if args.game and args.games:
        parser.error("--game and --games are mutually exclusive")

    return args


def resolve_games(args: argparse.Namespace) -> List[str]:
    """Return the ordered list of environments to launch based on CLI flags."""

    if args.game:
        return [args.game]
    if args.games:
        return list(dict.fromkeys(args.games))  # Preserve order while removing duplicates.
    return list(ALL_GAMES)


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
    print(f'{game_id=} {envseed=}')
    obs, _ = env.reset(seed=envseed)
    h, w, _ = obs.shape
    obs_s[g_idx, :h, :w].copy_(torch.from_numpy(obs), non_blocking=True)
    bind_logger(game_id, g_idx, info_s)
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
            args=(first_start_at, g, offset + i, obs_s, act_s, info_s, shutdown, fps, max_episode_steps),
        )
        for i, g in enumerate(game_chunk)
    ]
    for t in threads: t.start()
    for t in threads: t.join()

def agent_proc(obs_s, act_s, info_s, shutdown):
    seed('agent', 0)
    from myagent import Agent
    agent = Agent()

    save_path = "agent.pt"
    try: # only first load attempt allowed to fail
        print(f"loading from {save_path=}")
        agent.load(save_path)
    except Exception: pass
    print(f"saving to {save_path=}")
    agent.save(save_path)
    print(f"loading from {save_path=}")
    agent.load(save_path) # success required

    last_save_time = time.time()
    while not shutdown.is_set():
        # NOTE: THE AGENT IS CALLED IN A LOOP AS FAST AS POSSIBLE. THERE IS NO SLEEP STATEMENT IN THIS BLOCK.
        # (a very fast agent would do multiple passes per frame. a slow agent would take multiple frames to do a pass.)
        # NOTE: THE act_and_learn ARGUMENTS HAVE CHANGED
        # EACH ROW IN info_s IS LIKE (acc_reward, acc_frames, acc_term, acc_trunc)
        agent.act_and_learn(obs_s, info_s.clone(), act_s)
        if time.time() - last_save_time > 29*60:
            print(f"saving to {save_path=}")
            agent.save(save_path)
            print(f"loading from {save_path=}")
            agent.load(save_path)
            last_save_time = time.time()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])

    selected_games = resolve_games(args)
    if not selected_games:
        raise ValueError("No environments selected – specify at least one via --game/--games.")

    max_episode_steps = args.max_episode_steps
    if max_episode_steps is None:
        max_episode_steps = int(45 * 60 * args.fps)

    device = resolve_device(args.device)
    start_method = select_start_method(args.start_method)
    mp.set_start_method(start_method, force=True)

    num_envs = len(selected_games)
    print(f"Selected {num_envs} environment(s): {', '.join(selected_games)}")
    print(f"Using device={device} | start_method={start_method} | fps={args.fps} | max_episode_steps={max_episode_steps}")

    obs_s = torch.zeros((num_envs, 250, 160, 3), dtype=torch.uint8, device=device).share_memory_()
    act_s = torch.zeros(num_envs, dtype=torch.int64, device=device).share_memory_()
    info_s = torch.zeros((num_envs, 4), dtype=torch.float32, device=device).share_memory_()
    shutdown = mp.Event()

    proc_configs = [{'target': agent_proc, 'args': (obs_s, act_s, info_s, shutdown)}]

    num_procs = args.num_procs or min(DEFAULT_NUM_PROCS, num_envs)
    num_procs = max(1, num_procs)
    # ``np.array_split`` keeps the ordering of ``selected_games`` while distributing
    # them roughly evenly across worker processes.
    raw_chunks = np.array_split(selected_games, num_procs)
    game_chunks: List[List[str]] = [list(chunk) for chunk in raw_chunks if len(chunk) > 0]

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

    proc_configs.append({'target': bg_record_proc, 'args': (obs_s, info_s, shutdown, selected_games, first_start_at)})

    procs = [mp.Process(**cfg) for cfg in proc_configs]

    for p in procs:
        p.start()

    try:
        duration = int(os.environ["RUNDURATIONSECONDS"])
        while time.time() - first_start_at < duration:
            time.sleep(15)
            for i, p in enumerate(procs):
                if not p.is_alive():
                    print("RIP SOMEONE CRASHED", file=sys.stderr)
                    sys.exit(1)
            sys.stdout.flush()
            sys.stderr.flush()
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


if __name__ == "__main__":
    main()

# agent ranking code has moved but essentially you want your top few episodes to score within 50% of the all-time record on each and every game. especially adventure, pong, pitfall, and skiing. nobody willing to face the pain with skiing... so much pain in that game lol
