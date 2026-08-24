"""
NEAT training on Super Mario Bros (stable-retro).

Each genome in the NEAT population plays one episode (until it dies or
gets stuck for too long), guided by an observation built from RAM
(Mario's Y position + info about nearby enemies).
Fitness is the maximum distance reached in the level.
"""

import argparse
import glob
import gzip
import itertools
import json
import os
import pickle
import random
import re
import select
import sys
import time
from datetime import datetime

import neat
import numpy as np

import stable_retro

DEFAULT_TIME_BUDGET_MINUTES = 60.0
PROMPT_TIMEOUT_SECONDS = 15


def parse_duration_to_minutes(raw: str) -> float:
    """Parses a duration string into minutes.

    Accepted formats:
    - a plain number (interpreted as minutes), e.g. "30" or "12.5"
    - "XXhYYm" (hours and minutes), e.g. "1h30m"
    - "XXh" (hours only), e.g. "2h"

    Raises ValueError on any other format.
    """
    value = raw.strip()

    if re.fullmatch(r"\d+(\.\d+)?", value):
        return float(value)

    match = re.fullmatch(r"(\d+)h(\d+)m", value)
    if match:
        hours, minutes = int(match.group(1)), int(match.group(2))
        return hours * 60 + minutes

    match = re.fullmatch(r"(\d+)h", value)
    if match:
        hours = int(match.group(1))
        return hours * 60

    raise ValueError(
        f"Invalid time format: '{raw}'. Use a plain number of minutes (e.g. '30'), "
        f"'XXhYYm' (e.g. '1h30m'), or 'XXh' (e.g. '2h')."
    )


def resolve_time_budget_minutes(cli_value: str | None) -> float:
    """Resolves the time budget for this run: from the CLI argument if given,
    otherwise by prompting the user with a timeout, falling back to the default."""
    if cli_value is not None:
        return parse_duration_to_minutes(cli_value)

    print(f"The script will run for {DEFAULT_TIME_BUDGET_MINUTES:.0f} minutes. "
          f"Type a value in minutes if you want to change it: ", end="", flush=True)

    ready, _, _ = select.select([sys.stdin], [], [], PROMPT_TIMEOUT_SECONDS)
    if not ready:
        print(f"\nNo input received within {PROMPT_TIMEOUT_SECONDS} seconds. "
              f"Using the default: {DEFAULT_TIME_BUDGET_MINUTES:.0f} minutes.")
        return DEFAULT_TIME_BUDGET_MINUTES

    line = sys.stdin.readline().strip()
    if line == "":
        print(f"Using the default: {DEFAULT_TIME_BUDGET_MINUTES:.0f} minutes.")
        return DEFAULT_TIME_BUDGET_MINUTES

    return parse_duration_to_minutes(line)

# --- RAM addresses validated with ram_probe.py / ram_probe_advanced.py ---
ADDR_X_SCREEN = 0x0086
ADDR_X_PAGE = 0x006D
ADDR_Y_POS = 0x00CE
ADDR_X_SPEED = 0x0057
ADDR_Y_SPEED = 0x009F

N_ENEMY_SLOTS = 5
ADDR_ENEMY_DRAWN = 0x000F
ADDR_ENEMY_TYPE = 0x0016   # validated: enemy type per slot (e.g. 6 = Koopa Troopa)
ADDR_ENEMY_X_SCREEN = 0x0087
ADDR_ENEMY_Y_POS = 0x00CF

ADDR_POWER_STATE = 0x0756  # candidate, NOT yet validated on this ROM: 0=small, 1=big, 2=fire

ADDR_LEVEL_HI = 1887  # validated (from data.json): world index (0-based)
ADDR_LEVEL_LO = 1884  # validated (from data.json): level-within-world index (0-based)

# Static world-level -> type lookup, based on the well-known original SMB1 level design:
# castles are always the 4th level of each world, water levels are 2-2 and 7-2,
# everything else (including underground levels, which share normal run/jump physics) is "normal".
WATER_LEVELS = {(2, 2), (7, 2)}


def get_level_type(world: int, level: int) -> str:
    if level == 4:
        return "castle"
    if (world, level) in WATER_LEVELS:
        return "water"
    return "normal"


def get_level_type_onehot(world: int, level: int) -> tuple:
    level_type = get_level_type(world, level)
    return (
        1.0 if level_type == "normal" else 0.0,
        1.0 if level_type == "water" else 0.0,
        1.0 if level_type == "castle" else 0.0,
    )

TILE_BUFFER_BASE = 0x0500
TILE_ROWS = 13
TILE_COLS_PER_PAGE = 16

# Tile grid: columns ahead of/behind Mario and rows above/below (in pixels, step 16 = 1 tile)
# Extended forward reach (more reaction time) and one extra row up (ceiling awareness for narrow passages)
TILE_COL_OFFSETS = [-16, 0, 16, 32, 48, 64, 80, 96, 112, 128]
TILE_ROW_OFFSETS = [-48, -32, -16, 0, 16, 32]

ENEMY_CEILING_CHECK_TILES = 3  # how many tiles above an enemy to check for jump-over clearance

MAX_STEPS_PER_EPISODE = 5000
STUCK_STEPS_LIMIT = 250  # end the episode if Mario hasn't progressed for N steps


def get_tile_absolute(ram: np.ndarray, x: int, y: int) -> int:
    """Returns 1 if the tile at absolute world coordinates (x, y) is solid, 0 otherwise."""
    page = (x // 256) % 2
    col = (x % 256) // 16
    row = (y - 32) // 16

    if row < 0 or row >= TILE_ROWS:
        return 0

    addr = TILE_BUFFER_BASE + page * TILE_ROWS * TILE_COLS_PER_PAGE + row * TILE_COLS_PER_PAGE + col
    if addr < 0 or addr >= len(ram):
        return 0

    return 1 if ram[addr] != 0 else 0


def get_tile(ram: np.ndarray, mario_world_x: int, mario_y: int, dx: int, dy: int) -> int:
    """Returns 1 if the tile at (mario_world_x+dx, mario_y+dy) is solid, 0 otherwise."""
    x = mario_world_x + dx
    y = mario_y + dy - 16  # empirical vertical offset used in reference scripts
    return get_tile_absolute(ram, x, y)


def enemy_ceiling_clearance(ram: np.ndarray, enemy_world_x: int, enemy_y: int,
                             max_check: int = ENEMY_CEILING_CHECK_TILES) -> float:
    """Counts how many empty tiles are directly above an enemy (up to max_check),
    normalized to 0-1. A low value means there's little or no room to jump onto
    the enemy from above; a high value means it's safe to land on top of it."""
    clearance = 0
    for i in range(1, max_check + 1):
        y = enemy_y - 16 * i - 16  # same vertical offset convention as get_tile
        if get_tile_absolute(ram, enemy_world_x, y) == 0:
            clearance += 1
        else:
            break
    return clearance / max_check


def build_observation(ram: np.ndarray) -> list:
    """Builds the input vector for the NEAT network: Y position, speed,
    nearby enemies, and terrain tile grid."""
    mario_x_screen = int(ram[ADDR_X_SCREEN])
    mario_x_page = int(ram[ADDR_X_PAGE])
    mario_world_x = mario_x_page * 256 + mario_x_screen
    mario_y = int(ram[ADDR_Y_POS])

    x_speed = int(np.int8(ram[ADDR_X_SPEED]))
    y_speed = int(np.int8(ram[ADDR_Y_SPEED]))

    obs = [
        mario_y / 240.0,
        x_speed / 30.0,   # rough normalization (observed max speed ~28)
        y_speed / 10.0,   # rough normalization (observed max speed ~5)
        int(ram[ADDR_POWER_STATE]) / 2.0,  # 0=small, 1=big, 2=fire (rough normalization)
    ]

    world = int(np.int8(ram[ADDR_LEVEL_HI])) + 1
    level = int(np.int8(ram[ADDR_LEVEL_LO])) + 1
    obs.extend(get_level_type_onehot(world, level))

    # Enemies: presence, dx, dy, type, ceiling clearance above, estimated time-to-impact
    # (5 slots x 6 values = 30)

    for i in range(N_ENEMY_SLOTS):
        drawn = int(ram[ADDR_ENEMY_DRAWN + i])
        if drawn:
            enemy_x = int(ram[ADDR_ENEMY_X_SCREEN + i])
            enemy_y = int(ram[ADDR_ENEMY_Y_POS + i])
            enemy_type = int(ram[ADDR_ENEMY_TYPE + i])
            dx = enemy_x - mario_x_screen
            dy = enemy_y - mario_y
            enemy_world_x = mario_world_x + dx

            clearance = enemy_ceiling_clearance(ram, enemy_world_x, enemy_y)
            # Rough "frames until horizontally aligned" estimate, sign preserved (ahead/behind)
            time_to_enemy = dx / (abs(x_speed) + 1) / 50.0

            obs.extend([1.0, dx / 256.0, dy / 240.0, enemy_type / 10.0, clearance, time_to_enemy])
        else:
            obs.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    for row_offset in TILE_ROW_OFFSETS:
        for col_offset in TILE_COL_OFFSETS:
            obs.append(float(get_tile(ram, mario_world_x, mario_y, col_offset, row_offset)))

    return obs


def outputs_to_action(outputs) -> np.ndarray:
    """Converts the network's (continuous) outputs into a 0/1 button array."""
    return np.array([1 if o > 0.5 else 0 for o in outputs], dtype=np.int8)


def eval_genomes(genomes, config, env, render=False):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        obs, info = env.reset()
        ram = env.get_ram()

        max_world_x = 0
        last_progress_step = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            observation = build_observation(ram)
            outputs = net.activate(observation)
            action = outputs_to_action(outputs)

            obs, reward, terminated, truncated, info = env.step(action)
            ram = env.get_ram()

            world_x = info.get("xscrollHi", 0) * 256 + info.get("xscrollLo", 0)
            if world_x > max_world_x:
                max_world_x = world_x
                last_progress_step = step

            if render:
                env.render()

            if terminated or truncated:
                break

            if step - last_progress_step > STUCK_STEPS_LIMIT:
                # Mario has been stuck too long: no point continuing the episode
                break

        genome.fitness = float(max_world_x)


def find_checkpoints(root_dir: str) -> list:
    """Recursively finds all NEAT checkpoint files under root_dir (including subfolders)."""
    pattern = os.path.join(root_dir, "**", "neat-checkpoint-*")
    return sorted(glob.glob(pattern, recursive=True))


def read_checkpoint_summary(filename: str) -> dict | None:
    """Reads just enough from a checkpoint file to get its best fitness so
    far, without needing a matching neat-config.txt. Returns None if the
    file can't be read (e.g. corrupted or incompatible)."""
    try:
        with gzip.open(filename) as f:
            _generation, _config, population, _species_set, _rndstate = pickle.load(f)
    except Exception as e:
        print(f"Warning: could not read checkpoint '{filename}' ({e}); skipping it.")
        return None

    fitnesses = [g.fitness for g in population.values() if g.fitness is not None]
    return {"best_fitness": max(fitnesses) if fitnesses else None}


RUN_INFO_FILENAME = "run_info.json"


def generate_run_id() -> str:
    return datetime.now().astimezone().strftime("run-%Y%m%d-%H%M%S")


def load_run_info(run_dir: str) -> dict | None:
    path = os.path.join(run_dir, RUN_INFO_FILENAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def write_run_info(run_dir: str, run_id: str, parent_run_id: str | None,
                    start_time: datetime, end_time: datetime | None = None,
                    best_fitness: float | None = None):
    info = {
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat() if end_time else None,
        "best_fitness": best_fitness,
    }
    with open(os.path.join(run_dir, RUN_INFO_FILENAME), "w") as f:
        json.dump(info, f, indent=2)


def find_run_dirs(root_dir: str) -> list:
    """Finds every directory under root_dir (root_dir itself included) that
    contains at least one NEAT checkpoint file — i.e. one training run."""
    dirs = {os.path.dirname(p) for p in find_checkpoints(root_dir)}
    return sorted(dirs)


def summarize_run(run_dir: str, root_dir: str) -> dict:
    """Builds a one-line-worthy summary of a training run: identity,
    parentage, timing, and best fitness reached. Falls back to inferring
    missing info (legacy runs predating run_info.json) from the checkpoint
    files themselves."""
    checkpoint_files = glob.glob(os.path.join(run_dir, "neat-checkpoint-*"))
    latest_checkpoint, latest_gen = None, -1
    for f in checkpoint_files:
        m = re.search(r"neat-checkpoint-(\d+)$", f)
        if m:
            gen = int(m.group(1))
            if gen > latest_gen:
                latest_gen, latest_checkpoint = gen, f

    info = load_run_info(run_dir)

    if info:
        run_id = info["run_id"]
        parent_run_id = info.get("parent_run_id")
        start_time = datetime.fromisoformat(info["start_time"]) if info.get("start_time") else None
        end_time = datetime.fromisoformat(info["end_time"]) if info.get("end_time") else None
        best_fitness = info.get("best_fitness")
    else:
        run_id = "current" if os.path.abspath(run_dir) == os.path.abspath(root_dir) else os.path.basename(run_dir)
        parent_run_id = None
        start_time = None
        end_time = None
        best_fitness = None

    if best_fitness is None and latest_checkpoint:
        checkpoint_summary = read_checkpoint_summary(latest_checkpoint)
        best_fitness = checkpoint_summary["best_fitness"] if checkpoint_summary else None

    if start_time is None and checkpoint_files:
        oldest = min(checkpoint_files, key=os.path.getmtime)
        start_time = datetime.fromtimestamp(os.path.getmtime(oldest)).astimezone()

    if end_time is None and latest_checkpoint:
        end_time = datetime.fromtimestamp(os.path.getmtime(latest_checkpoint)).astimezone()

    return {
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "start_time": start_time,
        "end_time": end_time,
        "best_fitness": best_fitness,
        "resume_checkpoint": latest_checkpoint,
        "generation": latest_gen if latest_gen >= 0 else None,
        "dir": run_dir,
    }


def compute_run_arrows(runs: list) -> dict:
    """Compares each run's best fitness to its parent run's best fitness
    (when the parent is also in the list), returning {run_id: '▲'|'='|'▼'|' '}."""
    by_id = {r["run_id"]: r for r in runs}
    arrows = {}
    for r in runs:
        parent = by_id.get(r["parent_run_id"]) if r["parent_run_id"] else None
        if parent is None or parent["best_fitness"] is None or r["best_fitness"] is None:
            arrows[r["run_id"]] = " "
        elif r["best_fitness"] > parent["best_fitness"]:
            arrows[r["run_id"]] = "▲"
        elif r["best_fitness"] < parent["best_fitness"]:
            arrows[r["run_id"]] = "▼"
        else:
            arrows[r["run_id"]] = "="
    return arrows


def format_duration(start: datetime | None, end: datetime | None) -> str:
    if start is None or end is None:
        return "n/a"
    seconds = int((end - start).total_seconds())
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def format_run_line(r: dict, arrow: str) -> str:
    fitness_str = f"{r['best_fitness']:.0f}" if r["best_fitness"] is not None else "n/a"
    start_str = r["start_time"].strftime("%Y-%m-%d %H:%M %Z") if r["start_time"] else "n/a"
    end_str = r["end_time"].strftime("%Y-%m-%d %H:%M %Z") if r["end_time"] else "in progress"
    duration_str = format_duration(r["start_time"], r["end_time"])
    parent_str = r["parent_run_id"] or "-"
    return (f"{arrow} {r['run_id']:<20} parent={parent_str:<20} "
            f"start={start_str:<20} end={end_str:<20} "
            f"dur={duration_str:<8} fitness={fitness_str}")


def pick_run_interactively(runs: list, arrows: dict, default_index: int) -> int:
    """Full-screen arrow-key picker (curses). Up/Down to move, SPACE/ENTER to
    select, ESC to pick 'fresh start'. Auto-selects default_index if no key
    is pressed at all within PROMPT_TIMEOUT_SECONDS; once any key is pressed,
    the timeout is cancelled and it waits for an explicit choice."""
    import curses

    def _inner(stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        current = default_index
        fresh_idx = len(runs)
        n_options = len(runs) + 1
        deadline = time.time() + PROMPT_TIMEOUT_SECONDS
        interacted = False

        while True:
            stdscr.erase()
            stdscr.addstr(0, 0, "Select a run to resume from  (up/down: move, SPACE/ENTER: select, ESC: fresh start)")
            if not interacted:
                remaining = max(0.0, deadline - time.time())
                stdscr.addstr(1, 0, f"Auto-selecting the default in {remaining:4.1f}s if no input...")
            for i, r in enumerate(runs):
                marker = "> " if i == current else "  "
                attr = curses.A_REVERSE if i == current else curses.A_NORMAL
                tag = " (default)" if i == default_index else ""
                line = f"{marker}{format_run_line(r, arrows.get(r['run_id'], ' '))}{tag}"
                try:
                    stdscr.addstr(3 + i, 0, line, attr)
                except curses.error:
                    pass
            marker = "> " if current == fresh_idx else "  "
            attr = curses.A_REVERSE if current == fresh_idx else curses.A_NORMAL
            try:
                stdscr.addstr(3 + fresh_idx + 1, 0, f"{marker}[Start a fresh training run]", attr)
            except curses.error:
                pass
            stdscr.refresh()

            if not interacted and time.time() > deadline:
                return default_index

            stdscr.timeout(150)
            key = stdscr.getch()
            if key == -1:
                continue
            interacted = True
            if key in (curses.KEY_UP, ord('k')):
                current = (current - 1) % n_options
            elif key in (curses.KEY_DOWN, ord('j')):
                current = (current + 1) % n_options
            elif key in (10, 13, curses.KEY_ENTER, ord(' ')):
                return current
            elif key == 27:
                return fresh_idx

    return curses.wrapper(_inner)


def restore_checkpoint_with_config(filename: str, config: neat.Config) -> neat.Population:
    """Like neat.Checkpointer.restore_checkpoint, but rebuilds the population
    using the given (freshly loaded) config instead of the one frozen inside
    the checkpoint file. Without this, resuming from a checkpoint silently
    keeps whatever mutation rates, elitism, etc. were in effect when that
    checkpoint was saved, ignoring any changes since made to neat-config.txt."""
    with gzip.open(filename) as f:
        generation, _old_config, population, species_set, rndstate = pickle.load(f)
    random.setstate(rndstate)
    pop = neat.Population(config, (population, species_set, generation))

    # A freshly-built config's node-id counter starts from scratch. Left as-is,
    # it would eventually hand out a node ID that's already in use by some
    # genome carried over from the checkpoint (different lineages can have
    # very different node ID ranges), crashing with an AssertionError deep
    # inside a later mutation. Seed it past the highest node ID already in
    # use anywhere in the restored population to avoid that collision.
    max_node_id = -1
    for genome in population.values():
        if genome.nodes:
            max_node_id = max(max_node_id, max(genome.nodes.keys()))
    if max_node_id >= 0:
        config.genome_config.node_indexer = itertools.count(max_node_id + 1)

    return pop


def run_training(config_path: str, n_generations: int | None = None, time_budget_minutes: float | None = None,
                  checkpoint_prefix: str = "neat-checkpoint-", resume_from: str | None = None):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    run_dir = os.getcwd()
    run_start_time = datetime.now().astimezone()
    run_id = generate_run_id()

    parent_run_id = None
    if resume_from:
        parent_dir = os.path.dirname(os.path.abspath(resume_from)) or run_dir
        parent_info = load_run_info(parent_dir)
        parent_run_id = parent_info["run_id"] if parent_info else os.path.basename(parent_dir)
        print(f"Resuming training from checkpoint: {resume_from} (using the current neat-config.txt)")
        population = restore_checkpoint_with_config(resume_from, config)
    else:
        population = neat.Population(config)

    write_run_info(run_dir, run_id, parent_run_id, run_start_time)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(5, filename_prefix=checkpoint_prefix))

    env = stable_retro.make("SuperMarioBros-Nes-v0", render_mode=None)

    def eval_wrapper(genomes, cfg):
        eval_genomes(genomes, cfg, env)

    start_time = time.time()
    winner = None

    try:
        if time_budget_minutes is not None:
            budget_seconds = time_budget_minutes * 60
            gen_count = 0
            while True:
                elapsed = time.time() - start_time
                remaining = budget_seconds - elapsed
                if remaining <= 0:
                    print(f"\nTime budget reached ({time_budget_minutes:.0f} min). Stopping after "
                          f"{gen_count} generation(s) in this run.")
                    break
                print(f"\n[Time budget: {remaining / 60:.1f} min remaining]")
                population.run(eval_wrapper, 1)
                gen_count += 1
            winner = stats.best_genome()
        else:
            winner = population.run(eval_wrapper, n_generations)
    finally:
        env.close()
        try:
            best_so_far = stats.best_genome().fitness
        except Exception:
            best_so_far = None
        write_run_info(run_dir, run_id, parent_run_id, run_start_time,
                        end_time=datetime.now().astimezone(), best_fitness=best_so_far)

    elapsed = time.time() - start_time

    print(f"\nTraining completed in {elapsed / 60:.1f} minutes.")
    print(f"Best genome fitness: {winner.fitness}")

    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("Best genome saved to winner.pkl")

    return winner, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEAT training on Super Mario Bros.")
    parser.add_argument(
        "--minutes", "-m",
        type=str,
        default=None,
        help="Max time to run for. Plain number = minutes (e.g. '30'), "
             "or 'XXhYYm' (e.g. '1h30m'), or 'XXh' (e.g. '2h'). "
             "If omitted, you'll be prompted interactively.",
    )
    parser.add_argument(
        "--run", "-r",
        type=str,
        default=None,
        help="Run ID (or folder name) to resume from, skipping the interactive picker. "
             "Use 'none' to force a fresh start even if previous runs exist.",
    )
    args = parser.parse_args()

    try:
        time_budget_minutes = resolve_time_budget_minutes(args.minutes)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    local_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(local_dir, "neat-config.txt")
    search_root = os.getcwd()

    run_dirs = find_run_dirs(search_root)
    runs = [summarize_run(d, search_root) for d in run_dirs]

    resume_path = None

    if args.run is not None:
        if args.run.lower() == "none":
            print("Starting a fresh training run (forced via --run none).")
        else:
            match = next(
                (r for r in runs if r["run_id"] == args.run or os.path.basename(r["dir"]) == args.run),
                None,
            )
            if match is None:
                print(f"Error: no run found matching '{args.run}'.")
                sys.exit(1)
            resume_path = match["resume_checkpoint"]
            print(f"Using run: {match['run_id']}")
    elif not runs:
        print("No previous runs found: starting fresh.")
    else:
        dated_runs = [r for r in runs if r["start_time"] is not None]
        default_index = (
            runs.index(max(dated_runs, key=lambda r: r["start_time"]))
            if dated_runs else 0
        )
        arrows = compute_run_arrows(runs)
        choice = pick_run_interactively(runs, arrows, default_index)
        if choice == len(runs):
            print("\nStarting a fresh training run.")
        else:
            resume_path = runs[choice]["resume_checkpoint"]
            print(f"\nResuming from run: {runs[choice]['run_id']}")

    if resume_path:
        run_training(config_path, time_budget_minutes=time_budget_minutes, resume_from=resume_path)
    else:
        print(f"Running for up to {time_budget_minutes:.1f} minutes.")
        run_training(config_path, time_budget_minutes=time_budget_minutes)
