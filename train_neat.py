"""
NEAT training on Super Mario Bros (stable-retro).

Each genome in the NEAT population plays one episode (until it dies or
gets stuck for too long), guided by an observation built from RAM
(Mario's Y position + info about nearby enemies).
Fitness is the maximum distance reached in the level.
"""

import argparse
import os
import pickle
import re
import select
import sys
import time

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

TILE_BUFFER_BASE = 0x0500
TILE_ROWS = 13
TILE_COLS_PER_PAGE = 16

# Tile grid: columns ahead of/behind Mario and rows above/below (in pixels, step 16 = 1 tile)
TILE_COL_OFFSETS = [-16, 0, 16, 32, 48, 64, 80, 96]
TILE_ROW_OFFSETS = [-32, -16, 0, 16, 32]

MAX_STEPS_PER_EPISODE = 5000
STUCK_STEPS_LIMIT = 250  # end the episode if Mario hasn't progressed for N steps


def get_tile(ram: np.ndarray, mario_world_x: int, mario_y: int, dx: int, dy: int) -> int:
    """Returns 1 if the tile at (mario_world_x+dx, mario_y+dy) is solid, 0 otherwise."""
    x = mario_world_x + dx
    y = mario_y + dy - 16  # empirical vertical offset used in reference scripts

    page = (x // 256) % 2
    col = (x % 256) // 16
    row = (y - 32) // 16

    if row < 0 or row >= TILE_ROWS:
        return 0

    addr = TILE_BUFFER_BASE + page * TILE_ROWS * TILE_COLS_PER_PAGE + row * TILE_COLS_PER_PAGE + col
    if addr < 0 or addr >= len(ram):
        return 0

    return 1 if ram[addr] != 0 else 0


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
    ]

    # Enemies: presence, dx, dy, type (5 slots x 4 values = 20)

    for i in range(N_ENEMY_SLOTS):
        drawn = int(ram[ADDR_ENEMY_DRAWN + i])
        if drawn:
            enemy_x = int(ram[ADDR_ENEMY_X_SCREEN + i])
            enemy_y = int(ram[ADDR_ENEMY_Y_POS + i])
            enemy_type = int(ram[ADDR_ENEMY_TYPE + i])
            dx = (enemy_x - mario_x_screen) / 256.0
            dy = (enemy_y - mario_y) / 240.0
            obs.extend([1.0, dx, dy, enemy_type / 10.0])  # rough normalization
        else:
            obs.extend([0.0, 0.0, 0.0, 0.0])

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


def run_training(config_path: str, n_generations: int | None = None, time_budget_minutes: float | None = None,
                  checkpoint_prefix: str = "neat-checkpoint-", resume_from: str | None = None):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    if resume_from:
        print(f"Resuming training from checkpoint: {resume_from}")
        population = neat.Checkpointer.restore_checkpoint(resume_from)
    else:
        population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(5, filename_prefix=checkpoint_prefix))

    env = stable_retro.make("SuperMarioBros-Nes-v0", render_mode=None)

    def eval_wrapper(genomes, cfg):
        eval_genomes(genomes, cfg, env)

    start_time = time.time()

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

    elapsed = time.time() - start_time

    env.close()

    print(f"\nTraining completed in {elapsed / 60:.1f} minutes.")
    print(f"Best genome fitness: {winner.fitness}")

    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("Best genome saved to winner.pkl")

    return winner, stats


if __name__ == "__main__":
    import glob

    parser = argparse.ArgumentParser(description="NEAT training on Super Mario Bros.")
    parser.add_argument(
        "--minutes", "-m",
        type=str,
        default=None,
        help="Max time to run for. Plain number = minutes (e.g. '30'), "
             "or 'XXhYYm' (e.g. '1h30m'), or 'XXh' (e.g. '2h'). "
             "If omitted, you'll be prompted interactively.",
    )
    args = parser.parse_args()

    try:
        time_budget_minutes = resolve_time_budget_minutes(args.minutes)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    local_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(local_dir, "neat-config.txt")

    checkpoint_files = glob.glob(os.path.join(local_dir, "neat-checkpoint-*"))
    latest_checkpoint = None
    latest_gen = -1
    for f in checkpoint_files:
        match = re.search(r"neat-checkpoint-(\d+)$", f)
        if match:
            gen = int(match.group(1))
            if gen > latest_gen:
                latest_gen = gen
                latest_checkpoint = f

    if latest_checkpoint:
        print(f"Found checkpoint at generation {latest_gen}: resuming and running for up to "
              f"{time_budget_minutes:.1f} minutes.")
        run_training(config_path, time_budget_minutes=time_budget_minutes, resume_from=latest_checkpoint)
    else:
        print(f"No checkpoint found: starting training from scratch, running for up to "
              f"{time_budget_minutes:.1f} minutes.")
        run_training(config_path, time_budget_minutes=time_budget_minutes)
