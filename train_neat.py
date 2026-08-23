"""
NEAT training on Super Mario Bros (stable-retro).

Each genome in the NEAT population plays one episode (until it dies or
gets stuck for too long), guided by an observation built from RAM
(Mario's Y position + info about nearby enemies).
Fitness is the maximum distance reached in the level.
"""

import os
import pickle
import time

import neat
import numpy as np

import stable_retro

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


def run_training(config_path: str, n_generations: int = 50, checkpoint_prefix: str = "neat-checkpoint-",
                  resume_from: str | None = None):
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
    import re

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

    ADDITIONAL_GENERATIONS = 50  # how many generations to add on top of the found checkpoint

    if latest_checkpoint:
        print(f"Found checkpoint at generation {latest_gen}: continuing for "
              f"{ADDITIONAL_GENERATIONS} more generations (total {latest_gen + ADDITIONAL_GENERATIONS}).")
        run_training(config_path, n_generations=ADDITIONAL_GENERATIONS, resume_from=latest_checkpoint)
    else:
        print("No checkpoint found: starting training from scratch (100 generations).")
        run_training(config_path, n_generations=100)
