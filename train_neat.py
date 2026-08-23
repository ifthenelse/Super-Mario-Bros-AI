"""
Training NEAT su Super Mario Bros (stable-retro).

Ogni genoma della popolazione NEAT gioca un episodio (o finché non muore /
resta bloccato troppo a lungo), guidato da un'osservazione costruita a
partire dalla RAM (posizione Y di Mario + info sui nemici vicini).
La fitness è la distanza massima raggiunta nel livello.
"""

import os
import pickle
import time

import neat
import numpy as np

import stable_retro

# --- Indirizzi RAM validati con ram_probe.py ---
ADDR_X_SCREEN = 0x0086
ADDR_Y_POS = 0x00CE
N_ENEMY_SLOTS = 5
ADDR_ENEMY_DRAWN = 0x000F
ADDR_ENEMY_X_SCREEN = 0x0087
ADDR_ENEMY_Y_POS = 0x00CF

MAX_STEPS_PER_EPISODE = 5000
STUCK_STEPS_LIMIT = 250  # termina l'episodio se Mario non avanza per N step


def build_observation(ram: np.ndarray) -> list:
    """Costruisce il vettore di input a 16 valori per la rete NEAT."""
    mario_x = int(ram[ADDR_X_SCREEN])
    mario_y = int(ram[ADDR_Y_POS])

    obs = [mario_y / 240.0]  # posizione Y normalizzata

    for i in range(N_ENEMY_SLOTS):
        drawn = int(ram[ADDR_ENEMY_DRAWN + i])
        if drawn:
            enemy_x = int(ram[ADDR_ENEMY_X_SCREEN + i])
            enemy_y = int(ram[ADDR_ENEMY_Y_POS + i])
            dx = (enemy_x - mario_x) / 256.0
            dy = (enemy_y - mario_y) / 240.0
            obs.extend([1.0, dx, dy])
        else:
            obs.extend([0.0, 0.0, 0.0])

    return obs


def outputs_to_action(outputs) -> np.ndarray:
    """Converte gli output della rete (continui) in un array di bottoni 0/1."""
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
                # Mario è bloccato da troppo tempo: inutile continuare l'episodio
                break

        genome.fitness = float(max_world_x)


def run_training(config_path: str, n_generations: int = 50, checkpoint_prefix: str = "neat-checkpoint-"):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

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

    print(f"\nTraining completato in {elapsed / 60:.1f} minuti.")
    print(f"Fitness del miglior genoma: {winner.fitness}")

    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("Miglior genoma salvato in winner.pkl")

    return winner, stats


if __name__ == "__main__":
    local_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(local_dir, "neat-config.txt")
    run_training(config_path, n_generations=50)
