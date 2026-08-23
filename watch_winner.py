"""
Carica il miglior genoma salvato in winner.pkl e lo fa giocare
con la finestra visibile, a velocità di gioco reale, per osservare
concretamente cosa ha imparato (e dove si blocca).
"""

import os
import pickle
import time

import neat

import stable_retro

from train_neat import build_observation, outputs_to_action


def main():
    local_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(local_dir, "neat-config.txt")

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    with open("winner.pkl", "rb") as f:
        winner = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(winner, config)

    env = stable_retro.make("SuperMarioBros-Nes-v0", render_mode="human")
    obs, info = env.reset()
    ram = env.get_ram()

    frame_time = 1.0 / 60.0
    max_world_x = 0
    prev_lives = info.get("lives")

    print(f"Fitness del genoma caricato (dal training): {winner.fitness}")
    print("Avvio partita...\n")

    step = 0
    while True:
        step_start = time.time()

        observation = build_observation(ram)
        outputs = net.activate(observation)
        action = outputs_to_action(outputs)

        obs, reward, terminated, truncated, info = env.step(action)
        ram = env.get_ram()

        world_x = info.get("xscrollHi", 0) * 256 + info.get("xscrollLo", 0)
        if world_x > max_world_x:
            max_world_x = world_x

        lives = info.get("lives")
        if lives is not None and prev_lives is not None and lives < prev_lives:
            print(f"[step {step}] Mario ha perso una vita. "
                  f"Posizione raggiunta: {world_x} (max finora: {max_world_x})")
        prev_lives = lives

        env.render()

        elapsed = time.time() - step_start
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)

        if terminated or truncated:
            print(f"\nEpisodio terminato allo step {step}.")
            print(f"Distanza massima raggiunta: {max_world_x}")
            break

        step += 1

    try:
        env.close()
    except AttributeError:
        pass


if __name__ == "__main__":
    main()
