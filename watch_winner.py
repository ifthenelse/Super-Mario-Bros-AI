"""
Loads the best genome saved in winner.pkl and plays it with the window
visible, at real game speed, to see concretely what it learned
(and where it gets stuck).
"""

import os
import pickle
import time

import neat
import numpy as np

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

    print(f"Loaded genome fitness (from training): {winner.fitness}")
    print("Starting the game...\n")

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
            print(f"[step {step}] Mario lost a life. "
                  f"Position reached: {world_x} (max so far: {max_world_x})")
        prev_lives = lives

        env.render()

        elapsed = time.time() - step_start
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)

        if terminated or truncated:
            print(f"\nEpisode ended at step {step}.")
            print(f"Maximum distance reached: {max_world_x}")
            break

        step += 1

    try:
        env.close()
    except AttributeError:
        pass


if __name__ == "__main__":
    main()
