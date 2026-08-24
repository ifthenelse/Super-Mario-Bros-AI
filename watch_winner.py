"""
Loads the best genome saved in winner.pkl and plays it with the window
visible, at real game speed, to see concretely what it learned
(and where it gets stuck).

On every death, prints the observation the network was actually seeing in
the steps leading up to it (Mario's state, nearby enemies, jump-clearance
estimate, whether the jump button was pressed), to help diagnose whether a
death is caused by bad/missing information or by the network's decision.
"""

import os
import pickle
import sys
import time
from collections import deque
from datetime import datetime

import neat
import numpy as np
import pyglet

import stable_retro

from train_neat import Tee, build_observation, debug_snapshot, outputs_to_action

STEPS_BEFORE_DEATH_TO_SHOW = 60


def print_death_trace(trace):
    print("  Steps leading up to this death (most recent last):")
    for step, snap, jump_pressed in trace:
        enemy_str = "no enemies in range"
        if snap["enemies"]:
            # Show the closest enemy ahead (smallest positive dx), or the closest overall
            ahead = [e for e in snap["enemies"] if e["dx"] >= 0]
            e = min(ahead, key=lambda e: e["dx"]) if ahead else min(snap["enemies"], key=lambda e: abs(e["dx"]))
            enemy_str = (f"type={e['type']} dx={e['dx']:+d} dy={e['dy']:+d} "
                         f"clearance={e['ceiling_clearance']:.2f} time_to_impact={e['time_to_enemy']:+.2f}")
        print(f"    [step {step}] y={snap['mario_y']:3d} x_speed={snap['x_speed']:+3d} "
              f"y_speed={snap['y_speed']:+3d} jump_pressed={jump_pressed}  |  {enemy_str}")

    last_snap = trace[-1][1]
    print("  Tile grid at the moment of death (X = solid, . = empty, M = Mario):")
    for line in last_snap["tile_grid"]:
        print(f"    {line}")


def main():
    local_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(local_dir, "neat-config.txt")

    log_path = os.path.join(os.getcwd(), f"watch-{datetime.now():%Y%m%d-%H%M%S}.log")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file = open(log_path, "w")
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    try:
        print(f"Logging full output to: {log_path}")

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
        env.render()  # forces the pyglet window to be created, so we can hook key events

        speed_state = {"multiplier": 1.0, "paused": False}

        def on_key_press(symbol, modifiers):
            if symbol in (pyglet.window.key.PLUS, pyglet.window.key.EQUAL, pyglet.window.key.NUM_ADD):
                speed_state["multiplier"] = min(speed_state["multiplier"] * 1.5, 16.0)
                print(f"Speed: {speed_state['multiplier']:.2f}x")
            elif symbol in (pyglet.window.key.MINUS, pyglet.window.key.NUM_SUBTRACT):
                speed_state["multiplier"] = max(speed_state["multiplier"] / 1.5, 0.1)
                print(f"Speed: {speed_state['multiplier']:.2f}x")
            elif symbol == pyglet.window.key._0:
                speed_state["multiplier"] = 1.0
                print("Speed reset to 1.00x")
            elif symbol in (pyglet.window.key.SPACE, pyglet.window.key.ENTER, pyglet.window.key.RETURN):
                speed_state["paused"] = not speed_state["paused"]
                print("PAUSED" if speed_state["paused"] else "RESUMED")

        env.viewer.window.push_handlers(on_key_press=on_key_press)

        frame_time = 1.0 / 60.0
        max_world_x = 0
        prev_lives = info.get("lives")
        recent_steps = deque(maxlen=STEPS_BEFORE_DEATH_TO_SHOW)

        print(f"Loaded genome fitness (from training): {winner.fitness}")
        print("Controls (game window must have focus): '+' speeds up, '-' slows down, "
              "'0' resets to 1x, SPACE/ENTER pauses/resumes.")
        print("Starting the game...\n")

        step = 0
        while True:
            step_start = time.time()

            if speed_state["paused"]:
                env.render()  # keeps the window responsive to keys while paused
                time.sleep(0.05)
                continue

            observation = build_observation(ram)
            outputs = net.activate(observation)
            action = outputs_to_action(outputs)

            # Snapshot what the network saw and decided *before* stepping the emulator,
            # so a death detected right after this step can be traced back to this decision.
            recent_steps.append((step, debug_snapshot(ram), bool(action[8])))

            obs, reward, terminated, truncated, info = env.step(action)
            ram = env.get_ram()

            world_x = info.get("xscrollHi", 0) * 256 + info.get("xscrollLo", 0)
            if world_x > max_world_x:
                max_world_x = world_x

            lives = info.get("lives")
            if lives is not None and prev_lives is not None and lives < prev_lives:
                print(f"[step {step}] Mario lost a life. "
                      f"Position reached: {world_x} (max so far: {max_world_x})")
                print_death_trace(recent_steps)
            prev_lives = lives

            env.render()

            elapsed = time.time() - step_start
            target_frame_time = frame_time / speed_state["multiplier"]
            if elapsed < target_frame_time:
                time.sleep(target_frame_time - elapsed)

            if terminated or truncated:
                print(f"\nEpisode ended at step {step}.")
                print(f"Maximum distance reached: {max_world_x}")
                break

            step += 1

        try:
            env.close()
        except AttributeError:
            pass

        print(f"Full log saved to: {log_path}")
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


if __name__ == "__main__":
    main()
