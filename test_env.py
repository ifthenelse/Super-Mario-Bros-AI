"""
Quick test of the stable-retro environment for Super Mario Bros.
Takes random actions for a few hundred frames and shows the game window,
to confirm that the ROM, core, and emulator are communicating correctly.
"""

import argparse
import time

import stable_retro

from train_neat import load_state_offset, set_render_scale


def main(state: str | None = None):
    kwargs = {"render_mode": "human"}
    if state:
        kwargs["state"] = state
    env = stable_retro.make("SuperMarioBros-Nes-v0", **kwargs)

    obs, info = env.reset()
    set_render_scale(env)
    if state:
        offset = load_state_offset(state)
        print(
            f"Starting from state: {state}"
            + (f" (level_offset={offset})" if offset else "")
        )
    print("Environment created successfully.")
    print("Observation (frame) shape:", obs.shape)
    print("Action space:", env.action_space)

    total_reward = 0.0
    n_steps = 600  # ~10 seconds at 60fps, if properly throttled
    frame_time = 1.0 / 60.0

    for step in range(n_steps):
        step_start = time.time()

        action = env.action_space.sample()  # random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        env.render()

        # Throttle the loop to ~60fps real time instead of running at max CPU speed
        elapsed = time.time() - step_start
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)

        if terminated or truncated:
            print(f"Episode ended at step {step}. Total reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0.0

    try:
        env.close()
    except AttributeError:
        # Known pyglet 1.5.x bug on macOS (Cocoa) when closing the window.
        # Harmless: the window still closes correctly.
        pass

    print("Test completed without errors.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quick smoke test of the stable-retro environment."
    )
    parser.add_argument(
        "--state",
        "-s",
        type=str,
        default=None,
        help="stable-retro state to start from (e.g. a custom one saved via probe_level_type.py's "
        "'S' key), instead of the game's default start.",
    )
    args = parser.parse_args()
    main(args.state)
