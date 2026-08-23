"""
Quick test of the stable-retro environment for Super Mario Bros.
Takes random actions for a few hundred frames and shows the game window,
to confirm that the ROM, core, and emulator are communicating correctly.
"""

import time

import stable_retro


def main():
    env = stable_retro.make("SuperMarioBros-Nes-v0", render_mode="human")

    obs, info = env.reset()
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
    main()
