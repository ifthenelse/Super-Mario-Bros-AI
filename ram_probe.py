"""
Validates Super Mario Bros (NES) RAM addresses before using them
as observation for NEAT.

The addresses below are the ones commonly documented for the NTSC ROM
of Super Mario Bros (see the Data Crystal RAM map). With a PAL/Europe
ROM some might be slightly different: that's why we validate them by
printing them while Mario moves, and checking the values make sense
(e.g. the X position increases when Mario moves right, Y changes during jumps).
"""

import time

import numpy as np

import stable_retro

# Candidate addresses (to be validated)
ADDR_X_PAGE = 0x006D       # Mario's horizontal page/screen
ADDR_X_SCREEN = 0x0086     # Mario's X position on the current screen
ADDR_Y_POS = 0x00CE        # Mario's Y position
ADDR_PLAYER_STATE = 0x000E  # Mario's state (e.g. 0x06/0x0B = dying)

N_ENEMY_SLOTS = 5
ADDR_ENEMY_DRAWN = 0x000F      # 5 bytes: 1 if the enemy in that slot is active
ADDR_ENEMY_X_SCREEN = 0x0087   # 5 bytes: enemy X position on screen
ADDR_ENEMY_Y_POS = 0x00CF      # 5 bytes: enemy Y position


def read_ram_values(ram: np.ndarray) -> dict:
    values = {
        "mario_x_page": int(ram[ADDR_X_PAGE]),
        "mario_x_screen": int(ram[ADDR_X_SCREEN]),
        "mario_y": int(ram[ADDR_Y_POS]),
        "player_state": int(ram[ADDR_PLAYER_STATE]),
        "enemies": [],
    }

    for i in range(N_ENEMY_SLOTS):
        drawn = int(ram[ADDR_ENEMY_DRAWN + i])
        if drawn:
            values["enemies"].append(
                {
                    "slot": i,
                    "x": int(ram[ADDR_ENEMY_X_SCREEN + i]),
                    "y": int(ram[ADDR_ENEMY_Y_POS + i]),
                }
            )

    return values


def main():
    env = stable_retro.make("SuperMarioBros-Nes-v0", render_mode="human")
    obs, info = env.reset()

    print("RAM address validation. Check whether the values make sense:")
    print("- mario_x_screen should change when Mario moves")
    print("- mario_y should change during jumps/falls")
    print("- player_state should change if Mario dies\n")

    n_steps = 300
    frame_time = 1.0 / 60.0
    print_every = 30  # print roughly every half second

    for step in range(n_steps):
        step_start = time.time()

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        ram = env.get_ram()
        values = read_ram_values(ram)

        if step % print_every == 0:
            print(f"[step {step:4d}] "
                  f"x_page={values['mario_x_page']:>3} "
                  f"x_screen={values['mario_x_screen']:>3} "
                  f"y={values['mario_y']:>3} "
                  f"state={values['player_state']:>3} "
                  f"active_enemies={len(values['enemies'])} "
                  f"| info_xscrollLo={info.get('xscrollLo')} "
                  f"lives={info.get('lives')}")

        env.render()

        elapsed = time.time() - step_start
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)

        if terminated or truncated:
            print(f"Episode ended at step {step}.")
            obs, info = env.reset()

    try:
        env.close()
    except AttributeError:
        pass

    print("\nValidation completed.")


if __name__ == "__main__":
    main()
