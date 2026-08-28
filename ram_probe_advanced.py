"""
Validates the reading of:
- the local tile grid around Mario (terrain/solid blocks, pits)
- Mario's X and Y speed

The formula for reading tiles from the level buffer (base address 0x0500)
is the one historically used by Lua bot scripts for SMB on NES: since the
game screen is split into two "pages" of 13 rows x 16 columns of tiles,
it computes which page/row/column a world point (x, y) falls into and
reads the corresponding byte (0 = empty, non-zero = solid).

The speed addresses (0x0057 for X, 0x009F for Y) are candidates
documented in the classic SMB disassembly: we validate them by printing
them while Mario walks/jumps/falls.
"""

import argparse
import time

import numpy as np
import pyglet

import stable_retro

from train_neat import load_state_offset, set_render_scale

ADDR_X_SCREEN = 0x0086
ADDR_X_PAGE = 0x006D
ADDR_Y_POS = 0x00CE

ADDR_X_SPEED = 0x0057  # candidate: horizontal speed
ADDR_Y_SPEED = 0x009F  # candidate: vertical speed

N_ENEMY_SLOTS = 5
ADDR_ENEMY_DRAWN = 0x000F
ADDR_ENEMY_TYPE = (
    0x0016  # candidate: enemy type per slot (0=Goomba, 1=green Koopa, ...)
)
ADDR_ENEMY_X_SCREEN = 0x0087
ADDR_ENEMY_Y_POS = 0x00CF

TILE_BUFFER_BASE = 0x0500
TILE_ROWS = 13
TILE_COLS_PER_PAGE = 16


def get_tile(
    ram: np.ndarray, mario_world_x: int, mario_y: int, dx: int, dy: int
) -> int:
    """Returns 1 if the tile at (mario_world_x+dx, mario_y+dy) is solid, 0 otherwise."""
    x = mario_world_x + dx
    y = mario_y + dy - 16  # empirical vertical offset used in reference scripts

    page = (x // 256) % 2
    col = (x % 256) // 16
    row = (y - 32) // 16

    if row < 0 or row >= TILE_ROWS:
        return 0

    addr = (
        TILE_BUFFER_BASE
        + page * TILE_ROWS * TILE_COLS_PER_PAGE
        + row * TILE_COLS_PER_PAGE
        + col
    )
    if addr < 0 or addr >= len(ram):
        return 0

    return 1 if ram[addr] != 0 else 0


def print_tile_grid(ram: np.ndarray, mario_world_x: int, mario_y: int):
    """Prints a 10-columns x 8-rows ASCII grid around Mario."""
    print("Tile grid (X = solid, . = empty, M = Mario):")
    for row_offset in range(-3, 5):  # rows above/below
        line = ""
        for col_offset in range(-2, 8):  # columns behind/ahead
            if row_offset == 0 and col_offset == 0:
                line += "M"
                continue
            dx = col_offset * 16
            dy = row_offset * 16
            solid = get_tile(ram, mario_world_x, mario_y, dx, dy)
            line += "X" if solid else "."
        print(line)
    print()


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
    env.render()  # forces the pyglet window to be created, so we can hook key events

    speed_state = {"multiplier": 1.0, "paused": False}

    def on_key_press(symbol, modifiers):
        if symbol in (
            pyglet.window.key.PLUS,
            pyglet.window.key.EQUAL,
            pyglet.window.key.NUM_ADD,
        ):
            speed_state["multiplier"] = min(speed_state["multiplier"] * 1.5, 16.0)
            print(f"Speed: {speed_state['multiplier']:.2f}x")
        elif symbol in (pyglet.window.key.MINUS, pyglet.window.key.NUM_SUBTRACT):
            speed_state["multiplier"] = max(speed_state["multiplier"] / 1.5, 0.1)
            print(f"Speed: {speed_state['multiplier']:.2f}x")
        elif symbol == pyglet.window.key._0:
            speed_state["multiplier"] = 1.0
            print("Speed reset to 1.00x")
        elif symbol == pyglet.window.key.SPACE:
            speed_state["paused"] = not speed_state["paused"]
            print("PAUSED" if speed_state["paused"] else "RESUMED")

    env.viewer.window.push_handlers(on_key_press=on_key_press)

    print("Tile grid and speed validation.")
    print(
        "Game window controls: '+' speeds up, '-' slows down, '0' resets to 1x, SPACE pauses/resumes."
    )
    print("Press Ctrl+C in the terminal to stop once you've seen enough.\n")

    frame_time = 1.0 / 60.0
    print_every = 30
    step = 0

    # Action: constant right + periodic jump, to clear obstacles
    # and get far enough to encounter different enemies (useful only for validation)
    action = np.zeros(9, dtype=np.int8)
    action[7] = 1  # RIGHT (index confirmed earlier)

    try:
        while True:
            step_start = time.time()

            if speed_state["paused"]:
                env.render()  # keeps the window responsive to keys while paused
                time.sleep(0.05)
                continue

            # Periodic jump (index 8 = A, candidate) to clear obstacles
            action[8] = 1 if (step % 90) < 15 else 0

            obs, reward, terminated, truncated, info = env.step(action)
            ram = env.get_ram()

            mario_x_screen = int(ram[ADDR_X_SCREEN])
            mario_x_page = int(ram[ADDR_X_PAGE])
            mario_world_x = mario_x_page * 256 + mario_x_screen
            mario_y = int(ram[ADDR_Y_POS])

            x_speed_raw = int(np.int8(ram[ADDR_X_SPEED]))  # read as signed
            y_speed_raw = int(np.int8(ram[ADDR_Y_SPEED]))

            if step % print_every == 0:
                print(
                    f"[step {step:4d}] world_x={mario_world_x} y={mario_y} "
                    f"x_speed={x_speed_raw} y_speed={y_speed_raw}"
                )

                enemy_info = []
                for i in range(N_ENEMY_SLOTS):
                    if int(ram[ADDR_ENEMY_DRAWN + i]):
                        etype = int(ram[ADDR_ENEMY_TYPE + i])
                        ex = int(ram[ADDR_ENEMY_X_SCREEN + i])
                        enemy_info.append(f"slot{i}:type={etype},x={ex}")
                if enemy_info:
                    print("  Active enemies: " + " | ".join(enemy_info))

                print_tile_grid(ram, mario_world_x, mario_y)

            env.render()

            elapsed = time.time() - step_start
            target_frame_time = frame_time / speed_state["multiplier"]
            if elapsed < target_frame_time:
                time.sleep(target_frame_time - elapsed)

            if terminated or truncated:
                print("Episode ended, resetting.")
                obs, info = env.reset()

            step += 1

    except KeyboardInterrupt:
        pass

    try:
        env.close()
    except AttributeError:
        pass

    print("\nValidation interrupted by user.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validates tile grid and X/Y speed RAM addresses."
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
