"""
Validates:
- the level/area type (candidate address 0x0764: typically 0=water, 1=ground,
  2=underground, 3=castle in the classic SMB disassembly)
- a "control locked" state candidate (0x0770), to check during cutscene-like
  moments (entering a pipe, climbing the flagpole) when Mario can't be controlled
- the "Object Pause" candidate (0x0747), reportedly set nonzero the instant
  Mario's death sequence begins (freezing all other on-screen action) — walk
  Mario into an enemy on purpose and watch for the ">>> object_pause CHANGED"
  line to confirm whether this is a reliable death-onset signal
- whether the page-based position (0x006D/0x0086, used for the tile grid) and
  the xscrollHi/xscrollLo-based position (used for fitness) agree with each
  other right at a level transition — walk into a pipe or across a level
  boundary and watch the ">>> LEVEL CHANGED" dump for a mismatch between the
  two, or a tile grid that doesn't match what's on screen

This probe gives you MANUAL keyboard control over Mario, plus quick-load
shortcuts to jump straight into different level types using the savestates
bundled with the stable-retro integration, so you can compare printed values
against what you actually see on screen.

Controls (game window must have focus):
  Arrow keys   - move / duck
  Z            - jump (A)
  X            - run / fireball (B)
  ENTER        - start
  1            - load Level1-1 (normal, ground)
  2            - load Level1-4 (castle)
  3            - load Level2-1 (normal; walk right through the whole level
                 and the pipe at the end to reach 2-2, a water level)
  S            - save the CURRENT emulator state as a custom, reusable
                 stable-retro state (see CUSTOM_SAVE_STATE_NAME below) —
                 useful for training on a specific stretch of the game that
                 has no bundled state of its own (e.g. the start of 1-2)
  +/-          - speed up / slow down emulation
  0            - reset speed to 1x
  SPACE        - pause / resume
  P            - print the current values on demand
"""

import gzip
import os
import time

import numpy as np
import pyglet

import stable_retro

from train_neat import ADDR_X_PAGE, ADDR_X_SCREEN, ADDR_Y_POS, get_tile

ADDR_AREA_TYPE = 0x0764  # candidate, found unreliable: 0x0764 never changed across real level transitions
ADDR_ENGINE_STATE = 0x0770  # candidate: internal game engine state/subroutine (also cited as "Gameplay Mode")
ADDR_OBJECT_PAUSE = 0x0747  # candidate: "Object Pause" — freezes all action except Mario, used upon dying
ADDR_LEVEL_HI = 1887  # validated (from data.json): world index
ADDR_LEVEL_LO = 1884  # validated (from data.json): level-within-world index

# Name used when saving a custom savestate with 'S' (see save_state below).
# stable-retro's SuperMarioBros-Nes-v0 integration only ships states for the
# first level of each world (Level1-1, Level2-1, ...) plus Level1-4 — there's
# no built-in "Level1-2" or similar. This lets you create one: walk manually
# to wherever you want an episode to start, press S, then pass
# `--state Level1-2-custom` to train_neat.py.
CUSTOM_SAVE_STATE_NAME = "Level1-2-custom"

# How many frames to dump in full detail right after a world-level change is
# detected, to check whether the RAM-page-based position (used for the tile
# grid) and the xscrollHi/xscrollLo-based position (used for fitness) agree
# with each other and with what's on screen during the transition.
TRANSITION_DUMP_FRAMES = 90

QUICK_LOAD_STATES = {
    pyglet.window.key._1: "Level1-1",
    pyglet.window.key._2: "Level1-4",
    pyglet.window.key._3: "Level2-1",
}


def save_state(env, name: str) -> str:
    """Saves the emulator's current state as a reusable stable-retro state
    file, in the same folder as the integration's own bundled states (found
    via stable-retro's own lookup, not guessed), so `--state <name>` later
    finds it the same way it finds the built-in ones.

    Raises RuntimeError with a specific, diagnostic message at whichever step
    fails, instead of a bare/opaque exception — this API was verified against
    a freshly downloaded stable-retro wheel, not against whatever version is
    actually installed here, so something not matching is a real possibility."""
    # Any bundled state works as a reference point to find the right folder —
    # Level1-1.state always exists for this integration.
    reference_path = stable_retro.data.get_file_path(
        "SuperMarioBros-Nes-v0", "Level1-1.state"
    )
    if reference_path is None:
        raise RuntimeError(
            "stable_retro.data.get_file_path('SuperMarioBros-Nes-v0', 'Level1-1.state') "
            "returned None — couldn't locate the integration's data folder at all."
        )
    if not os.path.isdir(os.path.dirname(reference_path)):
        raise RuntimeError(
            f"Resolved a target folder that doesn't exist: {os.path.dirname(reference_path)}"
        )
    target_dir = os.path.dirname(reference_path)
    target_path = os.path.join(target_dir, f"{name}.state")

    emulator = getattr(env, "unwrapped", env)
    emulator = getattr(emulator, "em", None)
    if emulator is None:
        raise RuntimeError(
            f"env.unwrapped has no 'em' attribute (type: {type(getattr(env, 'unwrapped', env))}). "
            f"Available attributes: {[a for a in dir(getattr(env, 'unwrapped', env)) if not a.startswith('_')]}"
        )
    if not hasattr(emulator, "get_state"):
        raise RuntimeError(
            f"The emulator object (type: {type(emulator)}) has no get_state() method in this "
            f"stable-retro version. Available methods: "
            f"{[m for m in dir(emulator) if not m.startswith('_')]}"
        )

    raw_state = emulator.get_state()
    if not raw_state:
        raise RuntimeError(
            f"emulator.get_state() returned empty/falsy data: {raw_state!r}"
        )

    try:
        with gzip.open(target_path, "wb") as f:
            f.write(raw_state)
    except OSError as e:
        raise RuntimeError(f"Could not write to {target_path}: {e}")

    if not os.path.exists(target_path) or os.path.getsize(target_path) == 0:
        raise RuntimeError(
            f"Write appeared to succeed but {target_path} is missing or empty."
        )

    return target_path


def print_tile_grid(ram, mario_world_x, mario_y):
    print("  Tile grid (X = solid, . = empty, M = Mario):")
    for row_offset in [-48, -32, -16, 0, 16, 32]:
        line = ""
        for col_offset in [-16, 0, 16, 32, 48, 64, 80, 96, 112, 128]:
            if row_offset == 0 and col_offset == 0:
                line += "M"
            else:
                line += (
                    "X"
                    if get_tile(ram, mario_world_x, mario_y, col_offset, row_offset)
                    else "."
                )
        print(f"    {line}")


def make_env(state=None):
    kwargs = {"render_mode": "human"}
    if state:
        kwargs["state"] = state
    env = stable_retro.make("SuperMarioBros-Nes-v0", **kwargs)
    return env


def main():
    shared = {
        "env": make_env(),
        "action": np.zeros(9, dtype=np.int8),
        "speed_multiplier": 1.0,
        "paused": False,
        "load_request": None,
        "print_now": False,
        "save_request": False,
    }
    shared["env"].reset()
    shared["env"].render()

    def attach_handlers(env):
        def on_key_press(symbol, modifiers):
            a = shared["action"]
            if symbol == pyglet.window.key.RIGHT:
                a[7] = 1
            elif symbol == pyglet.window.key.LEFT:
                a[6] = 1
            elif symbol == pyglet.window.key.UP:
                a[4] = 1
            elif symbol == pyglet.window.key.DOWN:
                a[5] = 1
            elif symbol == pyglet.window.key.Z:
                a[8] = 1  # A / jump
            elif symbol == pyglet.window.key.X:
                a[0] = 1  # B / run-fireball
            elif symbol == pyglet.window.key.ENTER:
                a[3] = 1  # START
            elif symbol == pyglet.window.key.PLUS or symbol == pyglet.window.key.EQUAL:
                shared["speed_multiplier"] = min(shared["speed_multiplier"] * 1.5, 16.0)
                print(f"Speed: {shared['speed_multiplier']:.2f}x")
            elif symbol == pyglet.window.key.MINUS:
                shared["speed_multiplier"] = max(shared["speed_multiplier"] / 1.5, 0.1)
                print(f"Speed: {shared['speed_multiplier']:.2f}x")
            elif symbol == pyglet.window.key._0:
                shared["speed_multiplier"] = 1.0
                print("Speed reset to 1.00x")
            elif symbol == pyglet.window.key.SPACE:
                shared["paused"] = not shared["paused"]
                print("PAUSED" if shared["paused"] else "RESUMED")
            elif symbol == pyglet.window.key.P:
                shared["print_now"] = True
            elif symbol == pyglet.window.key.S:
                shared["save_request"] = True
            elif symbol in QUICK_LOAD_STATES:
                shared["load_request"] = QUICK_LOAD_STATES[symbol]

        def on_key_release(symbol, modifiers):
            a = shared["action"]
            if symbol == pyglet.window.key.RIGHT:
                a[7] = 0
            elif symbol == pyglet.window.key.LEFT:
                a[6] = 0
            elif symbol == pyglet.window.key.UP:
                a[4] = 0
            elif symbol == pyglet.window.key.DOWN:
                a[5] = 0
            elif symbol == pyglet.window.key.Z:
                a[8] = 0
            elif symbol == pyglet.window.key.X:
                a[0] = 0
            elif symbol == pyglet.window.key.ENTER:
                a[3] = 0

        env.viewer.window.push_handlers(
            on_key_press=on_key_press, on_key_release=on_key_release
        )

    attach_handlers(shared["env"])

    print(
        "Manual probe started. See the docstring at the top of this file for controls."
    )
    print(
        "Press 1/2/3 to quick-load Level1-1 / Level1-4 / Level2-1. Press P to print values on demand."
    )
    print(
        f"Press S to save the current position as a reusable state (default name: "
        f"'{CUSTOM_SAVE_STATE_NAME}').\n"
    )

    frame_time = 1.0 / 60.0
    print_every = 60  # once per second at normal speed

    step = 0
    prev_object_pause = None
    prev_world_level = None
    transition_dump_remaining = 0
    while True:
        step_start = time.time()
        env = shared["env"]

        if shared["load_request"] is not None:
            state_name = shared["load_request"]
            shared["load_request"] = None
            print(f"\nLoading state: {state_name}...")
            env.close()
            new_env = make_env(state=state_name)
            new_env.reset()
            new_env.render()
            attach_handlers(new_env)
            shared["env"] = new_env
            shared["action"][:] = 0
            step = 0
            continue

        if shared["save_request"]:
            shared["save_request"] = False
            try:
                saved_path = save_state(env, CUSTOM_SAVE_STATE_NAME)
                print(
                    f"\n>>> Saved current state to: {saved_path} ({os.path.getsize(saved_path)} bytes)"
                )
                print(
                    f">>> Use it with: python train_neat.py --state {CUSTOM_SAVE_STATE_NAME}\n"
                )
            except Exception as e:
                print(f"\n>>> Failed to save state: {e}\n")

        if shared["paused"]:
            env.render()
            time.sleep(0.05)
            continue

        obs, reward, terminated, truncated, info = env.step(shared["action"])
        ram = env.get_ram()

        area_type = int(ram[ADDR_AREA_TYPE])
        engine_state = int(ram[ADDR_ENGINE_STATE])
        object_pause = int(ram[ADDR_OBJECT_PAUSE])
        world = int(np.int8(ram[ADDR_LEVEL_HI])) + 1
        level = int(np.int8(ram[ADDR_LEVEL_LO])) + 1
        world_level = (world, level)

        mario_x_screen = int(ram[ADDR_X_SCREEN])
        mario_x_page = int(ram[ADDR_X_PAGE])
        mario_y = int(ram[ADDR_Y_POS])
        world_x_from_page = mario_x_page * 256 + mario_x_screen
        world_x_from_scroll = info.get("xscrollHi", 0) * 256 + info.get("xscrollLo", 0)

        if prev_world_level is not None and world_level != prev_world_level:
            print(
                f"\n>>> [step {step:5d}] LEVEL CHANGED: {prev_world_level} -> {world_level}. "
                f"Dumping the next {TRANSITION_DUMP_FRAMES} frames in detail:"
            )
            transition_dump_remaining = TRANSITION_DUMP_FRAMES
        prev_world_level = world_level

        if transition_dump_remaining > 0:
            transition_dump_remaining -= 1
            print(
                f"  [step {step:5d}] page-based world_x={world_x_from_page:5d} "
                f"(page={mario_x_page} screen={mario_x_screen})  "
                f"scroll-based world_x={world_x_from_scroll:5d} "
                f"(xscrollHi={info.get('xscrollHi')} xscrollLo={info.get('xscrollLo')})  "
                f"mario_y={mario_y}"
            )
            print_tile_grid(ram, world_x_from_page, mario_y)

        if step % print_every == 0 or shared["print_now"]:
            shared["print_now"] = False
            print(
                f"[step {step:5d}] world-level={world}-{level}  area_type={area_type}  "
                f"engine_state={engine_state}  object_pause={object_pause}  lives={info.get('lives')}"
            )

        if object_pause != prev_object_pause:
            print(
                f"  >>> [step {step:5d}] object_pause CHANGED: {prev_object_pause} -> {object_pause} "
                f"(world-level={world}-{level}, lives={info.get('lives')})"
            )
        prev_object_pause = object_pause

        env.render()

        elapsed = time.time() - step_start
        target_frame_time = frame_time / shared["speed_multiplier"]
        if elapsed < target_frame_time:
            time.sleep(target_frame_time - elapsed)

        if terminated or truncated:
            print("Episode ended, resetting.")
            env.reset()

        step += 1


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProbe interrupted by user.")
