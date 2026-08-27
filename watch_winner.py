"""
Loads the best genome from a chosen run's winner.pkl and plays it with the
window visible, at real game speed, to see concretely what it learned
(and where it gets stuck).

Which run to load from:
- Pass --run <run_id_or_folder> to load a specific one directly, skipping
  the picker entirely.
- Otherwise, an interactive picker always appears, listing every run that
  has a winner.pkl (including the current directory's, if present). It
  defaults to the current directory's run if there is one, otherwise to
  the most recently started run.

Which state to start from: auto-detected from the chosen run's run_info.json
(the same state it was trained with, if train_neat.py was run with --state) —
watching a genome from the game's normal start when it was actually trained
starting mid-level would put it somewhere it's never seen, and it wouldn't
know what to do. Override with --state if you need to.

On every death, prints the observation the network was actually seeing in
the steps leading up to it (Mario's state, nearby enemies, jump-clearance
estimate, whether the jump button was pressed), to help diagnose whether a
death is caused by bad/missing information or by the network's decision.
"""

import argparse
import glob
import os
import pickle
import sys
import time
from collections import deque
from datetime import datetime, timezone

import neat
import numpy as np
import pyglet

import stable_retro

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover - optional dependency
    try:
        import imageio
    except ImportError:
        imageio = None

from train_neat import (
    Tee,
    build_observation,
    compute_run_arrows,
    debug_snapshot,
    load_run_info,
    load_state_offset,
    outputs_to_action,
    pick_run_interactively,
    summarize_run,
)

HISTORY_BUFFER_SIZE = 200  # rolling window of raw per-frame data we keep around

# Candidate RAM address: "Object Pause". Per SMB1 disassembly documentation,
# this freezes all on-screen action except Mario and is explicitly used upon
# dying — i.e. it goes from 0 to nonzero the instant the death sequence
# begins, which is exactly the "collision, if any" moment we want to trace
# back to (not yet validated on this ROM).
ADDR_OBJECT_PAUSE = 0x0747

# How many frames of context to show *before* the flag is set — i.e. the
# actual moment control was lost (the collision) — not the death animation
# that follows it, which carries no useful information.
CONTEXT_STEPS_TO_SHOW = 40

# Rolling buffer of rendered frames, saved as a video when a death is
# detected. Reading the numeric trace tells you what the network *saw* and
# *decided*; watching the clip tells you what that actually looked like,
# which has repeatedly turned out to be the faster way to spot what's going
# wrong. Kept separate from HISTORY_BUFFER_SIZE since raw RGB frames cost
# far more memory than the small per-frame dicts.
VIDEO_FRAMES_BEFORE_DEATH = 100
VIDEO_FPS = 30


def find_winner_dirs(root_dir: str) -> list:
    """Finds every directory under root_dir that contains a winner.pkl."""
    return sorted(
        {
            os.path.dirname(p)
            for p in glob.glob(
                os.path.join(root_dir, "**", "winner.pkl"), recursive=True
            )
        }
    )


def resolve_winner_path(run_arg: str | None, search_root: str) -> str:
    """Figures out which winner.pkl to load, per the rules in the module
    docstring. Exits with an error message if the request can't be satisfied."""
    if run_arg is not None:
        winner_dirs = find_winner_dirs(search_root)
        match = next((d for d in winner_dirs if os.path.basename(d) == run_arg), None)
        if match is None:
            for d in winner_dirs:
                info = load_run_info(d)
                if info and info.get("run_id") == run_arg:
                    match = d
                    break
        if match is None:
            print(f"Error: no run with a winner.pkl found matching '{run_arg}'.")
            sys.exit(1)
        return os.path.join(match, "winner.pkl")

    winner_dirs = find_winner_dirs(search_root)
    if not winner_dirs:
        print(f"Error: no winner.pkl found anywhere under {search_root}.")
        sys.exit(1)

    runs = [summarize_run(d, search_root) for d in winner_dirs]
    # Most recently started first; runs with no known start time sort last.
    runs.sort(
        key=lambda r: r["start_time"] or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    arrows = compute_run_arrows(runs)

    # Default to the winner.pkl sitting directly in the current directory (the
    # active, not-yet-archived run), if there is one; otherwise the most
    # recently started run (which, now that the list is sorted, is index 0).
    cwd_index = next(
        (
            i
            for i, r in enumerate(runs)
            if os.path.abspath(r["dir"]) == os.path.abspath(search_root)
        ),
        None,
    )
    default_index = cwd_index if cwd_index is not None else 0

    choice = pick_run_interactively(
        runs, arrows, default_index, last_option_label="[Cancel]"
    )
    if choice == len(runs):
        print("Cancelled.")
        sys.exit(0)
    return os.path.join(runs[choice]["dir"], "winner.pkl")


def save_death_video(frames, step: int, out_dir: str) -> str | None:
    """Writes the buffered frames leading up to a death to an mp4. Returns the
    path written, or None if it couldn't be saved (missing dependency, no
    frames, or a write error) — never raises, since a failed recording
    shouldn't interrupt watching the run."""
    if imageio is None:
        print(
            "  (video not saved: imageio isn't installed — "
            "run `pip install 'imageio[ffmpeg]'` to enable death clips)"
        )
        return None
    if not frames:
        return None

    path = os.path.join(out_dir, f"death-{datetime.now():%Y%m%d-%H%M%S}-step{step}.mp4")
    try:
        imageio.mimsave(path, frames, fps=VIDEO_FPS, macro_block_size=1)
    except Exception as e:
        print(f"  (video not saved: {e})")
        return None
    return path


def print_death_trace(trace):
    print("  Steps leading up to this death (most recent last):")
    for step, snap, jump_pressed in trace:
        if snap["enemies"]:
            # Show every active enemy slot, closest first, not just one —
            # the actual colliding enemy might not be the "closest ahead".
            enemies_sorted = sorted(snap["enemies"], key=lambda e: abs(e["dx"]))
            enemy_str = " | ".join(
                f"type={e['type']} dx={e['dx']:+d} dy={e['dy']:+d} "
                f"clearance={e['ceiling_clearance']:.2f} time_to_impact={e['time_to_enemy']:+.2f}"
                for e in enemies_sorted
            )
        else:
            enemy_str = "no enemies in range"
        print(
            f"    [step {step}] y={snap['mario_y']:3d} x_speed={snap['x_speed']:+3d} "
            f"y_speed={snap['y_speed']:+3d} jump_pressed={jump_pressed}  |  {enemy_str}"
        )

    last_snap = trace[-1][1]
    print("  Tile grid at the moment of death (X = solid, . = empty, M = Mario):")
    for line in last_snap["tile_grid"]:
        print(f"    {line}")


def main(run_arg: str | None = None, state_arg: str | None = None):
    local_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(local_dir, "neat-config.txt")

    winner_path = resolve_winner_path(run_arg, os.getcwd())

    # The genome only makes sense in whatever situation it was actually
    # trained in — if it was trained with --state (e.g. starting mid-level,
    # via train_neat.py), watching it from the game's normal default start
    # puts it somewhere it's never seen, and it behaves accordingly (usually
    # not moving at all). An explicit --state always wins; otherwise, use
    # whatever state this run's run_info.json says it was trained with.
    if state_arg is not None:
        state = state_arg
    else:
        run_info = load_run_info(os.path.dirname(winner_path))
        state = run_info.get("state") if run_info else None
    # Distance already covered before this state's starting point (see
    # probe_level_type.py's 'S' key / train_neat.py's --state) — added to
    # every position shown below, so it's on the same scale as a run that
    # started from the very beginning instead of looking artificially low.
    initial_level_offset = load_state_offset(state)

    log_path = os.path.join(os.getcwd(), f"watch-{datetime.now():%Y%m%d-%H%M%S}.log")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file = open(log_path, "w")
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    try:
        print(f"Logging full output to: {log_path}")
        print(f"Loading genome from: {winner_path}")
        if state:
            print(f"Starting from state: {state}")

        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path,
        )

        with open(winner_path, "rb") as f:
            winner = pickle.load(f)

        net = neat.nn.FeedForwardNetwork.create(winner, config)

        env = (
            stable_retro.make("SuperMarioBros-Nes-v0", state=state, render_mode="human")
            if state
            else stable_retro.make("SuperMarioBros-Nes-v0", render_mode="human")
        )
        obs, info = env.reset()
        ram = env.get_ram()
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
            elif symbol in (
                pyglet.window.key.SPACE,
                pyglet.window.key.ENTER,
                pyglet.window.key.RETURN,
            ):
                speed_state["paused"] = not speed_state["paused"]
                print("PAUSED" if speed_state["paused"] else "RESUMED")

        env.viewer.window.push_handlers(on_key_press=on_key_press)

        frame_time = 1.0 / 60.0
        max_world_x = 0
        prev_lives = info.get("lives")
        history = deque(maxlen=HISTORY_BUFFER_SIZE)
        video_frames = deque(maxlen=VIDEO_FRAMES_BEFORE_DEATH)
        state_change_reported = False

        print(f"Loaded genome fitness (from training): {winner.fitness}")
        raw_d = getattr(winner, "raw_distance", None)
        lives = getattr(winner, "lives_lost", None)
        bonus = getattr(winner, "caution_bonus", None)
        idle_bonus = getattr(winner, "anti_idle_bonus", None)
        jwb_bonus = getattr(winner, "jump_when_blocked_bonus", None)
        ng_bonus = getattr(winner, "narrow_gap_bonus", None)
        run_bonus = getattr(winner, "running_bonus", None)
        adv_bonus = getattr(winner, "advance_bonus", None)
        cp_bonus = getattr(winner, "clear_path_bonus", None)
        dh_bonus = getattr(winner, "down_hold_bonus", None)
        if raw_d is not None:
            print(
                f"  (raw distance during training: {raw_d:.0f}, lives lost: {lives}, "
                f"caution bonus: {bonus:.1f}, anti-idle bonus: {idle_bonus:.1f}, "
                f"jump-when-blocked bonus: {jwb_bonus:.1f}, narrow-gap bonus: {ng_bonus:.1f}, "
                f"running bonus: {run_bonus:.1f}, advance bonus: {adv_bonus:.1f}, "
                f"clear-path bonus: {cp_bonus:.1f}, down-hold bonus: {dh_bonus:.1f})"
            )
        print(
            "Controls (game window must have focus): '+' speeds up, '-' slows down, "
            "'0' resets to 1x, SPACE/ENTER pauses/resumes."
        )
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
            # so this frame can later be shown as context leading up to a state change.
            snap = debug_snapshot(ram)
            history.append((step, snap, bool(action[8])))
            # obs is the emulator's RGB frame; buffer it for the death clip.
            video_frames.append(np.asarray(obs, dtype=np.uint8))

            player_state = int(ram[ADDR_OBJECT_PAUSE])
            if player_state == 0:
                state_change_reported = False
            elif not state_change_reported:
                # The instant Object Pause is set (nonzero) — per the docs this
                # happens right when the death sequence begins, freezing
                # everything except Mario's death-jump animation. The
                # collision, if any, is at or right before the last frame
                # shown here, not in this frame or after.
                state_change_reported = True
                context = [e for e in history if e[0] < step][-CONTEXT_STEPS_TO_SHOW:]
                print(
                    f"\n[step {step}] Object Pause flag set (0x0747 = {player_state}) — "
                    f"death sequence likely starting now. Steps leading up to this moment:"
                )
                if context:
                    print_death_trace(context)
                else:
                    print("  (not enough history before this point to show)")
                video_path = save_death_video(list(video_frames), step, os.getcwd())
                if video_path:
                    print(
                        f"  Clip of the {len(video_frames)} frames before this death: {video_path}"
                    )

            obs, reward, terminated, truncated, info = env.step(action)
            ram = env.get_ram()

            world_x = (
                initial_level_offset
                + info.get("xscrollHi", 0) * 256
                + info.get("xscrollLo", 0)
            )
            if world_x > max_world_x:
                max_world_x = world_x

            lives = info.get("lives")
            if lives is not None and prev_lives is not None and lives < prev_lives:
                print(
                    f"[step {step}] Mario lost a life (registered by the game here; "
                    f"the actual collision was reported above, near the start of the fall). "
                    f"Position reached: {world_x} (max so far: {max_world_x})"
                )
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
    parser = argparse.ArgumentParser(
        description="Watch the best genome from a training run play Super Mario Bros."
    )
    parser.add_argument(
        "--run",
        "-r",
        type=str,
        default=None,
        help="Run ID (or folder name) to load winner.pkl from, skipping the interactive picker. "
        "If omitted: uses winner.pkl in the current directory if present, otherwise "
        "shows a picker over every archived run that has one.",
    )
    parser.add_argument(
        "--state",
        "-s",
        type=str,
        default=None,
        help="stable-retro state to start from, overriding whatever this run's run_info.json "
        "says it was trained with (auto-detected by default — you normally don't need this).",
    )
    args = parser.parse_args()
    main(args.run, args.state)
