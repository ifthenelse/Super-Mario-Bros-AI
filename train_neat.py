"""
NEAT training on Super Mario Bros (stable-retro).

Each genome in the NEAT population plays one episode (until it dies or
gets stuck for too long), guided by an observation built from RAM
(Mario's Y position + info about nearby enemies).
Fitness is the maximum distance reached in the level.
"""

import argparse
import glob
import gzip
import itertools
import json
import os
import pickle
import random
import re
import select
import sys
import time
from datetime import datetime, timezone

import neat
import numpy as np

import stable_retro

DEFAULT_TIME_BUDGET_MINUTES = 60.0
PROMPT_TIMEOUT_SECONDS = 15


class Tee:
    """Writes everything to multiple streams at once (e.g. the real console
    and a log file), so training output stays visible live in the terminal
    while also being saved in full to a file that can be copied/searched
    afterwards without hitting the terminal's scrollback limit."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def parse_duration_to_minutes(raw: str) -> float:
    """Parses a duration string into minutes.

    Accepted formats:
    - a plain number (interpreted as minutes), e.g. "30" or "12.5"
    - "XXhYYm" (hours and minutes), e.g. "1h30m"
    - "XXh" (hours only), e.g. "2h"

    Raises ValueError on any other format.
    """
    value = raw.strip()

    if re.fullmatch(r"\d+(\.\d+)?", value):
        return float(value)

    match = re.fullmatch(r"(\d+)h(\d+)m", value)
    if match:
        hours, minutes = int(match.group(1)), int(match.group(2))
        return hours * 60 + minutes

    match = re.fullmatch(r"(\d+)h", value)
    if match:
        hours = int(match.group(1))
        return hours * 60

    raise ValueError(
        f"Invalid time format: '{raw}'. Use a plain number of minutes (e.g. '30'), "
        f"'XXhYYm' (e.g. '1h30m'), or 'XXh' (e.g. '2h')."
    )


def resolve_time_budget_minutes(cli_value: str | None) -> float:
    """Resolves the time budget for this run: from the CLI argument if given,
    otherwise by prompting the user with a timeout, falling back to the default."""
    if cli_value is not None:
        return parse_duration_to_minutes(cli_value)

    print(f"The script will run for {DEFAULT_TIME_BUDGET_MINUTES:.0f} minutes. "
          f"Type a value in minutes if you want to change it: ", end="", flush=True)

    ready, _, _ = select.select([sys.stdin], [], [], PROMPT_TIMEOUT_SECONDS)
    if not ready:
        print(f"\nNo input received within {PROMPT_TIMEOUT_SECONDS} seconds. "
              f"Using the default: {DEFAULT_TIME_BUDGET_MINUTES:.0f} minutes.")
        return DEFAULT_TIME_BUDGET_MINUTES

    line = sys.stdin.readline().strip()
    if line == "":
        print(f"Using the default: {DEFAULT_TIME_BUDGET_MINUTES:.0f} minutes.")
        return DEFAULT_TIME_BUDGET_MINUTES

    return parse_duration_to_minutes(line)

# --- RAM addresses validated with ram_probe.py / ram_probe_advanced.py ---
ADDR_X_SCREEN = 0x0086
ADDR_X_PAGE = 0x006D
ADDR_Y_POS = 0x00CE
ADDR_X_SPEED = 0x0057
ADDR_Y_SPEED = 0x009F

N_ENEMY_SLOTS = 5
ADDR_ENEMY_DRAWN = 0x000F
ADDR_ENEMY_TYPE = 0x0016   # validated: enemy type per slot (e.g. 6 = Koopa Troopa)
ADDR_ENEMY_X_SCREEN = 0x0087
ADDR_ENEMY_Y_POS = 0x00CF

ADDR_POWER_STATE = 0x0756  # candidate, NOT yet validated on this ROM: 0=small, 1=big, 2=fire

ADDR_LEVEL_HI = 1887  # validated (from data.json): world index (0-based)
ADDR_LEVEL_LO = 1884  # validated (from data.json): level-within-world index (0-based)

# Static world-level -> type lookup, based on the well-known original SMB1 level design:
# castles are always the 4th level of each world, water levels are 2-2 and 7-2,
# everything else (including underground levels, which share normal run/jump physics) is "normal".
WATER_LEVELS = {(2, 2), (7, 2)}


def get_level_type(world: int, level: int) -> str:
    if level == 4:
        return "castle"
    if (world, level) in WATER_LEVELS:
        return "water"
    return "normal"


def get_level_type_onehot(world: int, level: int) -> tuple:
    level_type = get_level_type(world, level)
    return (
        1.0 if level_type == "normal" else 0.0,
        1.0 if level_type == "water" else 0.0,
        1.0 if level_type == "castle" else 0.0,
    )

TILE_BUFFER_BASE = 0x0500
TILE_ROWS = 13
TILE_COLS_PER_PAGE = 16

# Tile grid: columns ahead of/behind Mario and rows above/below (in pixels, step 16 = 1 tile)
# Extended forward reach (more reaction time) and one extra row up (ceiling awareness for narrow passages)
TILE_COL_OFFSETS = [-16, 0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160]
TILE_ROW_OFFSETS = [-64, -48, -32, -16, 0, 16, 32, 48, 64]

ENEMY_CEILING_CHECK_TILES = 3  # how many tiles above an enemy to check for jump-over clearance

MAX_STEPS_PER_EPISODE = 5000

# Two different "stuck" tolerances, not one: with a single high limit, evolution
# never sees the difference between a genome that's waiting out an unresolved
# threat and one that's just looping in place with nothing blocking it — both
# get cut off identically, so there's no pressure to fix the latter (observed:
# a genome killed nearby enemies, then bounced in place forever with "no
# enemies in range" until the in-game timer ran out — training's cutoff had
# already locked in its fitness long before that, hiding the problem).
STUCK_STEPS_LIMIT_WITH_THREAT = 600    # patience is warranted: an unjumpable enemy is still nearby
STUCK_STEPS_LIMIT_NO_THREAT = 120      # no excuse: nothing nearby justifies not moving

LIFE_LOST_PENALTY = 100  # fitness penalty per life lost during the episode, to discourage reckless deaths

# Reward shaping: since fitness is the *maximum* distance ever reached, briefly
# backing off or waiting near danger already costs nothing on its own — but
# nothing was actively rewarding it either, so evolution never had a direct
# incentive to discover it. This gives a small continuous bonus for holding
# still or retreating specifically when very close to an enemy that can't be
# safely jumped over (clearance=0), instead of freezing in place or pushing
# forward into it.
CAUTION_DANGER_DX = 30       # pixels: how close counts as "immediate danger"
CAUTION_BONUS_PER_FRAME = 0.5  # fitness bonus per frame of demonstrated caution

# Without a cap, a genome can "camp" indefinitely next to an unjumpable enemy,
# banking bonus every frame right up until the stuck-cutoff ends the episode
# (empirically: exactly this happened — a genome earned 298.5 bonus, i.e. 597
# frames, just under the 600-frame stuck cutoff — while making far less real
# distance progress than genomes that got zero bonus). Capping it removes the
# incentive to camp instead of actually resolving the situation and moving on.
MAX_CAUTION_BONUS_PER_EPISODE = 30  # equivalent to 60 frames of caution, at most

# General, cause-agnostic anti-idleness incentive. Repeatedly observed: once a
# genome hits something it's never specifically learned to handle (a jump it's
# never attempted, a gap shaped differently from what it's seen), it just goes
# fully static — same held button state for hundreds of frames — until the
# stuck-cutoff or the in-game clock ends the episode. Training's cutoff treats
# "froze completely" and "tried something and failed" identically (episode
# just ends either way), so there was never any pressure toward the general
# instinct of "keep trying different things when stuck", regardless of *why*
# it's stuck. This rewards trying a genuinely new action combination during a
# stall, once per distinct action, so idle experimentation beats pure freezing
# without being farmable by simply toggling between two states repeatedly.
IDLE_THRESHOLD_FRAMES = 60          # how long without progress before "try something new" is rewarded
ANTI_IDLE_BONUS_PER_NEW_ACTION = 0.3
MAX_ANTI_IDLE_BONUS_PER_EPISODE = 15  # capped so exploring is worth less than actually resolving the stall

# Targeted incentive for a specific, repeatedly observed failure: a solid
# block directly ahead at ground level, with clear room directly above it to
# jump over — and the network just keeps walking into it instead, resetting
# to a stop each time, for hundreds of frames, never once pressing jump. The
# general anti-idleness bonus above only rewards trying each new *button
# combination* once, so it fades out quickly without necessarily reinforcing
# the specific, broadly useful skill of "jump when blocked but clear above" —
# this rewards that exact situation directly, every frame it's handled
# correctly, capped so it can never be milked by holding jump indiscriminately
# (it only fires when genuinely blocked with room to clear it).
JUMP_WHEN_BLOCKED_BONUS_PER_FRAME = 0.3
MAX_JUMP_WHEN_BLOCKED_BONUS_PER_EPISODE = 20


def get_tile_absolute(ram: np.ndarray, x: int, y: int) -> int:
    """Returns 1 if the tile at absolute world coordinates (x, y) is solid, 0 otherwise."""
    page = (x // 256) % 2
    col = (x % 256) // 16
    row = (y - 32) // 16

    if row < 0 or row >= TILE_ROWS:
        return 0

    addr = TILE_BUFFER_BASE + page * TILE_ROWS * TILE_COLS_PER_PAGE + row * TILE_COLS_PER_PAGE + col
    if addr < 0 or addr >= len(ram):
        return 0

    return 1 if ram[addr] != 0 else 0


def get_tile(ram: np.ndarray, mario_world_x: int, mario_y: int, dx: int, dy: int) -> int:
    """Returns 1 if the tile at (mario_world_x+dx, mario_y+dy) is solid, 0 otherwise."""
    x = mario_world_x + dx
    y = mario_y + dy - 16  # empirical vertical offset used in reference scripts
    return get_tile_absolute(ram, x, y)


def enemy_ceiling_clearance(ram: np.ndarray, enemy_world_x: int, enemy_y: int,
                             max_check: int = ENEMY_CEILING_CHECK_TILES) -> float:
    """Counts how many empty tiles are directly above an enemy (up to max_check),
    normalized to 0-1. A low value means there's little or no room to jump onto
    the enemy from above; a high value means it's safe to land on top of it."""
    clearance = 0
    for i in range(1, max_check + 1):
        y = enemy_y - 16 * i - 16  # same vertical offset convention as get_tile
        if get_tile_absolute(ram, enemy_world_x, y) == 0:
            clearance += 1
        else:
            break
    return clearance / max_check


def build_observation(ram: np.ndarray) -> list:
    """Builds the input vector for the NEAT network: Y position, speed,
    nearby enemies, and terrain tile grid."""
    mario_x_screen = int(ram[ADDR_X_SCREEN])
    mario_x_page = int(ram[ADDR_X_PAGE])
    mario_world_x = mario_x_page * 256 + mario_x_screen
    mario_y = int(ram[ADDR_Y_POS])

    x_speed = int(np.int8(ram[ADDR_X_SPEED]))
    y_speed = int(np.int8(ram[ADDR_Y_SPEED]))

    obs = [
        mario_y / 240.0,
        x_speed / 30.0,   # rough normalization (observed max speed ~28)
        y_speed / 10.0,   # rough normalization (observed max speed ~5)
        int(ram[ADDR_POWER_STATE]) / 2.0,  # 0=small, 1=big, 2=fire (rough normalization)
    ]

    world = int(np.int8(ram[ADDR_LEVEL_HI])) + 1
    level = int(np.int8(ram[ADDR_LEVEL_LO])) + 1
    obs.extend(get_level_type_onehot(world, level))

    # Enemies: presence, dx, dy, type, ceiling clearance above, estimated time-to-impact
    # (5 slots x 6 values = 30). Slots are filled in order of urgency (soonest
    # potential impact first, regardless of ahead/behind), not by their raw
    # in-game slot index or raw distance — an enemy about to hit Mario from
    # behind is a bigger threat than one further away straight ahead, even if
    # it isn't the closest by absolute position.
    active_enemies = []
    for i in range(N_ENEMY_SLOTS):
        drawn = int(ram[ADDR_ENEMY_DRAWN + i])
        if drawn:
            enemy_x = int(ram[ADDR_ENEMY_X_SCREEN + i])
            enemy_y = int(ram[ADDR_ENEMY_Y_POS + i])
            enemy_type = int(ram[ADDR_ENEMY_TYPE + i])
            dx = enemy_x - mario_x_screen
            dy = enemy_y - mario_y
            enemy_world_x = mario_world_x + dx

            clearance = enemy_ceiling_clearance(ram, enemy_world_x, enemy_y)
            # Rough "frames until horizontally aligned" estimate, sign preserved (ahead/behind)
            time_to_enemy = dx / (abs(x_speed) + 1) / 50.0

            active_enemies.append((dx, dy, enemy_type, clearance, time_to_enemy))

    active_enemies.sort(key=lambda e: abs(e[4]))  # soonest time-to-impact first

    for i in range(N_ENEMY_SLOTS):
        if i < len(active_enemies):
            dx, dy, enemy_type, clearance, time_to_enemy = active_enemies[i]
            obs.extend([1.0, dx / 256.0, dy / 240.0, enemy_type / 10.0, clearance, time_to_enemy])
        else:
            obs.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    for row_offset in TILE_ROW_OFFSETS:
        for col_offset in TILE_COL_OFFSETS:
            obs.append(float(get_tile(ram, mario_world_x, mario_y, col_offset, row_offset)))

    return obs


def debug_snapshot(ram: np.ndarray) -> dict:
    """Human-readable version of build_observation(), for diagnosing exactly
    what the network 'saw' at a given moment (e.g. right before a death)."""
    mario_x_screen = int(ram[ADDR_X_SCREEN])
    mario_x_page = int(ram[ADDR_X_PAGE])
    mario_world_x = mario_x_page * 256 + mario_x_screen
    mario_y = int(ram[ADDR_Y_POS])

    x_speed = int(np.int8(ram[ADDR_X_SPEED]))
    y_speed = int(np.int8(ram[ADDR_Y_SPEED]))

    world = int(np.int8(ram[ADDR_LEVEL_HI])) + 1
    level = int(np.int8(ram[ADDR_LEVEL_LO])) + 1

    enemies = []
    for i in range(N_ENEMY_SLOTS):
        if int(ram[ADDR_ENEMY_DRAWN + i]):
            enemy_x = int(ram[ADDR_ENEMY_X_SCREEN + i])
            enemy_y = int(ram[ADDR_ENEMY_Y_POS + i])
            enemy_type = int(ram[ADDR_ENEMY_TYPE + i])
            dx = enemy_x - mario_x_screen
            dy = enemy_y - mario_y
            enemy_world_x = mario_world_x + dx
            clearance = enemy_ceiling_clearance(ram, enemy_world_x, enemy_y)
            time_to_enemy = dx / (abs(x_speed) + 1) / 50.0
            enemies.append({
                "slot": i, "dx": dx, "dy": dy, "type": enemy_type,
                "ceiling_clearance": clearance, "time_to_enemy": time_to_enemy,
            })

    tile_grid_lines = []
    for row_offset in TILE_ROW_OFFSETS:
        line = ""
        for col_offset in TILE_COL_OFFSETS:
            if row_offset == 0 and col_offset == 0:
                line += "M"
            else:
                line += "X" if get_tile(ram, mario_world_x, mario_y, col_offset, row_offset) else "."
        tile_grid_lines.append(line)

    return {
        "mario_y": mario_y,
        "x_speed": x_speed,
        "y_speed": y_speed,
        "power_state": int(ram[ADDR_POWER_STATE]),
        "world_level": f"{world}-{level}",
        "level_type": get_level_type(world, level),
        "enemies": enemies,
        "tile_grid": tile_grid_lines,
    }


def outputs_to_action(outputs) -> np.ndarray:
    """Converts the network's (continuous) outputs into a 0/1 button array."""
    return np.array([1 if o > 0.5 else 0 for o in outputs], dtype=np.int8)


def find_most_urgent_enemy(ram: np.ndarray):
    """Returns (dx, clearance) for whichever active enemy has the soonest
    time-to-impact, or None if no enemies are active. Kept independent from
    build_observation() so its output shape/signature stays stable for other
    callers (e.g. watch_winner.py) — this is purely for reward shaping."""
    mario_x_screen = int(ram[ADDR_X_SCREEN])
    mario_x_page = int(ram[ADDR_X_PAGE])
    mario_world_x = mario_x_page * 256 + mario_x_screen
    mario_y = int(ram[ADDR_Y_POS])
    x_speed = int(np.int8(ram[ADDR_X_SPEED]))

    best = None
    best_abs_time = None
    for i in range(N_ENEMY_SLOTS):
        if int(ram[ADDR_ENEMY_DRAWN + i]):
            enemy_x = int(ram[ADDR_ENEMY_X_SCREEN + i])
            enemy_y = int(ram[ADDR_ENEMY_Y_POS + i])
            dx = enemy_x - mario_x_screen
            dy = enemy_y - mario_y
            enemy_world_x = mario_world_x + dx
            clearance = enemy_ceiling_clearance(ram, enemy_world_x, enemy_y)
            time_to_enemy = dx / (abs(x_speed) + 1) / 50.0
            if best_abs_time is None or abs(time_to_enemy) < best_abs_time:
                best_abs_time = abs(time_to_enemy)
                best = (dx, clearance)
    return best


def eval_genomes(genomes, config, env, render=False):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        obs, info = env.reset()
        ram = env.get_ram()

        max_world_x = 0
        last_progress_step = 0
        prev_lives = info.get("lives")
        lives_lost = 0
        caution_bonus = 0.0
        anti_idle_bonus = 0.0
        jump_when_blocked_bonus = 0.0
        idle_tried_actions = set()

        # xscrollHi/xscrollLo reset to 0 on every level transition (1-1 -> 1-2,
        # etc.) — they're a per-level coordinate, not a running total. Left as
        # raw values, "distance" would drop to ~0 the instant a new level
        # starts and would need to climb all the way back past the previous
        # level's ending position before ever registering as new progress
        # again — during which the stuck-cutoff timer keeps counting the
        # entire time, since it never resets on a level transition. In
        # practice this could make it near-impossible to ever get credit for
        # progress in a level shorter than the previous one's ending x. Fix:
        # accumulate an offset each time a level transition is detected, so
        # world_x is a true running total across the whole episode.
        level_offset = 0
        prev_world_level = (int(np.int8(ram[ADDR_LEVEL_HI])), int(np.int8(ram[ADDR_LEVEL_LO])))
        prev_raw_world_x = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            observation = build_observation(ram)
            outputs = net.activate(observation)
            action = outputs_to_action(outputs)

            urgent = find_most_urgent_enemy(ram)
            if urgent is not None:
                dx, clearance = urgent
                if clearance == 0.0 and abs(dx) < CAUTION_DANGER_DX:
                    # No safe way to jump over this enemy and it's very close:
                    # reward holding still or backing off, instead of freezing
                    # uselessly or pushing forward into it.
                    x_speed_now = int(np.int8(ram[ADDR_X_SPEED]))
                    if x_speed_now <= 0:
                        caution_bonus += CAUTION_BONUS_PER_FRAME

            # Solid block directly ahead at ground level, with clear room right
            # above it to jump over: reward pressing jump here specifically,
            # since the repeatedly observed failure mode is walking straight
            # into exactly this without ever attempting to clear it.
            mario_x_screen = int(ram[ADDR_X_SCREEN])
            mario_x_page = int(ram[ADDR_X_PAGE])
            mario_world_x_now = mario_x_page * 256 + mario_x_screen
            mario_y_now = int(ram[ADDR_Y_POS])
            blocked_ahead = get_tile(ram, mario_world_x_now, mario_y_now, 16, 0) == 1
            room_above = get_tile(ram, mario_world_x_now, mario_y_now, 16, -16) == 0
            if blocked_ahead and room_above and bool(action[8]):
                jump_when_blocked_bonus += JUMP_WHEN_BLOCKED_BONUS_PER_FRAME

            if step - last_progress_step >= IDLE_THRESHOLD_FRAMES:
                # Been stuck a while with no progress, regardless of why:
                # reward trying a genuinely new action combination, once per
                # distinct one, instead of just repeating the same held
                # buttons (or toggling between the same two) indefinitely.
                action_key = tuple(int(a) for a in action)
                if action_key not in idle_tried_actions:
                    idle_tried_actions.add(action_key)
                    anti_idle_bonus += ANTI_IDLE_BONUS_PER_NEW_ACTION

            obs, reward, terminated, truncated, info = env.step(action)
            ram = env.get_ram()

            world_level = (int(np.int8(ram[ADDR_LEVEL_HI])), int(np.int8(ram[ADDR_LEVEL_LO])))
            if world_level != prev_world_level:
                # Credit whatever distance was reached in the level just left,
                # before its coordinate resets to 0 in the new level.
                level_offset += prev_raw_world_x
            prev_world_level = world_level

            raw_world_x = info.get("xscrollHi", 0) * 256 + info.get("xscrollLo", 0)
            prev_raw_world_x = raw_world_x
            world_x = level_offset + raw_world_x
            if world_x > max_world_x:
                max_world_x = world_x
                last_progress_step = step
                idle_tried_actions = set()  # genuine progress: the stall is over

            lives = info.get("lives")
            if lives is not None and prev_lives is not None and lives < prev_lives:
                lives_lost += 1
            prev_lives = lives

            if render:
                env.render()

            if terminated or truncated:
                break

            has_blocking_threat = urgent is not None and urgent[1] == 0.0 and abs(urgent[0]) < CAUTION_DANGER_DX
            effective_stuck_limit = (STUCK_STEPS_LIMIT_WITH_THREAT if has_blocking_threat
                                      else STUCK_STEPS_LIMIT_NO_THREAT)
            if step - last_progress_step > effective_stuck_limit:
                # Mario has been stuck too long given the current situation:
                # no point continuing the episode
                break

        # Distance is still the dominant signal, but each life lost costs a small
        # penalty (two genomes reaching similar distance aren't equivalent if one
        # got there by recklessly dying repeatedly), and time spent cautiously
        # waiting out an unjumpable enemy or experimenting during any other kind
        # of stall earns a small, capped bonus (capped so lingering is never more
        # rewarding than actually resolving the situation and continuing).
        # Clamped at 0 to avoid negative fitness confusing NEAT's internal
        # stagnation/adjusted-fitness math.
        capped_caution_bonus = min(caution_bonus, MAX_CAUTION_BONUS_PER_EPISODE)
        capped_anti_idle_bonus = min(anti_idle_bonus, MAX_ANTI_IDLE_BONUS_PER_EPISODE)
        capped_jump_when_blocked_bonus = min(jump_when_blocked_bonus, MAX_JUMP_WHEN_BLOCKED_BONUS_PER_EPISODE)
        genome.fitness = max(0.0, float(max_world_x) - LIFE_LOST_PENALTY * lives_lost
                              + capped_caution_bonus + capped_anti_idle_bonus + capped_jump_when_blocked_bonus)

        # Kept alongside the composite fitness (not used by NEAT itself) so we
        # can tell, e.g., a genome with real distance progress apart from one
        # that mostly racked up caution-bonus without advancing much further.
        genome.raw_distance = float(max_world_x)
        genome.lives_lost = lives_lost
        genome.caution_bonus = capped_caution_bonus
        genome.anti_idle_bonus = capped_anti_idle_bonus
        genome.jump_when_blocked_bonus = capped_jump_when_blocked_bonus


def find_checkpoints(root_dir: str) -> list:
    """Recursively finds all NEAT checkpoint files under root_dir (including subfolders)."""
    pattern = os.path.join(root_dir, "**", "neat-checkpoint-*")
    return sorted(glob.glob(pattern, recursive=True))


def read_checkpoint_summary(filename: str) -> dict | None:
    """Reads just enough from a checkpoint file to get its best fitness so
    far, without needing a matching neat-config.txt. Returns None if the
    file can't be read (e.g. corrupted or incompatible)."""
    try:
        with gzip.open(filename) as f:
            _generation, _config, population, _species_set, _rndstate = pickle.load(f)
    except Exception as e:
        print(f"Warning: could not read checkpoint '{filename}' ({e}); skipping it.")
        return None

    fitnesses = [g.fitness for g in population.values() if g.fitness is not None]
    return {"best_fitness": max(fitnesses) if fitnesses else None}


RUN_INFO_FILENAME = "run_info.json"


def generate_run_id() -> str:
    return datetime.now().astimezone().strftime("run-%Y%m%d-%H%M%S")


def load_run_info(run_dir: str) -> dict | None:
    path = os.path.join(run_dir, RUN_INFO_FILENAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def write_run_info(run_dir: str, run_id: str, parent_run_id: str | None,
                    start_time: datetime, end_time: datetime | None = None,
                    best_fitness: float | None = None, best_raw_distance: float | None = None,
                    best_caution_bonus: float | None = None, best_anti_idle_bonus: float | None = None,
                    best_jump_when_blocked_bonus: float | None = None):
    info = {
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat() if end_time else None,
        "best_fitness": best_fitness,
        "best_raw_distance": best_raw_distance,
        "best_caution_bonus": best_caution_bonus,
        "best_anti_idle_bonus": best_anti_idle_bonus,
        "best_jump_when_blocked_bonus": best_jump_when_blocked_bonus,
    }
    with open(os.path.join(run_dir, RUN_INFO_FILENAME), "w") as f:
        json.dump(info, f, indent=2)


def find_run_dirs(root_dir: str) -> list:
    """Finds every directory under root_dir (root_dir itself included) that
    contains at least one NEAT checkpoint file — i.e. one training run."""
    dirs = {os.path.dirname(p) for p in find_checkpoints(root_dir)}
    return sorted(dirs)


def summarize_run(run_dir: str, root_dir: str) -> dict:
    """Builds a one-line-worthy summary of a training run: identity,
    parentage, timing, and best fitness reached. Falls back to inferring
    missing info (legacy runs predating run_info.json) from the checkpoint
    files themselves."""
    checkpoint_files = glob.glob(os.path.join(run_dir, "neat-checkpoint-*"))
    latest_checkpoint, latest_gen = None, -1
    for f in checkpoint_files:
        m = re.search(r"neat-checkpoint-(\d+)$", f)
        if m:
            gen = int(m.group(1))
            if gen > latest_gen:
                latest_gen, latest_checkpoint = gen, f

    info = load_run_info(run_dir)

    if info:
        run_id = info["run_id"]
        parent_run_id = info.get("parent_run_id")
        start_time = datetime.fromisoformat(info["start_time"]) if info.get("start_time") else None
        end_time = datetime.fromisoformat(info["end_time"]) if info.get("end_time") else None
        best_fitness = info.get("best_fitness")
    else:
        run_id = "current" if os.path.abspath(run_dir) == os.path.abspath(root_dir) else os.path.basename(run_dir)
        parent_run_id = None
        start_time = None
        end_time = None
        best_fitness = None

    if best_fitness is None and latest_checkpoint:
        checkpoint_summary = read_checkpoint_summary(latest_checkpoint)
        best_fitness = checkpoint_summary["best_fitness"] if checkpoint_summary else None

    if start_time is None and checkpoint_files:
        oldest = min(checkpoint_files, key=os.path.getmtime)
        start_time = datetime.fromtimestamp(os.path.getmtime(oldest)).astimezone()

    if end_time is None and latest_checkpoint:
        end_time = datetime.fromtimestamp(os.path.getmtime(latest_checkpoint)).astimezone()

    return {
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "start_time": start_time,
        "end_time": end_time,
        "best_fitness": best_fitness,
        "resume_checkpoint": latest_checkpoint,
        "generation": latest_gen if latest_gen >= 0 else None,
        "dir": run_dir,
    }


def compute_run_arrows(runs: list) -> dict:
    """Compares each run's best fitness to its parent run's best fitness
    (when the parent is also in the list), returning {run_id: '▲'|'='|'▼'|' '}."""
    by_id = {r["run_id"]: r for r in runs}
    arrows = {}
    for r in runs:
        parent = by_id.get(r["parent_run_id"]) if r["parent_run_id"] else None
        if parent is None or parent["best_fitness"] is None or r["best_fitness"] is None:
            arrows[r["run_id"]] = " "
        elif r["best_fitness"] > parent["best_fitness"]:
            arrows[r["run_id"]] = "▲"
        elif r["best_fitness"] < parent["best_fitness"]:
            arrows[r["run_id"]] = "▼"
        else:
            arrows[r["run_id"]] = "="
    return arrows


def format_duration(start: datetime | None, end: datetime | None) -> str:
    if start is None or end is None:
        return "n/a"
    seconds = int((end - start).total_seconds())
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def format_run_line(r: dict, arrow: str) -> str:
    fitness_str = f"{r['best_fitness']:.0f}" if r["best_fitness"] is not None else "n/a"
    start_str = r["start_time"].strftime("%Y-%m-%d %H:%M %Z") if r["start_time"] else "n/a"
    end_str = r["end_time"].strftime("%Y-%m-%d %H:%M %Z") if r["end_time"] else "in progress"
    duration_str = format_duration(r["start_time"], r["end_time"])
    parent_str = r["parent_run_id"] or "-"
    return (f"{arrow} {r['run_id']:<20} parent={parent_str:<20} "
            f"start={start_str:<20} end={end_str:<20} "
            f"dur={duration_str:<8} fitness={fitness_str}")


def pick_run_interactively(runs: list, arrows: dict, default_index: int,
                            last_option_label: str = "[Start a fresh training run]") -> int:
    """Full-screen arrow-key picker (curses). Up/Down to move, SPACE/ENTER to
    select, ESC to pick 'fresh start'. Auto-selects default_index if no key
    is pressed at all within PROMPT_TIMEOUT_SECONDS; once any key is pressed,
    the timeout is cancelled and it waits for an explicit choice."""
    import curses

    def _inner(stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        current = default_index
        fresh_idx = len(runs)
        n_options = len(runs) + 1
        deadline = time.time() + PROMPT_TIMEOUT_SECONDS
        interacted = False

        while True:
            stdscr.erase()
            stdscr.addstr(0, 0, "Select a run to resume from  (up/down: move, SPACE/ENTER: select, ESC: fresh start)")
            if not interacted:
                remaining = max(0.0, deadline - time.time())
                stdscr.addstr(1, 0, f"Auto-selecting the default in {remaining:4.1f}s if no input...")
            for i, r in enumerate(runs):
                marker = "> " if i == current else "  "
                attr = curses.A_REVERSE if i == current else curses.A_NORMAL
                tag = " (default)" if i == default_index else ""
                line = f"{marker}{format_run_line(r, arrows.get(r['run_id'], ' '))}{tag}"
                try:
                    stdscr.addstr(3 + i, 0, line, attr)
                except curses.error:
                    pass
            marker = "> " if current == fresh_idx else "  "
            attr = curses.A_REVERSE if current == fresh_idx else curses.A_NORMAL
            try:
                stdscr.addstr(3 + fresh_idx + 1, 0, f"{marker}{last_option_label}", attr)
            except curses.error:
                pass
            stdscr.refresh()

            if not interacted and time.time() > deadline:
                return default_index

            stdscr.timeout(150)
            key = stdscr.getch()
            if key == -1:
                continue
            interacted = True
            if key in (curses.KEY_UP, ord('k')):
                current = (current - 1) % n_options
            elif key in (curses.KEY_DOWN, ord('j')):
                current = (current + 1) % n_options
            elif key in (10, 13, curses.KEY_ENTER, ord(' ')):
                return current
            elif key == 27:
                return fresh_idx

    return curses.wrapper(_inner)


def restore_checkpoint_with_config(filename: str, config: neat.Config) -> neat.Population:
    """Like neat.Checkpointer.restore_checkpoint, but rebuilds the population
    using the given (freshly loaded) config instead of the one frozen inside
    the checkpoint file. Without this, resuming from a checkpoint silently
    keeps whatever mutation rates, elitism, etc. were in effect when that
    checkpoint was saved, ignoring any changes since made to neat-config.txt."""
    with gzip.open(filename) as f:
        generation, _old_config, population, species_set, rndstate = pickle.load(f)
    random.setstate(rndstate)
    pop = neat.Population(config, (population, species_set, generation))

    # A freshly-built config's node-id counter starts from scratch. Left as-is,
    # it would eventually hand out a node ID that's already in use by some
    # genome carried over from the checkpoint (different lineages can have
    # very different node ID ranges), crashing with an AssertionError deep
    # inside a later mutation. Seed it past the highest node ID already in
    # use anywhere in the restored population to avoid that collision.
    max_node_id = -1
    for genome in population.values():
        if genome.nodes:
            max_node_id = max(max_node_id, max(genome.nodes.keys()))
    if max_node_id >= 0:
        config.genome_config.node_indexer = itertools.count(max_node_id + 1)

    return pop


def run_training(config_path: str, n_generations: int | None = None, time_budget_minutes: float | None = None,
                  checkpoint_prefix: str = "neat-checkpoint-", resume_from: str | None = None):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    run_dir = os.getcwd()
    run_start_time = datetime.now().astimezone()
    run_id = generate_run_id()

    log_path = os.path.join(run_dir, f"{run_id}.log")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file = open(log_path, "w")
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    print(f"Logging full output to: {log_path}")

    try:
        parent_run_id = None
        if resume_from:
            parent_dir = os.path.dirname(os.path.abspath(resume_from)) or run_dir
            parent_info = load_run_info(parent_dir)
            parent_run_id = parent_info["run_id"] if parent_info else os.path.basename(parent_dir)
            print(f"Resuming training from checkpoint: {resume_from} (using the current neat-config.txt)")
            population = restore_checkpoint_with_config(resume_from, config)
        else:
            population = neat.Population(config)

        write_run_info(run_dir, run_id, parent_run_id, run_start_time)

        population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)
        population.add_reporter(neat.Checkpointer(5, filename_prefix=checkpoint_prefix))

        env = stable_retro.make("SuperMarioBros-Nes-v0", render_mode=None)

        def eval_wrapper(genomes, cfg):
            eval_genomes(genomes, cfg, env)

        start_time = time.time()
        winner = None

        try:
            if time_budget_minutes is not None:
                budget_seconds = time_budget_minutes * 60
                gen_count = 0
                while True:
                    elapsed = time.time() - start_time
                    remaining = budget_seconds - elapsed
                    if remaining <= 0:
                        print(f"\nTime budget reached ({time_budget_minutes:.0f} min). Stopping after "
                              f"{gen_count} generation(s) in this run.")
                        break
                    print(f"\n[Time budget: {remaining / 60:.1f} min remaining]")
                    population.run(eval_wrapper, 1)
                    gen_count += 1

                    gen_best = max(population.population.values(),
                                    key=lambda g: g.fitness if g.fitness is not None else -1)
                    raw_d = getattr(gen_best, "raw_distance", None)
                    lives = getattr(gen_best, "lives_lost", None)
                    bonus = getattr(gen_best, "caution_bonus", None)
                    idle_bonus = getattr(gen_best, "anti_idle_bonus", None)
                    jwb_bonus = getattr(gen_best, "jump_when_blocked_bonus", None)
                    breakdown = ""
                    if raw_d is not None:
                        breakdown = (f" (raw_distance={raw_d:.0f} lives_lost={lives} "
                                     f"caution_bonus={bonus:.1f} anti_idle_bonus={idle_bonus:.1f} "
                                     f"jump_when_blocked_bonus={jwb_bonus:.1f})")
                    print(f"  Best this generation: fitness={gen_best.fitness:.1f}{breakdown}")
                winner = stats.best_genome()
            else:
                winner = population.run(eval_wrapper, n_generations)
        finally:
            env.close()
            try:
                best_genome_so_far = stats.best_genome()
                best_so_far = best_genome_so_far.fitness
                best_raw_distance = getattr(best_genome_so_far, "raw_distance", None)
                best_caution_bonus = getattr(best_genome_so_far, "caution_bonus", None)
                best_anti_idle_bonus = getattr(best_genome_so_far, "anti_idle_bonus", None)
                best_jump_when_blocked_bonus = getattr(best_genome_so_far, "jump_when_blocked_bonus", None)
            except Exception:
                best_so_far = None
                best_raw_distance = None
                best_caution_bonus = None
                best_anti_idle_bonus = None
                best_jump_when_blocked_bonus = None
            write_run_info(run_dir, run_id, parent_run_id, run_start_time,
                            end_time=datetime.now().astimezone(), best_fitness=best_so_far,
                            best_raw_distance=best_raw_distance, best_caution_bonus=best_caution_bonus,
                            best_anti_idle_bonus=best_anti_idle_bonus,
                            best_jump_when_blocked_bonus=best_jump_when_blocked_bonus)

        elapsed = time.time() - start_time

        print(f"\nTraining completed in {elapsed / 60:.1f} minutes.")
        print(f"Best genome fitness: {winner.fitness}")
        winner_raw_d = getattr(winner, "raw_distance", None)
        winner_lives = getattr(winner, "lives_lost", None)
        winner_bonus = getattr(winner, "caution_bonus", None)
        winner_idle_bonus = getattr(winner, "anti_idle_bonus", None)
        winner_jwb_bonus = getattr(winner, "jump_when_blocked_bonus", None)
        if winner_raw_d is not None:
            print(f"  (raw distance: {winner_raw_d:.0f}, lives lost: {winner_lives}, "
                  f"caution bonus: {winner_bonus:.1f}, anti-idle bonus: {winner_idle_bonus:.1f}, "
                  f"jump-when-blocked bonus: {winner_jwb_bonus:.1f})")

        with open("winner.pkl", "wb") as f:
            pickle.dump(winner, f)
        print("Best genome saved to winner.pkl")
        print(f"Full log saved to: {log_path}")

        return winner, stats
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEAT training on Super Mario Bros.")
    parser.add_argument(
        "--minutes", "-m",
        type=str,
        default=None,
        help="Max time to run for. Plain number = minutes (e.g. '30'), "
             "or 'XXhYYm' (e.g. '1h30m'), or 'XXh' (e.g. '2h'). "
             "If omitted, you'll be prompted interactively.",
    )
    parser.add_argument(
        "--run", "-r",
        type=str,
        default=None,
        help="Run ID (or folder name) to resume from, skipping the interactive picker. "
             "Use 'none' to force a fresh start even if previous runs exist.",
    )
    args = parser.parse_args()

    try:
        time_budget_minutes = resolve_time_budget_minutes(args.minutes)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    local_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(local_dir, "neat-config.txt")
    search_root = os.getcwd()

    run_dirs = find_run_dirs(search_root)
    runs = [summarize_run(d, search_root) for d in run_dirs]
    # Most recently started first; runs with no known start time sort last.
    runs.sort(key=lambda r: r["start_time"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

    resume_path = None

    if args.run is not None:
        if args.run.lower() == "none":
            print("Starting a fresh training run (forced via --run none).")
        else:
            match = next(
                (r for r in runs if r["run_id"] == args.run or os.path.basename(r["dir"]) == args.run),
                None,
            )
            if match is None:
                print(f"Error: no run found matching '{args.run}'.")
                sys.exit(1)
            resume_path = match["resume_checkpoint"]
            print(f"Using run: {match['run_id']}")
    elif not runs:
        print("No previous runs found: starting fresh.")
    else:
        default_index = 0  # runs are sorted most-recently-started first
        arrows = compute_run_arrows(runs)
        choice = pick_run_interactively(runs, arrows, default_index)
        if choice == len(runs):
            print("\nStarting a fresh training run.")
        else:
            resume_path = runs[choice]["resume_checkpoint"]
            print(f"\nResuming from run: {runs[choice]['run_id']}")

    if resume_path:
        run_training(config_path, time_budget_minutes=time_budget_minutes, resume_from=resume_path)
    else:
        print(f"Running for up to {time_budget_minutes:.1f} minutes.")
        run_training(config_path, time_budget_minutes=time_budget_minutes)
