"""
Archives the current training run (checkpoints, winner.pkl, run_info.json,
and training log files) into a new folder, so a fresh training run can
start from a clean state without manually moving files around.

The destination folder is named after the run itself, not the archiving
time: "<run_id>-<best_fitness>-<best_raw_distance>", e.g.
"run-20260826-180525-3806-1299" — run_id already encodes the run's actual
start date/time (see generate_run_id() in train_neat.py), so this keeps
runs sorted chronologically by folder name while still showing at a glance
how well each one did. Falls back to reading fitness/distance straight from
winner.pkl if run_info.json is missing or incomplete (e.g. archiving a run
from before these fields existed), and to a fresh timestamp for the run_id
part if even that isn't available.
"""

import glob
import json
import os
import pickle
import shutil
import time


def determine_archive_name(
    winner_path: str, winner_exists: bool, run_info_path: str, run_info_exists: bool
) -> str:
    run_id = None
    best_fitness = None
    best_raw_distance = None

    if run_info_exists:
        try:
            with open(run_info_path) as f:
                info = json.load(f)
            run_id = info.get("run_id")
            best_fitness = info.get("best_fitness")
            best_raw_distance = info.get("best_raw_distance")
        except Exception as e:
            print(
                f"Could not read run_info.json ({e}); some naming details may fall back."
            )

    if (best_fitness is None or best_raw_distance is None) and winner_exists:
        try:
            with open(winner_path, "rb") as f:
                winner = pickle.load(f)
            if best_fitness is None:
                best_fitness = getattr(winner, "fitness", None)
            if best_raw_distance is None:
                best_raw_distance = getattr(winner, "raw_distance", None)
        except Exception as e:
            print(f"Could not read fitness from winner.pkl ({e}).")

    if run_id is None:
        run_id = time.strftime("run-%Y%m%d-%H%M%S")

    parts = [run_id]
    if best_fitness is not None:
        parts.append(str(int(round(best_fitness))))
    if best_raw_distance is not None:
        parts.append(str(int(round(best_raw_distance))))

    return "-".join(parts)


def archive_run():
    local_dir = os.getcwd()

    checkpoint_files = glob.glob(os.path.join(local_dir, "neat-checkpoint-*"))
    winner_path = os.path.join(local_dir, "winner.pkl")
    winner_exists = os.path.exists(winner_path)
    run_info_path = os.path.join(local_dir, "run_info.json")
    run_info_exists = os.path.exists(run_info_path)
    # Training logs from this lineage (possibly more than one, if the run was
    # resumed across several sessions before being archived)
    log_files = glob.glob(os.path.join(local_dir, "run-*.log"))

    if (
        not checkpoint_files
        and not winner_exists
        and not run_info_exists
        and not log_files
    ):
        print(
            "Nothing to archive: no checkpoints, winner.pkl, run_info.json, or run logs found "
            "in the current directory."
        )
        return

    dest_name = determine_archive_name(
        winner_path, winner_exists, run_info_path, run_info_exists
    )
    dest_dir = os.path.join(local_dir, dest_name)

    # Avoid clobbering an existing folder with the same name (e.g. two runs
    # that happened to reach the same fitness)
    suffix = 2
    original_dest_dir = dest_dir
    while os.path.exists(dest_dir):
        dest_dir = f"{original_dest_dir}-{suffix}"
        suffix += 1

    os.makedirs(dest_dir)

    moved = []
    for f in checkpoint_files:
        shutil.move(f, dest_dir)
        moved.append(os.path.basename(f))
    if winner_exists:
        shutil.move(winner_path, dest_dir)
        moved.append("winner.pkl")
    if run_info_exists:
        shutil.move(run_info_path, dest_dir)
        moved.append("run_info.json")
    for f in log_files:
        shutil.move(f, dest_dir)
        moved.append(os.path.basename(f))

    print(f"Archived {len(moved)} file(s) to {dest_dir}:")
    for name in sorted(moved):
        print(f"  - {name}")


if __name__ == "__main__":
    archive_run()
