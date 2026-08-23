"""
Archives the current training run (checkpoints + winner.pkl) into a new
folder, so a fresh training run can start from a clean state without
manually moving files around.

The destination folder is named after the best fitness found in the
existing winner.pkl (e.g. "old-run-3112"), matching the naming convention
used so far. If no winner.pkl is present, falls back to a timestamp.
"""

import glob
import os
import pickle
import shutil
import time


def archive_run():
    local_dir = os.getcwd()

    checkpoint_files = glob.glob(os.path.join(local_dir, "neat-checkpoint-*"))
    winner_path = os.path.join(local_dir, "winner.pkl")
    winner_exists = os.path.exists(winner_path)

    if not checkpoint_files and not winner_exists:
        print("Nothing to archive: no checkpoints or winner.pkl found in the current directory.")
        return

    fitness_label = None
    if winner_exists:
        try:
            with open(winner_path, "rb") as f:
                winner = pickle.load(f)
            fitness_label = int(winner.fitness)
        except Exception as e:
            print(f"Could not read fitness from winner.pkl ({e}); falling back to a timestamp.")

    if fitness_label is not None:
        dest_dir = os.path.join(local_dir, f"old-run-{fitness_label}")
    else:
        dest_dir = os.path.join(local_dir, f"old-run-{time.strftime('%Y%m%d-%H%M%S')}")

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

    print(f"Archived {len(moved)} file(s) to {dest_dir}:")
    for name in sorted(moved):
        print(f"  - {name}")


if __name__ == "__main__":
    archive_run()
