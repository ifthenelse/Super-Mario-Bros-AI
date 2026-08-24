# Super Mario AI — NEAT on Super Mario Bros (NES)

An AI that learns to play Super Mario Bros (NES) using **NEAT** (NeuroEvolution of Augmenting Topologies), an evolutionary algorithm that evolves both the weights and the topology of neural networks, without needing backpropagation or a labeled dataset.

The project uses [stable-retro](https://github.com/Farama-Foundation/stable-retro) to emulate the ROM and read its state from RAM (Mario's position, enemies, terrain, level type), and [neat-python](https://github.com/CodeReclaimers/neat-python) to evolve the population of networks.

## How it works

Each genome in the population (a candidate neural network) plays one episode of Super Mario Bros — which can span multiple consecutive levels if Mario clears them, since the episode only ends on game over or after getting stuck too long. The observation fed to the network is built by reading the emulator's RAM directly:

- Mario's Y position, velocity (X/Y), and power-up state (small/big/fire)
- The current level's type (normal, water, or castle), derived from the world/level number via a static lookup table
- Presence, relative position, type, jump-over clearance, and estimated time-to-impact for nearby enemies (up to 5 slots)
- A local grid of terrain tiles (solid blocks, pits) around Mario

The network's output is 9 values, one per NES controller button. The **fitness** of each genome is the maximum distance reached, minus a small penalty for every life lost during the episode — so two genomes reaching similar distance aren't treated as equivalent if one got there by recklessly dying repeatedly. Successive generations are created by selecting, mutating, and crossing over the best genomes.

## Requirements

- macOS (tested with pyenv + Python 3.11.11)
- A legally obtained Super Mario Bros ROM in `.nes` format (not included: source it yourself from a physical copy of the game you own)
- ~500 MB of free space for the virtual environment and dependencies

## Installation

1. **Set the project's Python version** (inside the project folder):

   ```bash
   pyenv local 3.11.11
   pyenv rehash
   python --version   # should print Python 3.11.11
   ```

2. **Create and activate the virtualenv**:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install stable-retro neat-python
   ```

4. **Import the ROM**:

   ```bash
   python -m retro.import "/path/to/your/rom.nes"
   ```

   If the import reports `Imported 0 games`, your ROM revision has a different checksum than the one stable-retro expects. In that case:
   - Find the integration folder: `python -c "import stable_retro; print(stable_retro.data.path())"` → look for `SuperMarioBros-Nes-v0`
   - Copy your ROM there, renaming it to `rom.nes`
   - Rename the `rom.sha` file in that same folder (e.g. to `rom.sha.bak`) to disable the checksum check

5. **Verify the installation**:
   ```bash
   python -c "import stable_retro; env = stable_retro.make('SuperMarioBros-Nes-v0'); print('OK'); env.close()"
   ```
   If it prints `OK`, you're good to go.

## Project structure

| File                    | What it does                                                                                                                                                                                                                        |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_env.py`           | Minimal test: opens a window with Mario moving randomly, to verify the ROM and emulator communicate correctly                                                                                                                       |
| `ram_probe.py`          | Validates the basic RAM addresses (position, lives, enemies) by printing them while Mario moves randomly                                                                                                                            |
| `ram_probe_advanced.py` | Validates the terrain tile grid, Mario's speed, and enemy type. Has keyboard controls in the game window: `+`/`-` for emulation speed, `0` to reset it, `SPACE` to pause/resume                                                     |
| `probe_level_type.py`   | Gives you **manual keyboard control** over Mario (arrows, `Z` jump, `X` run/fire), plus quick-load shortcuts (`1`/`2`/`3`) to jump straight into different level types, to validate world/level detection against the on-screen HUD |
| `neat-config.txt`       | NEAT configuration: network input/output size, population size, mutation rates, speciation rules                                                                                                                                    |
| `train_neat.py`         | Training script. See "Training" below for how run selection, time budgets, and logging work                                                                                                                                         |
| `watch_winner.py`       | Loads the best genome from a chosen run and plays it with the window visible, at real game speed, logging a detailed trace of every death                                                                                           |
| `archive_run.py`        | Moves the current run's checkpoints, `winner.pkl`, `run_info.json`, and training log into a new folder (named after the reached fitness), so you can start a fresh run without losing previous results                              |

## Tutorial: running the project step by step

### 1. Verify the environment

```bash
python test_env.py
```

You should see a window with Mario moving randomly for about 10 seconds. If it works, the emulator and ROM are set up correctly.

### 2. (Optional) Validate the RAM reads

If you want to verify or adapt the RAM addresses for a ROM different from yours:

```bash
python ram_probe.py
python ram_probe_advanced.py
python probe_level_type.py
```

Compare the values printed to the console with what you see on screen (position, speed, tile grid, enemy type, world-level).

### 3. Train

```bash
python train_neat.py
```

**Runs and checkpoints.** Every invocation of `train_neat.py` is a "run": it writes a `run_info.json` (run ID, parent run ID, start/end time, best fitness) alongside its checkpoints, and saves a checkpoint every 5 generations. On startup, unless you pass `--run`, a full-screen picker (arrow keys to move, `SPACE`/`ENTER` to select, `ESC` to start fresh) lists every run found under the current directory — including archived ones in `old-run-*` folders — showing each one's fitness and a ▲/=/▼ arrow comparing it to its parent run. It waits 15 seconds for input before auto-selecting a sensible default, then resumes training from there using the **current** `neat-config.txt` (not whatever settings were in effect when that checkpoint was saved).

- Resume a specific run directly, skipping the picker: `python train_neat.py --run run-20260824-162200` (also accepts a folder name like `old-run-3128`)
- Force a fresh start even if runs exist: `python train_neat.py --run none`

**Time budget.** Instead of a fixed number of generations, training runs for a time budget you choose, then stops and saves the best genome found so far.

- Pass it directly: `python train_neat.py --minutes 30` (or `-m 1h30m`, or `-m 2h`)
- Or leave it out: you'll be prompted `The script will run for 60 minutes. Type a value in minutes if you want to change it: ` and have 15 seconds to type a new value before it defaults to 60

To prevent macOS from sleeping during a long run, prefix the command with `caffeinate -i`:

```bash
caffeinate -i python train_neat.py -m 1h
```

**Logging.** Every run writes its full console output (including NEAT's own generation-by-generation stats and any warnings) to a `<run_id>.log` file in the current directory, in addition to showing it live — useful since training logs are often far longer than a terminal's scrollback buffer.

### 4. Watch the result

```bash
python watch_winner.py
```

Uses the same run picker as `train_neat.py` (pass `--run <id_or_folder>` to skip it). Plays the chosen `winner.pkl` in real time, with the same speed/pause controls as the probe scripts (`+`/`-`/`0`/`SPACE`/`ENTER`). On every death it prints a detailed trace of the preceding steps — Mario's position/speed, the nearest enemy's distance and jump-clearance, whether the jump button was pressed, and the terrain tile grid at the moment of death — also saved to a `watch-<timestamp>.log` file.

### 5. Iterate

If training plateaus (the best fitness doesn't improve for many generations):

- Use `watch_winner.py`'s death traces to understand _where_ and _why_ Mario dies — e.g. whether it's a missing/unclear observation, a genuinely impossible jump, or the network simply "freezing" (repeating the exact same output for many consecutive frames)
- If the problem is in the observation, add the missing information to `build_observation()` in `train_neat.py` and update `num_inputs` in `neat-config.txt` accordingly — this requires starting over, since it changes the network's structure
- If the problem is that the population isn't exploring enough (same fitness stuck for dozens of generations even across different lineages), reduce `elitism` / `species_elitism` and increase the mutation rates in `neat-config.txt`. A high `elitism` value can effectively "freeze" the best genome (or the best species) unchanged across generations, which looks like a plateau but is really the elite individual never being allowed to mutate
- If the death traces show reckless or frozen behavior rather than a genuine impossibility, consider adjusting the fitness shaping (`LIFE_LOST_PENALTY`) or the stuck-episode cutoff (`STUCK_STEPS_LIMIT`) in `train_neat.py`, so patience/caution near danger has a real chance to be selected for

Before starting a new run with a changed network structure, archive the current one instead of overwriting it:

```bash
python archive_run.py
python train_neat.py -m 1h
```

## Validated vs. candidate RAM addresses

Not all RAM addresses used in this project carry the same level of confidence:

- **Validated** (confirmed against on-screen behavior or already declared in the official stable-retro integration's `data.json`): Mario's position, speed, lives, enemy presence/position/type, terrain tile grid, world/level number.
- **Candidate, unvalidated** (taken from commonly cited SMB disassembly references, but never confirmed on this specific ROM): Mario's power-up state (`0x0756`). An earlier candidate for "area type" (`0x0764`) was tested and found unreliable (it never changed value across real level transitions), so it was dropped in favor of a static world/level → type lookup table instead.

If a new feature you add to the observation doesn't seem to help (or actively hurts) training, an unvalidated address is a reasonable first suspect.

## Known gotchas

- **Resuming and node-ID collisions**: `train_neat.py` rebuilds the population from a checkpoint using the _current_ config rather than the one frozen inside the checkpoint file (so config changes actually take effect on resume). This required a manual fix to NEAT's internal node-ID counter to avoid collisions between genomes carried over from different lineages — already handled in `restore_checkpoint_with_config()`, but worth knowing about if you see `AssertionError` deep inside `neat/genome.py` after modifying that function.
- **`curses` picker requires a real terminal**: the run picker in `train_neat.py`/`watch_winner.py` uses Python's built-in `curses` module (no install needed on macOS/Linux). It needs an interactive terminal — it won't work if output is redirected to a file or piped.

## Notes

- ROMs are copyrighted material: source them yourself from a legally owned copy of the game, and don't commit them to public repositories.
- Checkpoint files (`neat-checkpoint-*`), `winner.pkl`, and log files are often large — consider excluding them from version control (`.gitignore`) if the repository has a public remote.
