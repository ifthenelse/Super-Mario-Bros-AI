# Super Mario AI — NEAT on Super Mario Bros (NES)

An AI that learns to play Super Mario Bros (NES) using **NEAT** (NeuroEvolution of Augmenting Topologies), an evolutionary algorithm that evolves both the weights and the topology of neural networks, without needing backpropagation or a labeled dataset.

The project uses [stable-retro](https://github.com/Farama-Foundation/stable-retro) to emulate the ROM and read its state from RAM (Mario's position, enemies, terrain, level type), and [neat-python](https://github.com/CodeReclaimers/neat-python) to evolve the population of networks.

## How it works

Each genome in the population (a candidate neural network) plays one episode of Super Mario Bros. The observation fed to the network is built by reading the emulator's RAM directly:

- Mario's Y position, velocity (X/Y), and power-up state (small/big/fire)
- Presence, relative position, and type of nearby enemies (up to 5 slots)
- A local grid of terrain tiles (solid blocks, pits) around Mario
- The current level's type (normal, water, or castle), derived from the world/level number via a static lookup table

The network's output is 9 values, one per NES controller button. The **fitness** of each genome is the maximum distance reached before dying or getting stuck (across the whole episode, which can span multiple consecutive levels if Mario clears them). Successive generations are created by selecting, mutating, and crossing over the best genomes.

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
| `train_neat.py`         | Training script: builds the observation from RAM, evaluates each genome, evolves the population for a chosen amount of time. Automatically resumes from the latest checkpoint if one is present                                     |
| `watch_winner.py`       | Loads the best saved genome (`winner.pkl`) and plays it with the window visible, at real game speed                                                                                                                                 |
| `archive_run.py`        | Moves the current checkpoints and `winner.pkl` into a new folder (named after the reached fitness), so you can start a fresh run without losing previous results                                                                    |

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

### 3. Start training

```bash
python train_neat.py
```

The script evaluates 150 genomes per generation, printing average fitness, best fitness, number of active species, and stagnation stats to the console. Every 5 generations it saves a checkpoint (`neat-checkpoint-N`), which lets you resume training later instead of starting over — just rerun `python train_neat.py` and it will be detected automatically.

**How long it runs for**: instead of a fixed number of generations, the script runs for a time budget you choose, then stops and saves the best genome found so far.

- Pass it directly: `python train_neat.py --minutes 30` (or `-m 1h30m`, or `-m 2h`)
- Or leave it out: it will prompt `The script will run for 60 minutes. Type a value in minutes if you want to change it: ` and wait up to 15 seconds for you to type a new value before defaulting to 60 minutes

To prevent macOS from sleeping during a long run, prefix the command with `caffeinate -i`:

```bash
caffeinate -i python train_neat.py -m 1h
```

### 4. Watch the result

```bash
python watch_winner.py
```

Loads `winner.pkl` (the best genome found) and plays it in real time, printing to the console where and how Mario loses each life — useful for understanding the training's limitations and deciding how to improve the observation, reward, or configuration.

### 5. Iterate

If training plateaus (the best fitness doesn't improve for many generations):

- Check `watch_winner.py` to understand _where_ and _why_ Mario dies
- If the problem is in the observation (e.g. it doesn't "see" a certain obstacle, or a certain game state), add the missing information to `build_observation()` in `train_neat.py` and update `num_inputs` in `neat-config.txt` accordingly — this requires starting over, since it changes the network's structure
- If the problem is that the population isn't exploring enough (same fitness stuck for dozens of generations), reduce `elitism` / `species_elitism` and increase the mutation rates in `neat-config.txt`. A high `elitism` value can effectively "freeze" the best genome unchanged across generations, which looks like a plateau but is really the elite individual never being allowed to mutate

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

## Notes

- ROMs are copyrighted material: source them yourself from a legally owned copy of the game, and don't commit them to public repositories.
- Checkpoint files (`neat-checkpoint-*`) and `winner.pkl` are pickle binaries, often large — consider excluding them from version control (`.gitignore`) if the repository has a public remote.
