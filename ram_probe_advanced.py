"""
Verifica la lettura di:
- griglia di tile locale intorno a Mario (terreno/blocchi solidi, buche)
- velocità X e Y di Mario

La formula per leggere i tile dal buffer di livello (indirizzo base 0x0500)
è quella storicamente usata dagli script Lua di bot per SMB su NES: dato che
lo schermo di gioco è diviso in due "pagine" da 13 righe x 16 colonne di tile,
si calcola in quale pagina/riga/colonna cade un punto (x, y) del mondo e si
legge il byte corrispondente (0 = vuoto, diverso da 0 = solido).

Gli indirizzi di velocità (0x0057 per X, 0x009F per Y) sono candidati
documentati nella disassembly classica di SMB: li validiamo stampandoli
mentre Mario cammina/salta/cade.
"""

import time

import numpy as np

import stable_retro

ADDR_X_SCREEN = 0x0086
ADDR_X_PAGE = 0x006D
ADDR_Y_POS = 0x00CE

ADDR_X_SPEED = 0x0057  # candidato: velocita' orizzontale
ADDR_Y_SPEED = 0x009F  # candidato: velocita' verticale

TILE_BUFFER_BASE = 0x0500
TILE_ROWS = 13
TILE_COLS_PER_PAGE = 16


def get_tile(ram: np.ndarray, mario_world_x: int, mario_y: int, dx: int, dy: int) -> int:
    """Restituisce 1 se il tile in (mario_world_x+dx, mario_y+dy) e' solido, 0 altrimenti."""
    x = mario_world_x + dx
    y = mario_y + dy - 16  # offset verticale empirico usato negli script di riferimento

    page = (x // 256) % 2
    col = (x % 256) // 16
    row = (y - 32) // 16

    if row < 0 or row >= TILE_ROWS:
        return 0

    addr = TILE_BUFFER_BASE + page * TILE_ROWS * TILE_COLS_PER_PAGE + row * TILE_COLS_PER_PAGE + col
    if addr < 0 or addr >= len(ram):
        return 0

    return 1 if ram[addr] != 0 else 0


def print_tile_grid(ram: np.ndarray, mario_world_x: int, mario_y: int):
    """Stampa una griglia ASCII 10 colonne x 8 righe intorno a Mario."""
    print("Griglia tile (X = solido, . = vuoto, M = Mario):")
    for row_offset in range(-3, 5):  # righe sopra/sotto
        line = ""
        for col_offset in range(-2, 8):  # colonne dietro/avanti
            if row_offset == 0 and col_offset == 0:
                line += "M"
                continue
            dx = col_offset * 16
            dy = row_offset * 16
            solid = get_tile(ram, mario_world_x, mario_y, dx, dy)
            line += "X" if solid else "."
        print(line)
    print()


def main():
    env = stable_retro.make("SuperMarioBros-Nes-v0", render_mode="human")
    obs, info = env.reset()

    print("Validazione griglia tile e velocita'.")
    print("Premi Ctrl+C per interrompere quando hai visto abbastanza.\n")

    frame_time = 1.0 / 60.0
    print_every = 30
    step = 0

    # Azione fissa: solo destra, per avere movimento prevedibile da confrontare a schermo
    action = np.zeros(9, dtype=np.int8)
    action[7] = 1  # indice "RIGHT" plausibile in MultiBinary(9); verificare a schermo

    try:
        while True:
            step_start = time.time()

            obs, reward, terminated, truncated, info = env.step(action)
            ram = env.get_ram()

            mario_x_screen = int(ram[ADDR_X_SCREEN])
            mario_x_page = int(ram[ADDR_X_PAGE])
            mario_world_x = mario_x_page * 256 + mario_x_screen
            mario_y = int(ram[ADDR_Y_POS])

            x_speed_raw = int(np.int8(ram[ADDR_X_SPEED]))  # letto come signed
            y_speed_raw = int(np.int8(ram[ADDR_Y_SPEED]))

            if step % print_every == 0:
                print(f"[step {step:4d}] world_x={mario_world_x} y={mario_y} "
                      f"x_speed={x_speed_raw} y_speed={y_speed_raw}")
                print_tile_grid(ram, mario_world_x, mario_y)

            env.render()

            elapsed = time.time() - step_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

            if terminated or truncated:
                print("Episodio terminato, reset.")
                obs, info = env.reset()

            step += 1

    except KeyboardInterrupt:
        pass

    try:
        env.close()
    except AttributeError:
        pass

    print("\nValidazione interrotta dall'utente.")


if __name__ == "__main__":
    main()
