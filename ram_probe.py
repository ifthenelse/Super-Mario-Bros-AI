"""
Verifica gli indirizzi RAM di Super Mario Bros (NES) prima di usarli
come osservazione per NEAT.

Gli indirizzi qui sotto sono quelli comunemente documentati per la ROM NTSC
di Super Mario Bros (vedi Data Crystal RAM map). Con una ROM PAL/Europe
alcuni potrebbero essere leggermente diversi: per questo li validiamo
stampandoli mentre Mario si muove, e controllando che i valori abbiano senso
(es. la posizione X cresce quando Mario va a destra, la Y cambia durante i salti).
"""

import time

import numpy as np

import stable_retro

# Indirizzi candidati (da validare)
ADDR_X_PAGE = 0x006D       # pagina/schermo orizzontale di Mario
ADDR_X_SCREEN = 0x0086     # posizione X di Mario sullo schermo corrente
ADDR_Y_POS = 0x00CE        # posizione Y di Mario
ADDR_PLAYER_STATE = 0x000E  # stato di Mario (es. 0x06/0x0B = sta morendo)

N_ENEMY_SLOTS = 5
ADDR_ENEMY_DRAWN = 0x000F      # 5 byte: 1 se il nemico nello slot è attivo
ADDR_ENEMY_X_SCREEN = 0x0087   # 5 byte: posizione X nemico sullo schermo
ADDR_ENEMY_Y_POS = 0x00CF      # 5 byte: posizione Y nemico


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

    print("Validazione indirizzi RAM. Osserva se i valori hanno senso:")
    print("- mario_x_screen dovrebbe cambiare quando Mario si muove")
    print("- mario_y dovrebbe cambiare durante salti/cadute")
    print("- player_state dovrebbe cambiare se Mario muore\n")

    n_steps = 300
    frame_time = 1.0 / 60.0
    print_every = 30  # stampa ogni mezzo secondo circa

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
                  f"nemici_attivi={len(values['enemies'])} "
                  f"| info_xscrollLo={info.get('xscrollLo')} "
                  f"lives={info.get('lives')}")

        env.render()

        elapsed = time.time() - step_start
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)

        if terminated or truncated:
            print(f"Episodio terminato allo step {step}.")
            obs, info = env.reset()

    try:
        env.close()
    except AttributeError:
        pass

    print("\nValidazione completata.")


if __name__ == "__main__":
    main()
