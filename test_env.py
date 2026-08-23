"""
Test rapido dell'ambiente stable-retro per Super Mario Bros.
Esegue azioni casuali per qualche centinaio di frame e mostra la finestra
di gioco, per confermare che ROM, core ed emulatore comunichino bene.
"""

import time

import stable_retro


def main():
    env = stable_retro.make("SuperMarioBros-Nes-v0", render_mode="human")

    obs, info = env.reset()
    print("Ambiente creato correttamente.")
    print("Shape osservazione (frame):", obs.shape)
    print("Spazio azioni:", env.action_space)

    total_reward = 0.0
    n_steps = 600  # ~10 secondi a 60fps, se rallentato correttamente
    frame_time = 1.0 / 60.0

    for step in range(n_steps):
        step_start = time.time()

        action = env.action_space.sample()  # azione casuale
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        env.render()

        # Rallenta il loop per andare a ~60fps reali invece che a velocità massima CPU
        elapsed = time.time() - step_start
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)

        if terminated or truncated:
            print(f"Episodio terminato allo step {step}. Reward totale: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0.0

    try:
        env.close()
    except AttributeError:
        # Bug noto di pyglet 1.5.x su macOS (Cocoa) alla chiusura della finestra.
        # Innocuo: la finestra si chiude comunque correttamente.
        pass

    print("Test completato senza errori.")


if __name__ == "__main__":
    main()
