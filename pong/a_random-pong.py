from ple.games.pong import Pong
from ple import PLE
import numpy as np
import time

# --- Configuración del Entorno y Agente ---
# Inicializar el juego
game = Pong(width=200, height=320, MAX_SCORE=5) # Podemos pasar parámetros para modificar la dificultad
env = PLE(game, display_screen=True, fps=30) # fps=30 es más normal, display_screen=True para ver


# Inicializar el entorno
env.init()

# Obtener acciones posibles
actions = env.getActionSet() # Deberían ser [None, 119 (w), 115 (s)]

# Agente con acciones aleatorias
while True:
    env.reset_game()
    state_dict = env.getGameState()
    done = False
    total_reward_episode = 0
    print("\n--- Ejecutando agente aleatorio ---")

    while not done:
        action = np.random.choice(actions)  # Elegir una acción aleatoria
        reward = env.act(action)
        state_dict = env.getGameState()
        done = env.game_over()
        total_reward_episode += reward
        time.sleep(0.03)

    print(f"Recompensa episodio aleatorio: {total_reward_episode}")
