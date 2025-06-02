from ple.games.catcher import Catcher
from ple import PLE
import time
from QAgentPong import QAgent

# --- Configuración del Entorno y Agente ---
# Inicializar el juego
game = Catcher(width=200, height=320, init_lives=3) # Podemos pasar parámetros para modificar la dificultad
env = PLE(game, display_screen=True, fps=30) # fps=30 es más normal, display_screen=True para ver

# Inicializar el entorno
env.init()

# Obtener acciones posibles
actions = env.getActionSet() # Deberían ser [None, 119 (w), 115 (s)]

# Crear el agente
# Descomenta la línea de load_q_table_path si quieres cargar una tabla pre-entrenada
agent = QAgent(game, actions, load_q_table_path="catcher_q_table_final.pkl")


print(f"Acciones disponibles: {actions}")
print(f"Player Width (actual): {game.paddle_width}")
print(f"Game Height: {game.height}, Game Width: {game.width}")
print("\n--- Ejecutando agente entrenado (modo explotación) ---")

agent.epsilon = 0 # Sin exploración
env.display_screen = True

for episode in range(1): # Probar 5 episodios
    env.reset_game()
    state_dict = env.getGameState()
    done = False
    total_reward_episode = 0
    print(f"Iniciando episodio de prueba {episode+1}")
    while not done:
        action = agent.choose_action(state_dict) # Con epsilon=0, siempre explotará
        reward = env.act(action)
        state_dict = env.getGameState()
        done = env.game_over()
        total_reward_episode += reward
        time.sleep(0.03) # Más lento para ver bien
    print(f"Recompensa episodio de prueba {episode+1}: {total_reward_episode}")
