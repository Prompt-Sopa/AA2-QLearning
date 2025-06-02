from ple.games.pong import Pong
from ple import PLE
import numpy as np
import time
from QAgentPong import QAgent

# --- Configuración del Entorno y Agente ---
# Inicializar el juego
game = Pong(width=200, height=320, MAX_SCORE=5)
env = PLE(game, display_screen=True, fps=30)

# Inicializar el entorno
env.init()

# Obtener acciones posibles
actions = env.getActionSet() # Deberían ser [None, 119 (w), 115 (s)]

# Crear el agente
# Descomenta la línea de load_q_table_path si quieres cargar una tabla pre-entrenada
agent = QAgent(game, actions, epsilon=1.0, min_epsilon=0.05, epsilon_decay=0.999,
               learning_rate=0.2, discount_factor=0.95,
               load_q_table_path="pong_q_table.pkl")

# --- Bucle de Entrenamiento ---
num_episodes = 5000 # Número de episodios para entrenar
max_steps_per_episode = 5000 # Para evitar episodios infinitos si algo va mal
rewards_all_episodes = []

print(f"Acciones disponibles: {actions}")
print(f"Player Height (actual): {game.paddle_height}")
print(f"Game Height: {game.height}, Game Width: {game.width}")


for episode in range(num_episodes):
    env.reset_game()
    state_dict = env.getGameState()
    done = False
    current_episode_reward = 0
    
    # Desactivar la pantalla durante el entrenamiento para acelerar, activarla para ver
    if episode % 1000 == 0: # Mostrar cada 100 episodios
        env.display_screen = True
        print(f"Episodio {episode}, Epsilon: {agent.epsilon:.3f}")
    else:
        env.display_screen = False


    for step in range(max_steps_per_episode):
        action = agent.choose_action(state_dict)
        reward = env.act(action)
        next_state_dict = env.getGameState()
        done = env.game_over()

        agent.update_q_table(state_dict, action, reward, next_state_dict, done)

        state_dict = next_state_dict
        current_episode_reward += reward

        if env.display_screen:
            time.sleep(0.01) # Pequeña pausa para visualización si la pantalla está activa

        if done:
            break
    
    rewards_all_episodes.append(current_episode_reward)
    agent.decay_epsilon()

    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(rewards_all_episodes[-100:])
        print(f"Episodio: {episode+1}/{num_episodes}, Recompensa Promedio (últimos 100): {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
        # Guardar Q-table periódicamente
        agent.save_q_table("pong_q_table.pkl")


print("Entrenamiento completado.")
agent.save_q_table("pong_q_table_final.pkl")

# --- Opcional: Ejecutar el agente entrenado (sin exploración) ---
print("\n--- Ejecutando agente entrenado (modo explotación) ---")
agent.epsilon = 0 # Sin exploración
env.display_screen = True

for episode in range(5): # Probar 5 episodios
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
