from collections import defaultdict
import pickle
import random

import numpy as np


class QAgent:
    def __init__(self, game, actions, learning_rate=0.1, discount_factor=0.99,
                    epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01,
                    load_q_table_path=None):
        self.game = game  # game es una instancia de Pong
        self.actions = list(actions)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Acceder directamente a las propiedades del juego
        # Pong.__init__ define self.paddle_width
        self.paddle_width = self.game.paddle_width
        self.game_height = self.game.height # PLE pasa el objeto game directamente
        self.game_width = self.game.width

        if load_q_table_path:
            try:
                with open(load_q_table_path, 'rb') as f:
                    self.q_table = pickle.load(f)
                print(f"Q-table cargado desde {load_q_table_path}")
            except FileNotFoundError:
                print(f"Archivo Q-table no encontrado en {load_q_table_path}. Iniciando uno nuevo.")
                self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
        else:
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))

        # Parámetros de discretización (AJUSTAR ESTOS)
        self.num_bins = {
            'relative_fruit_x': 10,   # Diferencia vertical entre bola y centro de la paleta del jugador
            'player_velocity_sign': 3, # -1 (up), 0 (still), 1 (down)
            'fruit_y': 15,
            'fruit_y_on_player_side': 2, # 0 si la bola está del lado del CPU, 1 si está del lado del jugador
        }
        self.player_v_threshold = 0.5

    def _discretize_state(self, state_dict):
        # state_dict['player_y'] es el centro de la paleta del jugador (de Pong.getGameState)
        player_center_x = state_dict['player_x']

        # 1. Posición relativa de la bola al centro de la paleta del jugador
        relative_fruit_x = state_dict['fruit_x'] - player_center_x
        # Rango de relative_fruit_x es aprox. [-game_height, game_height]
        # Escalamos esto a [0, 1] para los bins, considerando un rango efectivo de
        # [-game_height/2, game_height/2] centrado en 0.
        # (valor + rango_max_positivo) / rango_total
        # Si relative_fruit_x es -H/2 => (-H/2 + H/2) / H = 0
        # Si relative_fruit_x es +H/2 => (H/2 + H/2) / H = 1
        scaled_relative_fruit_x = (relative_fruit_x + self.game_height / 2) / self.game_height
        relative_fruit_x_bin = int(np.clip(scaled_relative_fruit_x * self.num_bins['relative_fruit_x'], 0, self.num_bins['relative_fruit_x'] - 1))

        # 2. Signo de la velocidad del jugador
        if state_dict['player_vel'] < -self.player_v_threshold:
            player_velocity_sign_bin = 0 # Moviéndose arriba
        elif state_dict['player_vel'] > self.player_v_threshold:
            player_velocity_sign_bin = 2 # Moviéndose abajo
        else:
            player_velocity_sign_bin = 1 # Quieto o casi quieto

        # 3. Posición X de la bola
        # Normalizar fruit_y a [0, 1] y luego discretizar
        fruit_y_normalized = state_dict['fruit_y'] / self.game_width
        fruit_y_bin = int(np.clip(fruit_y_normalized * self.num_bins['fruit_y'], 0, self.num_bins['fruit_y'] - 1))

        # 4. ¿Está la bola del lado del jugador?
        # (Consideramos lado del jugador si está en la mitad izquierda Y moviéndose hacia el jugador)
        fruit_on_player_side_bin = 0
        if state_dict['fruit_y'] < self.game_width / 2:
                fruit_on_player_side_bin = 1
        # num_bins['ball_y_on_player_side'] debe ser 2 para esto

        # # 5. Dirección X de la bola
        # ball_vx_sign_bin = 0 if state_dict['ball_velocity_x'] < 0 else 1 # Hacia el jugador / Hacia CPU
        # # num_bins['ball_velocity_x_sign'] debe ser 2

        # # 6. Dirección Y de la bola (más granular)
        # bvy = state_dict['ball_velocity_y']
        # if bvy < -self.ball_vy_threshold_fast: # Arriba muy rápido
        #     ball_vy_sign_bin = 0
        # elif bvy < -self.ball_vy_threshold_slow: # Arriba rápido
        #     ball_vy_sign_bin = 1
        # elif bvy <= self.ball_vy_threshold_slow: # Lento o quieta (incluye 0)
        #     ball_vy_sign_bin = 2
        # elif bvy <= self.ball_vy_threshold_fast: # Abajo rápido
        #     ball_vy_sign_bin = 3
        # else: # Abajo muy rápido
        #     ball_vy_sign_bin = 4
        # # num_bins['ball_velocity_y_sign'] debe ser 5

        return (
            relative_fruit_x_bin,
            player_velocity_sign_bin,
            fruit_y_bin,
            fruit_on_player_side_bin
        )

    def choose_action(self, state_dict):
        discrete_state = self._discretize_state(state_dict)
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            q_values = self.q_table[discrete_state]
            return self.actions[np.argmax(q_values)]

    def update_q_table(self, state_dict, action, reward, next_state_dict, done):
        discrete_state = self._discretize_state(state_dict)
        discrete_next_state = self._discretize_state(next_state_dict)
        action_idx = self.actions.index(action)
        current_q = self.q_table[discrete_state][action_idx]
        max_future_q = 0
        if not done:
            max_future_q = np.max(self.q_table[discrete_next_state])
        new_q = current_q + self.lr * (reward + self.gamma * max_future_q - current_q)
        self.q_table[discrete_state][action_idx] = new_q

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_q_table(self, path):
        with open(path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Q-table guardado en {path}")