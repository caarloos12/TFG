import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import pandas as pd

# --- 1. El Entorno Modificado del TSP ---
# Ahora la recompensa solo se da al final del episodio.
class TSPEnvWithTourReward(gym.Env):
    """
    Entorno del Problema del Viajante (TSP) para Aprendizaje por Refuerzo.

    Estado:
        - Posición actual del agente.
        - Máscara binaria de ciudades visitadas.
    Acción:
        - Elegir la siguiente ciudad a visitar (de las no visitadas).
    Recompensa:
        - 0 en todos los pasos intermedios.
        - -longitud_total_del_tour al finalizar el episodio.
    """
    def __init__(self, num_cities=10):
        super().__init__()
        self.num_cities = num_cities
        
        # Generar coordenadas aleatorias para las ciudades
        self.cities = np.random.rand(num_cities, 2)
        self.dist_matrix = self._calculate_distance_matrix()

        self.action_space = gym.spaces.Discrete(num_cities)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(self.num_cities + 1,), dtype=np.float32
        )

    def _calculate_distance_matrix(self):
        matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(i + 1, self.num_cities):
                dist = np.linalg.norm(self.cities[i] - self.cities[j])
                matrix[i, j] = matrix[j, i] = dist
        return matrix

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_city = 0
        self.visited = np.zeros(self.num_cities, dtype=bool)
        self.visited[self.current_city] = True
        self.num_visited = 1
        self.tour = [self.current_city]
        
        return self._get_state(), {}

    def step(self, action):
        next_city = action
        
        if self.visited[next_city]:
            return self._get_state(), -100.0, True, False, {}

        # Actualizar estado
        self.current_city = next_city
        self.visited[self.current_city] = True
        self.num_visited += 1
        self.tour.append(self.current_city)
        
        done = self.num_visited == self.num_cities
        
        # **CÁLCULO DE LA RECOMPENSA BASADA EN EL TOUR COMPLETO**
        if done:
            # Calcular la longitud total del tour
            tour_length = 0
            for i in range(self.num_cities):
                start_node = self.tour[i]
                end_node = self.tour[(i + 1) % self.num_cities] # El último vuelve al primero
                tour_length += self.dist_matrix[start_node, end_node]
            
            # La recompensa es el negativo de la longitud total
            reward = -tour_length
        else:
            # Recompensa intermedia es 0
            reward = 0

        return self._get_state(), reward, done, False, {}

    def _get_state(self):
        state = np.zeros(self.num_cities + 1, dtype=np.float32)
        state[0] = self.current_city / self.num_cities
        state[1:] = self.visited.astype(np.float32)
        return state

# --- 2. La Red Neuronal (Política) ---
# No necesita cambios. Su función sigue siendo la misma.
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, state, visited_mask):
        probs = self.network(state)
        masked_probs = probs * (1 - visited_mask)
        if masked_probs.sum() > 0:
            masked_probs = masked_probs / masked_probs.sum()
        else:
            valid_actions = (1 - visited_mask)
            masked_probs = valid_actions / valid_actions.sum()
        return masked_probs

# La función train completa con la sección de evaluación corregida

def train(num_episodes=5000, num_cities=10, learning_rate=0.0005, gamma=1.0):
    # Nota: gamma se suele poner a 1.0 en problemas episódicos con recompensa final,
    # ya que no queremos descontar la única señal de recompensa que recibimos.
    
    # Inicialización
    env = TSPEnvWithTourReward(num_cities=num_cities)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    policy_net = PolicyNetwork(state_size, action_size)
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    
    # Listas para guardar datos para las gráficas
    episode_lengths = []
    episode_losses = []
    
    # Guardar el mejor tour encontrado
    best_tour_length = float('inf')
    best_tour = None
    
    print(f"Iniciando entrenamiento para TSP con {num_cities} ciudades...")

    for episode in range(num_episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        done = False

        # Jugar un episodio completo
        while not done:
            visited_mask = torch.FloatTensor(state[1:])
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = policy_net(state_tensor, visited_mask)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            state, reward, done, _, _ = env.step(action.item())
            rewards.append(reward)

        # Calcular los retornos. Como gamma=1 y la recompensa solo está al final,
        # el retorno para cada paso es simplemente la recompensa final.
        final_reward = rewards[-1]
        returns = [final_reward] * len(log_probs)
        
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Calcular la pérdida (loss)
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        # Actualizar la red
        optimizer.zero_grad()
        loss = torch.cat(policy_loss).sum()
        loss.backward()
        optimizer.step()

        # Guardar datos para las gráficas
        tour_length = -final_reward if final_reward < -1 else float('nan')
        episode_lengths.append(tour_length)
        episode_losses.append(loss.item())

        # Actualizar el mejor tour encontrado
        if not np.isnan(tour_length) and tour_length < best_tour_length:
            best_tour_length = tour_length
            best_tour = env.tour
        
        if (episode + 1) % 100 == 0:
            tour_length = -final_reward if final_reward < 0 else float('inf')
            print(f"Episodio {episode+1}/{num_episodes} | Longitud del tour: {tour_length:.2f}")

    print("Entrenamiento finalizado.")
    
    print("\nEvaluando el agente entrenado...")
    state, _ = env.reset()
    done = False
    # La lista 'tour' se obtiene directamente del entorno al final.
    
    with torch.no_grad():
        while not done:
            visited_mask = torch.FloatTensor(state[1:])
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = policy_net(state_tensor, visited_mask)
            action = torch.argmax(action_probs).item()
            state, _, done, _, _ = env.step(action)

    print(f"Mejor longitud de tour encontrada: {best_tour_length:.2f}")

    # --- 4. GENERACIÓN DE GRÁFICAS ---
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Gráfica 1: Longitud del Tour
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    lengths_series = pd.Series(episode_lengths)
    moving_avg_len = lengths_series.ffill().rolling(window=100).mean()
    ax1.plot(lengths_series.index, lengths_series.values, label='Longitud por Episodio', color='lightgray', alpha=0.6)
    ax1.plot(moving_avg_len.index, moving_avg_len.values, label='Media Móvil (100 ep.)', color='C0', linewidth=2)
    ax1.set_title('Evolución del Rendimiento del Agente', fontsize=16)
    ax1.set_xlabel('Episodio', fontsize=12)
    ax1.set_ylabel('Longitud del Tour', fontsize=12)
    ax1.legend(fontsize=12)
    fig1.tight_layout()
    plt.show()

    # Gráfica 2: Función de Pérdida (Loss)
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    loss_series = pd.Series(episode_losses)
    moving_avg_loss = loss_series.ffill().rolling(window=100).mean()
    ax2.plot(loss_series.index, loss_series.values, label='Loss por Episodio', color='lightgray', alpha=0.6)
    ax2.plot(moving_avg_loss.index, moving_avg_loss.values, label='Media Móvil (100 ep.)', color='C2', linewidth=2)
    ax2.set_title('Evolución de la Función de Pérdida (Loss)', fontsize=16)
    ax2.set_xlabel('Episodio', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=12)
    fig2.tight_layout()
    plt.show()

    # Gráfica 3: Mejor Tour Encontrado
    if best_tour:
        fig3, ax3 = plt.subplots(figsize=(8, 8))
        # Dibujar las ciudades
        ax3.scatter(env.cities[:, 0], env.cities[:, 1], c='C1', s=100, zorder=3)
        # Anotar cada ciudad con su número
        for i, city in enumerate(env.cities):
            ax3.text(city[0] + 0.01, city[1] + 0.01, str(i), fontsize=12)
        
        # Dibujar el camino del tour
        tour_path = np.array([env.cities[i] for i in best_tour + [best_tour[0]]])
        ax3.plot(tour_path[:, 0], tour_path[:, 1], 'C0-', linewidth=2, zorder=2)
        
        ax3.set_title(f'Mejor Tour Encontrado (Longitud: {best_tour_length:.2f})', fontsize=16)
        ax3.set_xticks([])
        ax3.set_yticks([])
        fig3.tight_layout()
        plt.show()

# --- Ejecutar el Entrenamiento ---
if __name__ == "__main__":
    # Puede que necesite más episodios o un ajuste del learning rate
    # porque la señal de recompensa es mucho más esporádica.
    train(num_episodes=10000, num_cities=50, learning_rate=0.0003)