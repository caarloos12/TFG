import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import re

# --- 1. Función para Cargar Datos en formato TSPLIB ---
def load_tsplib_problem(filepath):
    """
    Carga los datos del problema desde un archivo en formato TSPLIB (.tsp).
    Extrae las coordenadas de los nodos.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    coords = []
    in_coord_section = False
    dimension = 0

    for line in lines:
        line = line.strip()
        if line.startswith("DIMENSION"):
            dimension = int(line.split(":")[1])
        elif line.startswith("NODE_COORD_SECTION"):
            in_coord_section = True
            continue
        elif line.startswith("EOF"):
            break
        
        if in_coord_section:
            parts = line.split()
            # Formato: node_id x_coord y_coord
            coords.append([float(parts[1]), float(parts[2])])

    if len(coords) != dimension:
        print(f"Warning: El número de coordenadas ({len(coords)}) no coincide con la dimensión ({dimension}).")

    print(f"Datos TSPLIB cargados desde {filepath}: {len(coords)} ciudades.")
    return np.array(coords)


# --- 2. El Entorno Personalizado del TSP ---
class TSPEnvWithEdgeReward(gym.Env):
    """
    Entorno del TSP que carga coordenadas desde un archivo TSPLIB
    y calcula la matriz de distancias euclidianas.
    """
    def __init__(self, filepath):
        super().__init__()
        
        # Cargar las coordenadas de las ciudades desde el archivo
        self.cities = load_tsplib_problem(filepath)
        self.num_cities = self.cities.shape[0]
        
        # Calcular la matriz de distancias euclidianas a partir de las coordenadas
        self.dist_matrix = self._calculate_distance_matrix()
        
        self.action_space = gym.spaces.Discrete(self.num_cities)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(self.num_cities + 1,), dtype=np.float32
        )

    def _calculate_distance_matrix(self):
        # Calcula la matriz de distancias euclidianas entre todas las ciudades
        matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(i + 1, self.num_cities):
                dist = np.linalg.norm(self.cities[i] - self.cities[j])
                matrix[i, j] = matrix[j, i] = dist
        return matrix

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_city = 0 # Empezar siempre en la ciudad 0
        self.visited = np.zeros(self.num_cities, dtype=bool)
        self.visited[self.current_city] = True
        self.num_visited = 1
        self.tour = [self.current_city]
        return self._get_state(), {}

    def step(self, action):
        next_city = action
        if self.visited[next_city]:
            return self._get_state(), -100.0, True, False, {}

        reward = -self.dist_matrix[self.current_city, next_city]
        
        self.current_city = next_city
        self.visited[self.current_city] = True
        self.num_visited += 1
        self.tour.append(self.current_city)
        
        done = self.num_visited == self.num_cities
        if done:
            reward -= self.dist_matrix[self.current_city, self.tour[0]]

        return self._get_state(), reward, done, False, {}

    def _get_state(self):
        state = np.zeros(self.num_cities + 1, dtype=np.float32)
        state[0] = self.current_city / self.num_cities
        state[1:] = self.visited.astype(np.float32)
        return state

# --- 3. La Red Neuronal (Política) ---
# Aumentamos el tamaño de la red para un problema más complejo
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 256), # Más neuronas
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, state, visited_mask):
        probs = self.network(state)
        masked_probs = probs * (1 - visited_mask)
        if masked_probs.sum() > 1e-8:
            masked_probs = masked_probs / masked_probs.sum()
        else:
            valid_actions = (1 - visited_mask)
            # Evitar división por cero si no quedan acciones válidas
            if valid_actions.sum() > 0:
                masked_probs = valid_actions / valid_actions.sum()
            else:
                masked_probs = probs # fallback
        return masked_probs

# --- 4. Función para Graficar Resultados ---
def plot_results(tour_lengths, losses, cities, final_tour):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))
    fig.tight_layout(pad=5.0)

    # Gráfica 1: Longitud del Tour vs. Episodios
    ax1.plot(tour_lengths, color='dodgerblue')
    ax1.set_title("Evolución de la Longitud del Tour por Episodio", fontsize=16)
    ax1.set_xlabel("Episodio", fontsize=12)
    ax1.set_ylabel("Longitud del Tour", fontsize=12)
    ax1.grid(True)

    # Gráfica 2: Pérdida vs. Episodios
    ax2.plot(losses, color='tomato')
    ax2.set_title("Evolución de la Pérdida (Loss) por Episodio", fontsize=16)
    ax2.set_xlabel("Episodio", fontsize=12)
    ax2.set_ylabel("Pérdida", fontsize=12)
    ax2.grid(True)
    
    # Gráfica 3: Tour Final (reintroducida)
    tour_cities = np.array([cities[i] for i in final_tour + [final_tour[0]]])
    ax3.set_title("Tour Final Encontrado", fontsize=16)
    # Dibuja los puntos de las ciudades
    ax3.scatter(cities[:, 0], cities[:, 1], c='red', zorder=3, s=10) # Puntos más pequeños
    # Dibuja el recorrido
    ax3.plot(tour_cities[:, 0], tour_cities[:, 1], 'o-', zorder=2, color='navy', markersize=4)
    ax3.set_xlabel("Coordenada X", fontsize=12)
    ax3.set_ylabel("Coordenada Y", fontsize=12)
    ax3.grid(True)
    ax3.set_aspect('equal', adjustable='box')
    
    plt.show()

# --- 5. Entrenamiento del Agente con REINFORCE ---
def train(filepath, num_episodes=20000, learning_rate=0.0001, gamma=0.99):
    env = TSPEnvWithEdgeReward(filepath=filepath)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    policy_net = PolicyNetwork(state_size, action_size)
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    
    tour_lengths_history = []
    losses_history = []
    
    print(f"Iniciando entrenamiento para problema en {filepath}...")
    print(f"Problema grande ({action_size} ciudades). El entrenamiento puede ser largo.")

    for episode in range(num_episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        done = False

        while not done:
            visited_mask = torch.FloatTensor(state[1:])
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = policy_net(state_tensor, visited_mask)
            
            dist = Categorical(action_probs)
            action = dist.sample()
            
            log_probs.append(dist.log_prob(action))
            state, reward, done, _, _ = env.step(action.item())
            rewards.append(reward)

        returns = []
        discounted_reward = 0
        for r in reversed(rewards):
            discounted_reward = r + gamma * discounted_reward
            returns.insert(0, discounted_reward)
        
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        optimizer.zero_grad()
        loss = torch.cat(policy_loss).sum()
        loss.backward()
        optimizer.step()
        
        total_length = -sum(rewards)
        tour_lengths_history.append(total_length)
        losses_history.append(loss.item())
        
        if (episode + 1) % 500 == 0:
            print(f"Episodio {episode+1}/{num_episodes} | Longitud del tour: {total_length:.2f} | Loss: {loss.item():.2f}")

    print("Entrenamiento finalizado.")
    
    # Evaluar el agente entrenado
    print("\nEvaluando el agente entrenado...")
    state, _ = env.reset()
    done = False
    final_tour = [env.current_city]
    total_length = 0
    
    with torch.no_grad():
        # Reiniciar el entorno para la evaluación
        state, _ = env.reset()
        final_tour = [env.current_city]
        
        while len(final_tour) < env.num_cities:
            visited_mask = torch.FloatTensor(state[1:])
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = policy_net(state_tensor, visited_mask)
            action = torch.argmax(action_probs).item()
            
            if not env.visited[action]:
                dist = env.dist_matrix[env.current_city, action]
                total_length += dist
                state, _, _, _, _ = env.step(action)
                final_tour.append(action)
            else:
                # Si la red elige una acción inválida, hay que forzar una válida
                action_probs[0, env.visited] = -np.inf
                action = torch.argmax(action_probs).item()
                dist = env.dist_matrix[env.current_city, action]
                total_length += dist
                state, _, _, _, _ = env.step(action)
                final_tour.append(action)


    # Añadir distancia de vuelta al origen
    dist_final = env.dist_matrix[env.current_city, final_tour[0]]
    total_length += dist_final
    
    print(f"\nTour final encontrado (primeras 25 ciudades): {final_tour[:25]}...")
    print(f"Longitud total del tour: {total_length:.2f}")

    plot_results(tour_lengths_history, losses_history, env.cities, final_tour)


# --- Ejecutar el Entrenamiento ---
if __name__ == "__main__":
    problem_file = "rat195.tsp" 
    
    # Crear un archivo de ejemplo para que el script se pueda ejecutar directamente
    file_content = """
NAME : rat195
COMMENT : Rattled grid (Pulleyblank)
TYPE : TSP
DIMENSION : 195
EDGE_WEIGHT_TYPE : EUC_2D
NODE_COORD_SECTION
1 3 12
2 17 12
3 23 9
4 34 11
5 47 11
6 54 12
7 66 16
8 75 7
9 86 6
10 94 8
11 107 9
12 115 14
13 123 15
14 3 32
15 15 32
16 26 34
17 33 34
18 42 34
19 53 25
20 64 32
21 74 32
22 85 34
23 95 28
24 104 25
25 113 31
26 125 34
27 3 48
28 15 46
29 26 50
30 36 54
31 48 50
32 54 46
33 64 54
34 75 44
35 88 49
36 98 50
37 103 54
38 115 47
39 127 49
40 6 75
41 15 75
42 27 73
43 36 73
44 47 68
45 54 72
46 66 68
47 74 67
48 85 65
49 94 74
50 107 65
51 117 65
52 125 68
53 6 84
54 13 95
55 25 94
56 37 84
57 47 87
58 53 95
59 63 86
60 77 93
61 83 89
62 94 95
63 103 92
64 115 95
65 123 93
66 7 114
67 15 111
68 24 112
69 36 108
70 43 112
71 56 105
72 64 112
73 73 112
74 86 107
75 98 108
76 104 113
77 117 115
78 126 109
79 6 127
80 17 125
81 27 134
82 35 126
83 44 131
84 54 132
85 63 124
86 77 127
87 82 134
88 96 128
89 103 126
90 116 130
91 126 134
92 7 152
93 16 147
94 24 153
95 35 151
96 45 154
97 55 146
98 63 155
99 75 151
100 87 154
101 93 156
102 104 151
103 117 153
104 127 148
105 3 164
106 16 172
107 25 165
108 35 175
109 44 169
110 53 174
111 64 168
112 76 171
113 87 173
114 95 174
115 106 168
116 114 169
117 125 169
118 3 190
119 16 188
120 25 195
121 37 186
122 44 189
123 54 194
124 66 192
125 77 192
126 85 188
127 93 185
128 106 192
129 113 193
130 125 195
131 5 207
132 15 213
133 24 209
134 33 214
135 43 206
136 53 211
137 64 213
138 74 212
139 84 212
140 94 209
141 104 215
142 115 206
143 127 209
144 6 229
145 13 227
146 26 235
147 34 225
148 43 227
149 55 225
150 67 229
151 75 234
152 87 230
153 95 235
154 105 228
155 117 225
156 127 230
157 6 249
158 15 246
159 26 255
160 33 246
161 47 248
162 58 252
163 65 248
164 73 247
165 87 249
166 94 245
167 104 256
168 113 246
169 125 253
170 5 266
171 16 274
172 24 267
173 37 266
174 45 267
175 54 266
176 67 267
177 74 265
178 87 264
179 95 271
180 106 264
181 116 271
182 127 273
183 7 287
184 17 294
185 23 287
186 33 284
187 43 288
188 53 295
189 67 288
190 73 286
191 87 293
192 94 284
193 104 291
194 114 294
195 127 290
EOF
"""
    with open(problem_file, "w") as f:
        f.write(file_content)
        
    train(filepath=problem_file)