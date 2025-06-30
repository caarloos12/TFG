"""
==============================================================================
|| TFG - Trabajo de Fin de Grado en Matemáticas de Carlos San Román Cazorla ||
==============================================================================
"""

"Este código ha sido realizado por Carlos San Román Cazorla"
"para el desarrollo de su Trabajo de Fin de Grado en Matemáticas."

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Categorical
import re
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler
import os

# --- Configuración del Modelo y Entrenamiento ---
class Config:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # num_cities se usa solo para el entrenamiento
        self.num_cities = 100 
        self.embedding_dim = 128
        self.hidden_dim = 128
        self.batch_size = 128
        self.train_steps = 2500
        self.learning_rate = 1e-4

# Instanciar configuración
config = Config()

# --- Modelo de Red de Puntero (Pointer Network) ---
class PointerNet(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(PointerNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.encoder_embedding = nn.Linear(2, embedding_dim)
        self.encoder_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.decoder_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.pointer_head = nn.Linear(hidden_dim, hidden_dim)
        self.glimpse_head = nn.Linear(hidden_dim, hidden_dim)
        self.final_head = nn.Linear(hidden_dim, 1)

    def forward(self, cities_data):
        # cities_data: [batch_size, num_cities, 2]
        batch_size, num_cities, _ = cities_data.shape

        # --- Codificador ---
        encoder_embedded = self.encoder_embedding(cities_data)
        # encoder_embedded: [batch_size, num_cities, embedding_dim]
        
        encoder_outputs, (h_n, c_n) = self.encoder_lstm(encoder_embedded)
        # encoder_outputs: [batch_size, num_cities, hidden_dim]

        # --- Decodificador ---
        decoder_hidden = (h_n, c_n)
        decoder_input = torch.zeros(batch_size, 1, self.embedding_dim, device=config.device)
        tours, log_probs = [], []

        mask = torch.zeros(batch_size, num_cities, device=config.device)
        for i in range(num_cities):
            _, decoder_hidden = self.decoder_lstm(decoder_input, decoder_hidden)
            decoder_output = decoder_hidden[0].squeeze(0)

            # --- Mecanismo de Atención (Puntero) ---
            query = self.pointer_head(decoder_output).unsqueeze(1)
            refs = self.glimpse_head(encoder_outputs)
            logits = self.final_head(torch.tanh(refs + query)).squeeze(-1)
            logits = logits - 1e8 * mask
            probs = torch.softmax(logits, dim=1)
            dist = Categorical(probs)
            next_city_idx = dist.sample()

            tours.append(next_city_idx.unsqueeze(1))
            log_probs.append(dist.log_prob(next_city_idx))

            mask = mask.scatter(1, next_city_idx.unsqueeze(1), 1)
            decoder_input = encoder_embedded.gather(1, next_city_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.embedding_dim))
        
        tour_indices = torch.cat(tours, dim=1)
        log_probabilities = torch.stack(log_probs, dim=1).sum(dim=1)
        
        return tour_indices, log_probabilities

# --- Funciones de Utilidad ---

def parse_tsp_instance(file_path):
    """
    Parsea un archivo de instancia TSP, detectando automáticamente el formato.
    Puede manejar 'EUC_2D' (coordenadas) y 'EXPLICIT' (matriz de distancias).
    """
    with open(file_path, 'r') as f:
        content = f.read()

    edge_type_match = re.search(r'EDGE_WEIGHT_TYPE\s*:\s*(\w+)', content)
    if not edge_type_match: raise ValueError("No se pudo encontrar 'EDGE_WEIGHT_TYPE' en el archivo.")
    edge_type = edge_type_match.group(1).strip()

    dim_match = re.search(r'DIMENSION\s*:\s*(\d+)', content)
    if not dim_match: raise ValueError("No se pudo encontrar 'DIMENSION' en el archivo.")
    dimension = int(dim_match.group(1))

    if edge_type == 'EUC_2D':
        coord_section_match = re.search(r'NODE_COORD_SECTION', content)
        if not coord_section_match: raise ValueError("No se pudo encontrar 'NODE_COORD_SECTION' para el tipo EUC_2D.")
        
        data_str = content[coord_section_match.end():].split('EOF')[0]
        lines = data_str.strip().split('\n')
        
        # El bucle procesa correctamente cada línea para extraer las coordenadas.
        coords = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 3:
                coords.append([float(parts[1]), float(parts[2])])
        
        if len(coords) != dimension:
            raise ValueError(f"El número de coordenadas ({len(coords)}) no coincide con la dimensión ({dimension}).")
        
        return torch.from_numpy(np.array(coords)).float(), 'COORDS'

    elif edge_type == 'EXPLICIT':
        edge_weight_section_match = re.search(r'EDGE_WEIGHT_SECTION', content)
        if not edge_weight_section_match: raise ValueError("No se pudo encontrar 'EDGE_WEIGHT_SECTION' para el tipo EXPLICIT.")
        
        data_str = content[edge_weight_section_match.end():].split('EOF')[0]
        flat_weights = [int(n) for n in data_str.split()]
        
        dist_matrix = np.zeros((dimension, dimension))
        k = 0
        for i in range(dimension):
            for j in range(i + 1):
                dist_matrix[i, j] = flat_weights[k]
                dist_matrix[j, i] = flat_weights[k]
                k += 1
        return torch.from_numpy(dist_matrix).float(), 'MATRIX'
    else:
        raise NotImplementedError(f"El tipo de peso de arista '{edge_type}' no está soportado.")


def generate_tsp_data(batch_size, num_cities):
    """Genera coordenadas aleatorias para las ciudades (usado para entrenamiento)."""
    return torch.rand(batch_size, num_cities, 2, device=config.device)

def calculate_tour_length(cities, tour):
    """Calcula la longitud total de una ruta basada en coordenadas euclidianas."""
    ordered_cities = cities.gather(1, tour.unsqueeze(-1).expand(-1, -1, 2))
    rolled = torch.roll(ordered_cities, 1, dims=1)
    distances = torch.sqrt(((ordered_cities - rolled) ** 2).sum(dim=2))
    return distances.sum(dim=1)

def calculate_tour_length_from_matrix(dist_matrix, tour):
    """Calcula la longitud del tour usando la matriz de distancias provista."""
    tour_end = torch.roll(tour, -1, dims=1)
    # dist_matrix es [N, N], tour es [B, T] -> dist_matrix[tour] es [B, T, N]
    gathered_dists = dist_matrix[tour]
    tour_end_indices = tour_end.unsqueeze(2)
    edge_lengths = torch.gather(gathered_dists, 2, tour_end_indices).squeeze(2)
    return edge_lengths.sum(dim=1)

# --- Bucle de Entrenamiento ---
def train():
    """Función para entrenar el modelo. Al finalizar, muestra gráficas de rendimiento."""
    model = PointerNet(config.embedding_dim, config.hidden_dim).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    baseline = torch.zeros(1, device=config.device)
    loss_history, length_history, steps_history = [], [], []

    print("Iniciando entrenamiento...")
    for step in range(config.train_steps):
        model.train()
        cities = generate_tsp_data(config.batch_size, config.num_cities)
        tour, log_prob = model(cities)
        length = calculate_tour_length(cities, tour)
        reward = -length
        
        if baseline.item() == 0: baseline = reward.mean()
        else: baseline = 0.95 * baseline + 0.05 * reward.mean()

        advantage = reward - baseline.detach()
        loss = -(advantage * log_prob).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Paso {step}, Longitud media: {length.mean().item():.3f}, Pérdida: {loss.item():.4f}")
            loss_history.append(loss.item())
            length_history.append(length.mean().item())
            steps_history.append(step)

    print("Entrenamiento finalizado.")
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1); plt.plot(steps_history, loss_history, 'b-o'); plt.title('Pérdida'); plt.grid(True)
    plt.subplot(1, 2, 2); plt.plot(steps_history, length_history, 'r-o'); plt.title('Longitud Media del Tour'); plt.grid(True)
    plt.show()
    return model

# --- Inferencia y Visualización ---
def solve_instance_from_file(model, file_path, config):
    """
    Resuelve una instancia de TSP desde un archivo, usando lógica adaptativa
    y calculando la distancia final con las coordenadas originales.
    """
    print(f"\n--- Resolviendo instancia desde el archivo: {file_path} ---")
    
    try:
        data, data_type = parse_tsp_instance(file_path)
        data = data.to(config.device)
        num_cities = data.shape[0]
        print(f"Instancia '{os.path.basename(file_path)}' (tipo: {data_type}) cargada. Ciudades: {num_cities}")
    except Exception as e:
        print(f"Error al parsear el archivo TSP: {e}"); return
    
    model.eval()
    coords_for_plot = None
    tour_length = 0.0

    if data_type == 'COORDS':
        # --- Flujo para COORDENADAS (CORREGIDO) ---
        # 1. Guardar coordenadas originales para el cálculo final de la distancia.
        original_coords_tensor = data.unsqueeze(0) # Shape: [1, num_cities, 2]

        # 2. Normalizar coordenadas para que el modelo haga una predicción precisa.
        scaler = MinMaxScaler()
        scaled_coords_np = scaler.fit_transform(data.cpu().numpy())
        cities_tensor_for_model = torch.from_numpy(scaled_coords_np).float().unsqueeze(0).to(config.device)
        
        with torch.no_grad():
            tour_indices, _ = model(cities_tensor_for_model)
        
        # 3. ¡IMPORTANTE! Calcular la longitud del tour usando las coordenadas ORIGINALES.
        tour_length = calculate_tour_length(original_coords_tensor, tour_indices).item()
        print(f"Longitud del tour encontrado (Distancia Euclidiana Real): {tour_length:.4f}")

        # 4. Usar las coordenadas normalizadas para una visualización clara.
        coords_for_plot = scaled_coords_np

    elif data_type == 'MATRIX':
        # --- Flujo para MATRIZ DE DISTANCIAS ---
        dist_matrix = data
        print("Generando coordenadas 2D con MDS (esto puede tardar)...")
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_init=10, max_iter=500)
        coords_for_plot_original = mds.fit_transform(dist_matrix.cpu().numpy())
        
        scaler = MinMaxScaler()
        coords_for_plot = scaler.fit_transform(coords_for_plot_original)
        
        cities_tensor = torch.from_numpy(coords_for_plot).float().unsqueeze(0).to(config.device)
        
        with torch.no_grad():
            tour_indices, _ = model(cities_tensor)
        
        tour_length = calculate_tour_length_from_matrix(dist_matrix, tour_indices).item()
        print(f"Longitud del tour encontrado (según matriz original): {tour_length:.4f}")

    # --- Visualizar el resultado ---
    tour_indices_np = tour_indices.squeeze(0).cpu().numpy()
    
    # Crear un mapeo para etiquetar cada nodo con su orden en la ruta
    plot_tour_order = np.zeros(num_cities, dtype=int)
    for i in range(num_cities):
        plot_tour_order[tour_indices_np[i]] = i
        
    ordered_coords = coords_for_plot[tour_indices_np]
    plot_coords = np.vstack([ordered_coords, ordered_coords[0]])
    
    plt.figure(figsize=(10, 10))
    plt.plot(plot_coords[:, 0], plot_coords[:, 1], 'o-', label='Ruta encontrada')
    plt.scatter(coords_for_plot[:, 0], coords_for_plot[:, 1], color='red', s=50, zorder=5)
    for i, (x, y) in enumerate(coords_for_plot):
        plt.text(x, y, f' {plot_tour_order[i]}', fontsize=9)
    
    plt.title(f'Ruta para {os.path.basename(file_path)} ({num_cities} ciudades)\nLongitud Real: {tour_length:.2f}')
    plt.legend(); plt.grid(True); plt.show()


# --- Ejecución ---
if __name__ == '__main__':
    MODO_ENTRENAMIENTO = False
    MODEL_PATH = "tsp_pointer_network.pth"
    # ¡IMPORTANTE! Coloca aquí la ruta a tu archivo .tsp o .txt
    TSP_FILE_PATH = "C:/Users/carlo/OneDrive/Desktop/st70.txt"

    if MODO_ENTRENAMIENTO:
        print("--- MODO ENTRENAMIENTO SELECCIONADO ---")
        trained_model = train()
        print(f"Guardando modelo en {MODEL_PATH}...")
        torch.save(trained_model.state_dict(), MODEL_PATH)
        print("¡Modelo guardado con éxito!")
    else:
        print("--- MODO RESOLUCIÓN SELECCIONADO ---")
        
        if not os.path.exists(MODEL_PATH):
            print(f"Error: No se encontró el archivo del modelo en '{MODEL_PATH}'.")
            print("Por favor, pon 'MODO_ENTRENAMIENTO = True' para entrenar y guardar un modelo primero.")
        elif not os.path.exists(TSP_FILE_PATH):
            print(f"Error: No se encontró el archivo del problema TSP en '{TSP_FILE_PATH}'.")
            print("Asegúrate de que la ruta es correcta y el archivo existe.")
        else:
            print(f"Cargando modelo desde {MODEL_PATH}...")
            # El tamaño del modelo se adapta a la entrada, no necesitamos saber num_cities aquí.
            model = PointerNet(config.embedding_dim, config.hidden_dim).to(config.device)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=config.device))
            model.eval()
            print("¡Modelo cargado con éxito!")
            
            solve_instance_from_file(model, TSP_FILE_PATH, config)
