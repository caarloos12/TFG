import numpy as np
import re
from hilbertcurve.hilbertcurve import HilbertCurve

def calculate_tour_distance(points, path):
    """
    Calcula la distancia total de un recorrido TSP.
    """
    # Convierte los puntos a un array de numpy para un acceso más eficiente
    points_array = np.array(points)
    
    # Crea pares de puntos consecutivos (p1, p2), (p2, p3), ..., y añade el cierre del ciclo (pn, p1)
    path_pairs = zip(path, path[1:] + [path[0]])
    
    # Suma la distancia euclidea para cada par de puntos en el recorrido
    return sum(np.linalg.norm(points_array[p1] - points_array[p2]) for p1, p2 in path_pairs)
    
def arora_inspired_heuristic(points, grid_resolution=1024):
    """
    Implementa una heurística para el TSP inspirada en las ideas del
    algoritmo de Arora, usando una curva de Hilbert.

    Args:
        points (list): Lista de tuplas con las coordenadas (x, y) de cada punto.
        grid_resolution (int): El tamaño de la cuadrícula a la que se ajustarán
                               los puntos. Un valor más alto da más precisión.

    Returns:
        tuple: Una tupla con (ruta_final, distancia_total).
    """
    num_points = len(points)
    points_arr = np.array(points)

    # Encontrar los límites de los puntos para normalizarlos
    min_coords = points_arr.min(axis=0)
    max_coords = points_arr.max(axis=0)
    span = (max_coords - min_coords).max()

    # --- 1. Perturbación a una cuadrícula (Idea de Arora) ---
    # Normalizamos los puntos a un rango [0, grid_resolution-1] y los truncamos a enteros
    # para que caigan en los nodos de la cuadrícula.
    print(f"Paso 1: Ajustando {num_points} puntos a una cuadrícula de {grid_resolution}x{grid_resolution}...")
    normalized_points = (points_arr - min_coords) / span * (grid_resolution - 1)
    grid_points = np.round(normalized_points).astype(int)

    # --- 2. Ordenamiento con Curva de Hilbert ---
    # La curva de Hilbert nos da un ordenamiento 1D que preserva la localidad 2D.
    # El nivel 'p' de la curva debe ser tal que 2^p >= grid_resolution.
    p = (grid_resolution - 1).bit_length()
    hilbert_curve = HilbertCurve(p, 2)
    
    print(f"Paso 2: Calculando el orden de los puntos usando una curva de Hilbert de nivel {p}...")
    # Calcular la "distancia" de cada punto a lo largo de la curva
    hilbert_distances = [hilbert_curve.distance_from_point(gp) for gp in grid_points]

    # Crear una lista de índices originales de los puntos
    original_indices = list(range(num_points))
    
    # Ordenar los índices de los puntos basándose en su distancia en la curva de Hilbert
    sorted_indices = sorted(original_indices, key=lambda i: hilbert_distances[i])

    # --- 3. Construir el tour ---
    # El tour es simplemente los puntos en el orden que nos dio la curva.
    print("Paso 3: Construyendo el tour final...")
    final_path = sorted_indices

    # Calcular la distancia del tour usando las coordenadas ORIGINALES
    total_distance = calculate_tour_distance(points, final_path)
    
    return final_path, total_distance
