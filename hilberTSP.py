import numpy as np
import re
from hilbertcurve.hilbertcurve import HilbertCurve

def calculate_tour_distance(points, path):
    """Calcula la distancia total de un recorrido TSP."""
    total_distance = 0
    for i in range(len(path) - 1):
        p1_idx, p2_idx = path[i], path[i+1]
        total_distance += np.linalg.norm(np.array(points[p1_idx]) - np.array(points[p2_idx]))
    return total_distance

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
    # Cerrar el ciclo
    final_path.append(final_path[0])

    # Calcular la distancia del tour usando las coordenadas ORIGINALES
    total_distance = calculate_tour_distance(points, final_path)
    
    return final_path, total_distance

def read_tsp_file(filepath):
    """Lee un archivo de coordenadas en formato TSPLIB."""
    points = []
    with open(filepath, 'r') as f:
        content = f.read()

    coord_section_match = re.search(r'NODE_COORD_SECTION\s*([\s\S]*)', content)
    if not coord_section_match:
        raise ValueError("No se encontró 'NODE_COORD_SECTION' en el archivo.")
        
    coord_str = coord_section_match.group(1)
    lines = coord_str.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if line.upper() == 'EOF' or not line:
            break
        parts = line.split()
        if len(parts) >= 3:
            points.append((float(parts[1]), float(parts[2])))
            
    return points

# --- Ejemplo de uso con el archivo d2103.txt ---
if __name__ == "__main__":
    filename = "C:/Users/carlo/OneDrive/Desktop/d1655.txt"
    
    try:
        print(f"Leyendo el archivo de problema '{filename}'...")
        points_data = read_tsp_file(filename)
        
        # Ejecutar la heurística
        path, distance = arora_inspired_heuristic(points_data, grid_resolution=4096)

        print("\n" + "="*50)
        print("    RESULTADO - Heurística inspirada en Arora")
        print("="*50)
        print(f"Distancia total de la ruta encontrada: {distance:.2f}")
        
        # Imprimir solo una parte de la ruta si es muy larga
        if len(path) > 20:
            print(f"Ruta (primeros 10 nodos y últimos 2): {path[:10]}... -> {path[-2:]}")
        else:
            print(f"Ruta completa: {path}")

    except FileNotFoundError:
        print(f"Error: El archivo '{filename}' no fue encontrado.")
    except Exception as e:
        print(f"Ocurrió un error: {e}")
