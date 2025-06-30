"""
==============================================================================
|| TFG - Trabajo de Fin de Grado en Matemáticas de Carlos San Román Cazorla ||
==============================================================================
"""

"Este código ha sido realizado por Carlos San Román Cazorla"
"para el desarrollo de su Trabajo de Fin de Grado en Matemáticas."

import numpy as np
import re
import pymorton # Importamos la librería

def calculate_tour_distance(points, path):
    """Calcula la distancia total de un recorrido TSP."""
    total_distance = 0
    points_arr = np.array(points)
    for i in range(len(path) - 1):
        p1_idx, p2_idx = path[i], path[i+1]
        total_distance += np.linalg.norm(points_arr[p1_idx] - points_arr[p2_idx])
    return total_distance

def z_order_heuristic(points, grid_resolution=4096):
    """
    Implementa una heurística para el TSP inspirada en Arora, pero
    usando una curva Z-order (Morton) para el ordenamiento. (VERSIÓN CORREGIDA)

    Args:
        points (list): Lista de tuplas con las coordenadas (x, y) de cada punto.
        grid_resolution (int): El tamaño de la cuadrícula a la que se ajustarán
                               los puntos.

    Returns:
        tuple: Una tupla con (ruta_final, distancia_total).
    """
    num_points = len(points)
    points_arr = np.array(points)

    min_coords = points_arr.min(axis=0)
    max_coords = points_arr.max(axis=0)
    span = (max_coords - min_coords).max()
    
    if span == 0:
        return list(range(num_points)) + [0], 0.0

    print(f"Paso 1: Ajustando {num_points} puntos a una cuadrícula de {grid_resolution}x{grid_resolution}...")
    normalized_points = (points_arr - min_coords) / span * (grid_resolution - 1)
    grid_points = np.round(normalized_points).astype(int)

    # --- Paso 2: Ordenamiento con Curva Z-order (Morton) ---
    print(f"Paso 2: Calculando el orden de los puntos usando una curva Z-order...")
    
    # Convertimos explícitamente las coordenadas a enteros nativos de Python con int()
    # antes de pasarlos a la función de la librería pymorton.
    morton_codes = [pymorton.interleave(int(gp[0]), int(gp[1])) for gp in grid_points]

    original_indices = list(range(num_points))
    sorted_indices = sorted(original_indices, key=lambda i: morton_codes[i])

    print("Paso 3: Construyendo el tour final...")
    final_path = sorted_indices
    final_path.append(final_path[0])

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
    filename = #PONER NOMBRE DEL FICHERO
    
    try:
        print(f"Leyendo el archivo de problema '{filename}'...")
        points_data = read_tsp_file(filename)
        
        # Ejecutar la heurística con la curva Z-order
        path, distance = z_order_heuristic(points_data, grid_resolution=4096)

        print("\n" + "="*50)
        print("    RESULTADO - Heurística con Curva Z-order (Corregido)")
        print("="*50)
        print(f"Distancia total de la ruta encontrada: {distance:.2f}")
        
        if len(path) > 20:
            print(f"Ruta (primeros 10 nodos y últimos 2): {path[:10]}... -> {path[-2:]}")
        else:
            print(f"Ruta completa: {path}")

    except FileNotFoundError:
        print(f"Error: El archivo '{filename}' no fue encontrado.")
    except Exception as e:
        print(f"Ocurrió un error: {e}")
