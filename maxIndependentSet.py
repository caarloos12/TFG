import numpy as np
from scipy.optimize import linprog
from collections import defaultdict

def greedy_coloring(graph):
    """
    Colorea un grafo usando un algoritmo voraz.
    Cada color representa un conjunto independiente.

    Args:
        graph (dict): El grafo como una lista de adyacencia.
                      Ej: {0: [1, 2], 1: [0], 2: [0]}

    Returns:
        dict: Un diccionario que mapea cada vértice a un color (entero).
    """
    colors = {}
    nodes = sorted(graph.keys()) # Ordenar para consistencia

    for node in nodes:
        neighbor_colors = {colors[neighbor] for neighbor in graph.get(node, []) if neighbor in colors}
        
        current_color = 0
        while current_color in neighbor_colors:
            current_color += 1
        colors[node] = current_color
        
    return colors

def solve_max_independent_set(graph):
    """
    Implementación del Algoritmo 1 de la página 33 para aproximar
    la solución del problema MAX INDEPENDENT SET.

    Args:
        graph (dict): El grafo como una lista de adyacencia.
                      Los nodos deben ser enteros desde 0 hasta n-1.

    Returns:
        set: Un conjunto de nodos que forma un conjunto independiente.
    """
    num_nodes = len(graph)
    if num_nodes == 0:
        return set()

    node_list = sorted(graph.keys())
    node_map = {node: i for i, node in enumerate(node_list)}
    
    # --- Paso 1: Resolver la relajación lineal del problema ---
    # Queremos maximizar sum(x_i), que es equivalente a minimizar -sum(x_i)
    c = -np.ones(num_nodes)

    # Creamos las restricciones: para cada arista (u, v), x_u + x_v <= 1
    constraints = []
    for u in graph:
        for v in graph[u]:
            if u < v: # Evitar duplicados
                row = np.zeros(num_nodes)
                row[node_map[u]] = 1
                row[node_map[v]] = 1
                constraints.append(row)

    A_ub = np.array(constraints) if constraints else None
    b_ub = np.ones(len(constraints)) if constraints else None

    # Cada variable x_i debe estar entre 0 y 1
    bounds = (0, 1)

    # Solucionamos el problema de programación lineal
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)

    if not result.success:
        print("La relajación lineal no pudo ser resuelta.")
        # Como fallback, devolvemos el resultado de un coloreo simple sobre todo el grafo
        coloring = greedy_coloring(graph)
        color_groups = defaultdict(list)
        for node, color in coloring.items():
            color_groups[color].append(node)
        return set(max(color_groups.values(), key=len))

    # Clasificamos los vértices según el resultado de la relajación
    V0, V1, V_half = set(), set(), set()
    tolerance = 1e-5
    for i, x in enumerate(result.x):
        node = node_list[i]
        if abs(x - 0) < tolerance:
            V0.add(node)
        elif abs(x - 1) < tolerance:
            V1.add(node)
        elif abs(x - 0.5) < tolerance:
            V_half.add(node)
            
    # --- Paso 2: Colorear el subgrafo G[V_1/2] ---
    subgraph_half = {u: [v for v in neighbors if v in V_half] 
                     for u, neighbors in graph.items() if u in V_half}
    
    coloring = greedy_coloring(subgraph_half)
    
    # --- Paso 3: Encontrar la clase de color más grande (S_hat) ---
    if not coloring:
        S_hat = set()
    else:
        color_groups = defaultdict(list)
        for node, color in coloring.items():
            color_groups[color].append(node)
        S_hat = set(max(color_groups.values(), key=len))
        
    # --- Paso 4: La salida es la unión de V1 y S_hat ---
    final_solution = V1.union(S_hat)
    
    return final_solution
