import networkx as nx
import numpy as np
import re # Para procesar el texto del archivo

def christofides_tsp(points):
    """
    Implementa el algoritmo de Christofides (aproximación 3/2) para el TSP.
    Corresponde al "Algoritmo 5" de la página 39.

    Args:
        points (list): Una lista de tuplas, donde cada tupla son las coordenadas (x, y)
                       de una ciudad.

    Returns:
        tuple: Una tupla que contiene la ruta (lista de índices de ciudades) y
               la distancia total de la ruta.
    """
    num_nodes = len(points)
    
    # Crear un grafo completo donde el peso de cada arista es la distancia euclidiana
    G = nx.Graph()
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
            G.add_edge(i, j, weight=dist)

    # --- Paso 1: Encontrar un Árbol Recubridor Mínimo (MST) ---
    print("Paso 1: Calculando el Árbol Recubridor Mínimo (MST)...")
    mst = nx.minimum_spanning_tree(G)

    # --- Paso 2: Encontrar vértices de grado impar y el emparejamiento ---
    print("Paso 2: Identificando nodos de grado impar y calculando el emparejamiento perfecto...")
    odd_degree_nodes = [node for node, degree in mst.degree() if degree % 2 != 0]
    odd_subgraph = G.subgraph(odd_degree_nodes)
    
    # El cálculo del emparejamiento puede ser lento para grafos grandes.
    # Usamos el algoritmo de networkx que es eficiente.
    matching = nx.min_weight_matching(odd_subgraph)
    
    # --- Paso 3: Combinar MST y el emparejamiento ---
    print("Paso 3: Creando el multigrafo euleriano...")
    multigraph = nx.MultiGraph(mst)
    for u, v in matching:
        multigraph.add_edge(u, v, weight=G[u][v]['weight'])

    # --- Paso 4: Encontrar un circuito Euleriano ---
    print("Paso 4: Encontrando el circuito euleriano...")
    start_node = list(multigraph.nodes)[0]
    eulerian_tour_edges = list(nx.eulerian_circuit(multigraph, source=start_node))

    # --- Paso 5: Convertir a ciclo Hamiltoniano (eliminar repetidos) ---
    print("Paso 5: Creando el tour final eliminando nodos repetidos...")
    final_path = []
    visited = set()
    
    for u, v in eulerian_tour_edges:
        if u not in visited:
            final_path.append(u)
            visited.add(u)
    
    final_path.append(final_path[0])
    
    total_distance = sum(G[final_path[i]][final_path[i+1]]['weight'] for i in range(len(final_path)-1))
    
    return final_path, total_distance
