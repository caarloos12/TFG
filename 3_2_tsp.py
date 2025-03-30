import networkx as nx
import itertools

def tsp_apx_3_2(graph):
    # 1. Calcular el Árbol de Recubrimiento Mínimo (MST)
    mst = nx.minimum_spanning_tree(graph)
    
    # 2. Encontrar los nodos de grado impar en el MST
    odd_degree_nodes = [v for v, d in mst.degree() if d % 2 == 1]
    
    # 3. Encontrar el Emparejamiento Perfecto Mínimo entre los nodos de grado impar
    subgraph = graph.subgraph(odd_degree_nodes)
    matching = nx.algorithms.matching.min_weight_matching(subgraph, maxcardinality=True)
    
    # 4. Unir las aristas del MST con el emparejamiento para formar un multigrafo Euleriano
    eulerian_multigraph = nx.MultiGraph(mst)
    for u, v in matching:
        eulerian_multigraph.add_edge(u, v, weight=graph[u][v]['weight'])
    
    # 5. Encontrar el circuito Euleriano
    eulerian_circuit = list(nx.eulerian_circuit(eulerian_multigraph, source=next(iter(graph.nodes))))
    
    # 6. Convertirlo en un ciclo Hamiltoniano eliminando nodos repetidos
    visited = set()
    tsp_path = []
    for u, v in eulerian_circuit:
        if u not in visited:
            tsp_path.append(u)
            visited.add(u)
    tsp_path.append(tsp_path[0])  # Regresar al inicio
    
    return tsp_path

# Ejemplo de uso con un grafo completo ponderado
graph = nx.complete_graph(5)
nx.set_edge_attributes(graph, {(u, v): {'weight': (u+v) % 10 + 1} for u, v in graph.edges})

tour = tsp_apx_3_2(graph)
print("Recorrido aproximado del TSP:", tour)
