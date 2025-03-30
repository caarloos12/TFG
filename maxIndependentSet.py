import networkx as nx

def max_independent_set_r(G):
    """
    Algoritmo heurístico para dividir los nodos en V0, V1 y V1/2.
    - V0: conjunto de nodos eliminados (por heurística o reducción).
    - V1: conjunto de nodos que están seguramente en la solución óptima.
    - V1/2: conjunto de nodos restantes a procesar.
    """
    V0, V1, V1_2 = set(), set(), set(G.nodes)
    
    for node in list(G.nodes):
        if G.degree[node] == 0:
            V1.add(node)
            V1_2.remove(node)
        elif G.degree[node] == 1:
            neighbor = next(iter(G.neighbors(node)))
            V1.add(node)
            V0.add(neighbor)
            V1_2.discard(node)
            V1_2.discard(neighbor)
    
    return V0, V1, V1_2

def greedy_coloring(G):
    """Colorea el grafo usando el algoritmo de coloreo voraz."""
    coloring = nx.coloring.greedy_color(G, strategy="largest_first")
    color_classes = {}
    for node, color in coloring.items():
        if color not in color_classes:
            color_classes[color] = set()
        color_classes[color].add(node)
    
    # Seleccionar el conjunto independiente más grande
    max_color_class = max(color_classes.values(), key=len)
    return max_color_class

def solve_max_independent_set(G):
    # Paso 1: Obtener V0, V1 y V1/2
    V0, V1, V1_2 = max_independent_set_r(G)
    
    # Paso 2: Extraer el subgrafo G[V1/2]
    G_V1_2 = G.subgraph(V1_2)
    
    # Paso 3: Colorear G[V1/2]
    S_hat = greedy_coloring(G_V1_2)
    
    # Paso 4: Unir V1 y el conjunto coloreado más grande
    S = V1.union(S_hat)
    return S

# Ejemplo de uso
graph = nx.erdos_renyi_graph(10, 0.5)  # Grafo aleatorio con 10 nodos y probabilidad 0.5
independent_set = solve_max_independent_set(graph)
print("Conjunto independiente aproximado:", independent_set)
