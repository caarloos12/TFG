import networkx as nx
import numpy as np
import re # Para procesar el texto del archivo

def christofides_tsp(points):
    """
    Implementa el algoritmo de Christofides (aproximación 3/2) para el TSP.
    Corresponde al "Algorithm 5" de la página 39.

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


def read_and_solve_tsp_file(filepath):
    """
    Lee un archivo de coordenadas en formato TSPLIB, lo resuelve con Christofides
    y muestra el resultado.

    Args:
        filepath (str): La ruta al archivo .txt o .tsp.
    """
    points = []
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # Usar expresiones regulares para encontrar la sección de coordenadas
        coord_section_match = re.search(r'NODE_COORD_SECTION\s*([\s\S]*)', content)
        if not coord_section_match:
            raise ValueError("No se encontró 'NODE_COORD_SECTION' en el archivo.")
            
        coord_str = coord_section_match.group(1)
        
        # Parsea cada línea de la sección de coordenadas
        lines = coord_str.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.upper() == 'EOF':
                break
            
            # Divide la línea por espacios y toma las coordenadas (ignora el ID)
            parts = line.split()
            if len(parts) == 3:
                try:
                    points.append((float(parts[1]), float(parts[2])))
                except (ValueError, IndexError):
                    print(f"Advertencia: No se pudo parsear la línea: '{line}'")

        if not points:
            raise ValueError("No se encontraron coordenadas válidas en el archivo.")

        print(f"Se leyeron {len(points)} puntos del archivo '{filepath}'.")
        print("Iniciando el algoritmo de Christofides (puede tardar para muchos puntos)...")
        
        # Resolver el problema con los puntos leídos
        path, distance = christofides_tsp(points)

        print("\n" + "="*50)
        print("    RESULTADO FINAL")
        print("="*50)
        print(f"Distancia total de la ruta encontrada: {distance:.2f}")
        # Imprimir solo una parte de la ruta si es muy larga
        if len(path) > 20:
            print(f"Ruta (primeros 10 nodos y últimos 2): {path[:10]}... -> {path[-2:]}")
        else:
            print(f"Ruta completa: {path}")

    except FileNotFoundError:
        print(f"Error: El archivo '{filepath}' no fue encontrado.")
    except Exception as e:
        print(f"Ocurrió un error: {e}")

# --- Ejemplo de uso con el archivo proporcionado ---
if __name__ == "__main__":
    # Nombre del archivo que contiene los datos del problema d2103
    # Asegúrate de que este archivo esté en la misma carpeta que tu script.
    filename = "C:/Users/carlo/OneDrive/Desktop/d1655.txt"
    
    read_and_solve_tsp_file(filename)
nx.set_edge_attributes(graph, {(u, v): {'weight': (u+v) % 10 + 1} for u, v in graph.edges})

tour = tsp_apx_3_2(graph)
print("Recorrido aproximado del TSP:", tour)
