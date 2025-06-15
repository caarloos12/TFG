import networkx as nx
import itertools 
import re 
import numpy as np
import random
import matplotlib.pyplot as plt
import os # For file operations

# --- Helper functions for EUC_2D problems (from user's GA code) ---
def load_tsplib_problem_coords(filepath):
    """Carga las coordenadas de los nodos desde un archivo en formato TSPLIB (NODE_COORD_SECTION)."""
    print(f"GA Part (Coords Loader): Loading coordinates from {filepath}")
    with open(filepath, 'r') as f:
        lines = f.readlines()

    coords = []
    in_coord_section = False
    dimension = 0 # Try to get dimension from header as well
    for line in lines:
        line = line.strip()
        if line.startswith("DIMENSION"):
            try:
                dimension = int(line.split(':')[1].strip())
                print(f"GA Part (Coords Loader): Dimension from header: {dimension}")
            except:
                pass # Ignore if not found or malformed
        elif line.startswith("NODE_COORD_SECTION"):
            in_coord_section = True
            print("GA Part (Coords Loader): Entering NODE_COORD_SECTION.")
            continue
        elif line.startswith("EOF"):
            break
        
        if in_coord_section:
            parts = line.split()
            try:
                # TSPLIB node indices are often 1-based, but we store coords 0-indexed.
                # Node ID (parts[0]) is ignored here, assuming sequential nodes.
                coords.append([float(parts[1]), float(parts[2])])
            except ValueError:
                print(f"GA Part (Coords Loader): Warning - Could not parse coordinate line: {line}")
                continue
            
    if dimension > 0 and len(coords) != dimension:
        print(f"GA Part (Coords Loader): Warning - Number of coordinates ({len(coords)}) does not match DIMENSION ({dimension}).")
    
    print(f"GA Part (Coords Loader): Loaded {len(coords)} city coordinates.")
    return np.array(coords)

def calculate_distance_matrix_from_coords(cities_coords):
    """Calcula la matriz de distancias euclidianas entre todas las ciudades."""
    num_cities = len(cities_coords)
    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(i, num_cities): # Calculate only upper triangle + diagonal
            dist = np.linalg.norm(cities_coords[i] - cities_coords[j])
            dist_matrix[i, j] = dist_matrix[j, i] = dist
    print(f"GA Part (Coords Loader): Calculated distance matrix for {num_cities} cities.")
    return dist_matrix

# --- Part 1: Christofides 3/2-Approximation Algorithm ---
# (parse_tsplib_file_for_christofides is for EXPLICIT weights, not used for EUC_2D in the new main)
def parse_tsplib_file_for_christofides(file_path):
    """
    Parses a TSPLIB file (like hk48.txt with EXPLICIT LOWER_DIAG_ROW weights) 
    to create a NetworkX graph for the Christofides algorithm.
    """
    graph = nx.Graph()
    dimension = 0
    edge_weight_format = ""
    edge_weight_type = ""
    reading_edge_weights = False
    weights_data = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("COMMENT"):
                    continue

                if line.startswith("NAME"):
                    print(f"Christofides Part (Explicit Weights Parser): Parsing TSP problem: {line.split(':')[1].strip()}")
                elif line.startswith("TYPE"):
                    tsp_type = line.split(':')[1].strip()
                    if tsp_type not in ["TSP", "ATSP"]:
                        print(f"Christofides Part (Explicit Weights Parser): Warning: File type is {tsp_type}.")
                elif line.startswith("DIMENSION"):
                    dimension = int(line.split(':')[1].strip())
                    print(f"Christofides Part (Explicit Weights Parser): Dimension (number of cities): {dimension}")
                    graph.add_nodes_from(range(dimension))
                elif line.startswith("EDGE_WEIGHT_TYPE"):
                    edge_weight_type = line.split(':')[1].strip()
                    if edge_weight_type != "EXPLICIT":
                        print(f"Christofides Part (Explicit Weights Parser): Unsupported EDGE_WEIGHT_TYPE: {edge_weight_type}.")
                        return None
                elif line.startswith("EDGE_WEIGHT_FORMAT"):
                    edge_weight_format = line.split(':')[1].strip()
                    if edge_weight_format != "LOWER_DIAG_ROW":
                        print(f"Christofides Part (Explicit Weights Parser): Unsupported EDGE_WEIGHT_FORMAT: {edge_weight_format}.")
                        return None
                elif line.startswith("EDGE_WEIGHT_SECTION"):
                    if dimension == 0 or edge_weight_type != "EXPLICIT" or edge_weight_format != "LOWER_DIAG_ROW":
                        print("Christofides Part (Explicit Weights Parser): Error: EDGE_WEIGHT_SECTION found before DIMENSION or with unsupported formats.")
                        return None
                    reading_edge_weights = True
                    print("Christofides Part (Explicit Weights Parser): Reading EDGE_WEIGHT_SECTION...")
                    continue
                elif line == "EOF":
                    reading_edge_weights = False
                    break

                if reading_edge_weights:
                    weights_data.extend([int(x) for x in line.split()])

    except FileNotFoundError:
        print(f"Christofides Part (Explicit Weights Parser): Error: File not found at {file_path}")
        return None
    except ValueError:
        print("Christofides Part (Explicit Weights Parser): Error: Non-integer value found in edge weight section.")
        return None
    except Exception as e:
        print(f"Christofides Part (Explicit Weights Parser): An error occurred during file parsing: {e}")
        return None

    if not weights_data:
        print("Christofides Part (Explicit Weights Parser): Error: No edge weights found in the file.")
        return None

    if edge_weight_format == "LOWER_DIAG_ROW":
        current_idx = 0
        for i in range(dimension):
            for j in range(i + 1):
                if current_idx >= len(weights_data):
                    print("Christofides Part (Explicit Weights Parser): Error: Not enough weight data for the specified dimension.")
                    return None
                weight = weights_data[current_idx]
                if i != j:
                    graph.add_edge(i, j, weight=weight)
                current_idx += 1
        
        if current_idx < len(weights_data) and any(w != 0 for w in weights_data[current_idx:] if isinstance(w, (int, float))):
            is_significant_remaining = any(weights_data[k_rem] != 0 for k_rem in range(current_idx, len(weights_data)))
            if is_significant_remaining:
                 print(f"Christofides Part (Explicit Weights Parser): Warning: Extra data found in EDGE_WEIGHT_SECTION.")
    else:
        print(f"Christofides Part (Explicit Weights Parser): Logic for {edge_weight_format} not implemented.")
        return None
        
    print(f"Christofides Part (Explicit Weights Parser): Successfully parsed graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    return graph


def tsp_apx_3_2(graph):
    """
    Calculates an approximate TSP tour using Christofides' 3/2-approximation algorithm.
    """
    if graph is None or graph.number_of_nodes() == 0:
        print("Christofides Part: Error: Input graph is empty or None.")
        return []
    
    is_fully_connected_check = True
    if graph.number_of_nodes() > 1:
        if not nx.is_connected(graph):
            print("Christofides Part: Error: Graph is not connected. Cannot compute MST for all nodes.")
            if graph.number_of_nodes() == 1:
                node = list(graph.nodes())[0]
                return [node, node]
            return []

    print("Christofides Part: Calculating Minimum Spanning Tree (MST)...")
    mst = nx.minimum_spanning_tree(graph, weight='weight')
    print(f"Christofides Part: MST calculated with {mst.number_of_edges()} edges.")
    
    odd_degree_nodes = [v for v, d in mst.degree() if d % 2 == 1]
    print(f"Christofides Part: Found {len(odd_degree_nodes)} odd-degree nodes in MST: {odd_degree_nodes}")

    matching = set() 
    if len(odd_degree_nodes) > 0 :
        subgraph_odd_nodes_for_matching = graph.subgraph(odd_degree_nodes).copy() 
        
        if subgraph_odd_nodes_for_matching.number_of_nodes() > 1 and subgraph_odd_nodes_for_matching.number_of_edges() == 0:
            print(f"Christofides Part: Warning: Subgraph for matching has {subgraph_odd_nodes_for_matching.number_of_nodes()} nodes but no edges. This is unusual.")
        
        if subgraph_odd_nodes_for_matching.number_of_nodes() > 0 : 
            print(f"Christofides Part: Finding Minimum Weight Perfect Matching on {len(odd_degree_nodes)} odd-degree nodes...")
            matching = nx.min_weight_matching(subgraph_odd_nodes_for_matching, weight='weight')
            print(f"Christofides Part: Matching found with {len(matching)} edges: {list(matching)}")
        else:
            print("Christofides Part: No odd-degree nodes to form a subgraph for matching (e.g. single node MST).")
    else:
        print("Christofides Part: No odd-degree nodes in MST (MST is already Eulerian or trivial).")

    
    eulerian_multigraph = nx.MultiGraph(mst.edges(data=True))
    print(f"Christofides Part: Eulerian graph initialized with {eulerian_multigraph.number_of_edges()} edges from MST.")
    
    edges_added_from_matching = 0
    for u, v in matching: 
        if graph.has_edge(u,v):
            weight = graph[u][v].get('weight', 1) 
            eulerian_multigraph.add_edge(u, v, weight=weight)
            edges_added_from_matching +=1
        else:
            print(f"Christofides Part: Warning! Edge ({u},{v}) from matching not found in original graph. This is unexpected.")
    print(f"Christofides Part: Added {edges_added_from_matching} edges from matching to Eulerian graph. Total edges: {eulerian_multigraph.number_of_edges()}")


    if not list(eulerian_multigraph.nodes()):
        print("Christofides Part: Error: Eulerian multigraph has no nodes after MST and matching.")
        if graph.number_of_nodes() == 1: node = list(graph.nodes())[0]; return [node, node]
        return []
    
    source_node_candidates = list(eulerian_multigraph.nodes())
    if not source_node_candidates: 
        print("Christofides Part: Error: No nodes in Eulerian multigraph to select a source for circuit.")
        return []
    source_node = source_node_candidates[0]

    print(f"Christofides Part: Finding Eulerian circuit starting from node {source_node}...")
    try:
        is_eul = nx.is_eulerian(eulerian_multigraph)
        if not is_eul:
            print("Christofides Part: Error: Constructed multigraph is NOT Eulerian. Checking details...")
            if not nx.is_connected(nx.Graph(eulerian_multigraph)): 
                print("Christofides Part: Detail: Multigraph is not connected.")
            odd_degree_count = 0
            for n, d in eulerian_multigraph.degree():
                if d % 2 != 0:
                    print(f"Christofides Part: Detail: Node {n} has odd degree {d}.")
                    odd_degree_count +=1
            if odd_degree_count > 0:
                 print(f"Christofides Part: Detail: Found {odd_degree_count} nodes with odd degrees in multigraph.")
            return [] 

        eulerian_circuit_edges = list(nx.eulerian_circuit(eulerian_multigraph, source=source_node, keys=False))
    except nx.NetworkXError as e: 
        print(f"Christofides Part: Error finding Eulerian circuit: {e}")
        return [] 
    
    if not eulerian_circuit_edges:
        print("Christofides Part: Eulerian circuit is empty (no edges returned by nx.eulerian_circuit).")
        if graph.number_of_nodes() == 1: node = list(graph.nodes())[0]; return [node, node]
        if eulerian_multigraph.number_of_nodes() == 1 and eulerian_multigraph.number_of_edges() == 0:
            node = list(eulerian_multigraph.nodes())[0]
            return [node, node] 
        return []

    path_from_edges = [eulerian_circuit_edges[0][0]] 
    for edge_u, edge_v in eulerian_circuit_edges:
        path_from_edges.append(edge_v) 

    visited = set()
    tsp_path = []
    for node in path_from_edges:
        if node not in visited:
            tsp_path.append(node)
            visited.add(node)
    
    if tsp_path: 
        tsp_path.append(tsp_path[0])  
    elif graph.number_of_nodes() > 0: 
        node = list(graph.nodes())[0]
        tsp_path = [node, node]

    print(f"Christofides Part: Final TSP approximate tour: {tsp_path} (Length: {len(tsp_path)})")
    return tsp_path


# --- Part 2: Genetic Algorithm ---
class GeneticAlgorithmTSP:
    def __init__(self, num_cities, dist_matrix, initial_solution=None, 
                 population_size=100, mutation_rate=0.01, crossover_rate=0.9, elitism_size=5):
        self.num_cities = num_cities
        self.dist_matrix = np.array(dist_matrix) 
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_size = elitism_size
        
        self.processed_initial_solution = None
        if initial_solution:
            temp_initial = list(initial_solution)
            if len(temp_initial) > 1 and temp_initial[0] == temp_initial[-1]:
                self.processed_initial_solution = temp_initial[:-1]
            else:
                self.processed_initial_solution = temp_initial

            if not (len(self.processed_initial_solution) == self.num_cities and \
                    len(set(self.processed_initial_solution)) == self.num_cities and \
                    all(0 <= city < self.num_cities for city in self.processed_initial_solution)):
                print(f"GA Part: Warning: Provided initial solution is invalid after processing: {self.processed_initial_solution}. Num cities: {self.num_cities}. Ignoring it.")
                self.processed_initial_solution = None
        
        self.population = self._initialize_population()

    def _initialize_population(self):
        population = []
        if self.processed_initial_solution:
            print("GA Part: Using provided initial solution as the first individual.")
            population.append(list(self.processed_initial_solution)) 
            
        base_tour = list(range(self.num_cities))
        while len(population) < self.population_size:
            tour = random.sample(base_tour, self.num_cities)
            is_duplicate_of_initial = False
            if self.processed_initial_solution and population and tour == population[0]:
                is_duplicate_of_initial = True
            
            if not is_duplicate_of_initial:
                 population.append(tour)
        return population

    def _calculate_tour_distance(self, tour):
        total_distance = 0
        for i in range(self.num_cities):
            start_city = tour[i]
            end_city = tour[(i + 1) % self.num_cities] 
            total_distance += self.dist_matrix[start_city, end_city]
        return total_distance

    def _calculate_fitness(self, tour):
        distance = self._calculate_tour_distance(tour)
        return 1 / (distance + 1e-9) 

    def _selection(self, ranked_population):
        tournament_size = 3
        actual_tournament_size = min(tournament_size, len(ranked_population))
        if actual_tournament_size == 0: return random.sample(list(range(self.num_cities)), self.num_cities) 

        selection_candidates = random.sample(ranked_population, actual_tournament_size)
        selection_candidates.sort(key=lambda x: x[1], reverse=True) 
        return list(selection_candidates[0][0]) 

    def _crossover(self, parent1, parent2):
        child = [None] * self.num_cities
        start, end = sorted(random.sample(range(self.num_cities), 2))
        
        child[start:end+1] = parent1[start:end+1]
        cities_in_segment = set(parent1[start:end+1])
        
        parent2_pointer = 0
        for i in range(self.num_cities):
            child_idx_to_fill = (end + 1 + i) % self.num_cities 
            
            if child[child_idx_to_fill] is None: 
                while parent2_pointer < self.num_cities:
                    city_from_parent2 = parent2[parent2_pointer]
                    parent2_pointer += 1
                    if city_from_parent2 not in cities_in_segment:
                        child[child_idx_to_fill] = city_from_parent2
                        cities_in_segment.add(city_from_parent2) # Add to used cities conceptually
                        break 
        
        if None in child or len(set(child)) != self.num_cities:
            # print(f"GA Part: Crossover resulted in an invalid child: {child}. Parent1: {parent1}, Parent2: {parent2}, Start: {start}, End: {end}. Resorting to parent1.")
            present_cities_in_final_child = set()
            final_child = [None] * self.num_cities
            temp_child_copy = list(child) 
            
            for i in range(self.num_cities):
                if temp_child_copy[i] is not None and temp_child_copy[i] not in present_cities_in_final_child:
                    if 0 <= temp_child_copy[i] < self.num_cities : # Check if city index is valid
                        final_child[i] = temp_child_copy[i]
                        present_cities_in_final_child.add(temp_child_copy[i])

            all_possible_cities = set(range(self.num_cities))
            missing_cities = list(all_possible_cities - present_cities_in_final_child)
            random.shuffle(missing_cities)
            
            missing_cities_iter = iter(missing_cities)
            for i in range(self.num_cities):
                if final_child[i] is None:
                    try:
                        final_child[i] = next(missing_cities_iter)
                    except StopIteration: 
                        return list(parent1) 
            
            if None in final_child or len(set(final_child)) != self.num_cities:
                return list(parent1) 
            return final_child
        return child


    def _mutate(self, tour):
        if self.num_cities < 2: return tour
        idx1, idx2 = random.sample(range(self.num_cities), 2)
        mutated_tour = list(tour) 
        mutated_tour[idx1], mutated_tour[idx2] = mutated_tour[idx2], mutated_tour[idx1]
        return mutated_tour

    def run(self, generations):
        print("GA Part: Initiating evolution with Genetic Algorithm...")
        history = {'best_distance': [], 'avg_distance': []}

        for gen in range(generations):
            ranked_population = []
            total_distance_in_gen = 0
            
            if not self.population: 
                print("GA Part: Error - Population is empty at start of generation. Cannot proceed.")
                return None, float('inf'), history

            valid_tours_in_pop = 0
            for tour in self.population:
                if tour is None or len(tour) != self.num_cities or len(set(tour)) != self.num_cities:
                    continue 
                valid_tours_in_pop +=1
                distance = self._calculate_tour_distance(tour)
                fitness = self._calculate_fitness(tour) 
                ranked_population.append((tour, fitness, distance))
                total_distance_in_gen += distance
            
            if not ranked_population: 
                 print(f"GA Part: Error - No valid tours in population to rank (Total processed: {valid_tours_in_pop}). Cannot proceed.")
                 return None, float('inf'), history

            ranked_population.sort(key=lambda x: x[1], reverse=True)
            
            best_tour_data = ranked_population[0]
            history['best_distance'].append(best_tour_data[2])
            avg_dist_for_gen = total_distance_in_gen / valid_tours_in_pop if valid_tours_in_pop > 0 else float('inf')
            history['avg_distance'].append(avg_dist_for_gen)

            if (gen + 1) % 50 == 0 or gen == 0 or gen == generations -1 :
                print(f"GA Part: Gen {gen+1:4}/{generations} | Best Dist: {best_tour_data[2]:.2f} | Avg Dist: {avg_dist_for_gen:.2f} | Valid tours in pop: {valid_tours_in_pop}/{len(self.population)}")

            new_population = []
            elitism_count = min(self.elitism_size, len(ranked_population))
            for i in range(elitism_count):
                new_population.append(list(ranked_population[i][0])) 
            
            while len(new_population) < self.population_size:
                parent1 = self._selection(ranked_population)
                parent2 = self._selection(ranked_population)
                
                child = list(parent1) 
                if random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)
                
                if child is None or len(child) != self.num_cities or len(set(child)) != self.num_cities:
                    new_population.append(list(parent1)) 
                else:
                    new_population.append(child)

            self.population = new_population
        
        final_ranked_population = []
        final_total_dist = 0
        valid_final_tours = 0
        for tour in self.population: # Recalculate best from final population
            if tour is None or len(tour) != self.num_cities or len(set(tour)) != self.num_cities: continue
            valid_final_tours +=1
            dist = self._calculate_tour_distance(tour)
            final_ranked_population.append((tour, 1/(dist+1e-9), dist))
            final_total_dist += dist
        
        if not final_ranked_population:
            print("GA Part: Evolution finished, but final population had no valid tours.")
            return None, float('inf'), history

        final_ranked_population.sort(key=lambda x: x[1], reverse=True) # Sort by fitness (higher is better)
        best_overall_tour_data = final_ranked_population[0]
        best_overall_tour = best_overall_tour_data[0]
        best_overall_distance = best_overall_tour_data[2] # Distance is the 3rd element
        
        print(f"GA Part: Evolution finished. Best distance found: {best_overall_distance:.2f}")
        return best_overall_tour, best_overall_distance, history

# --- Part 3: Visualization ---

def plot_ga_evolution(history, problem_name="", title_suffix=""):
    """Graphs the evolution of best and average distance per generation."""
    plt.figure(figsize=(12, 6))
    plt.plot(history['best_distance'], label="Best Distance", color="dodgerblue", linewidth=2)
    plt.plot(history['avg_distance'], label="Average Distance", color="tomato", linestyle="--")
    full_title = f"GA Evolution for {problem_name} {title_suffix}".strip()
    plt.title(full_title, fontsize=16)
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Total Tour Distance", fontsize=12)
    plt.legend()
    plt.grid(True)
    
    # filename = f"ga_evolution_{problem_name}{title_suffix.replace(' ', '_').replace('(', '').replace(')', '').replace(':', '')}.png"
    # plt.savefig(filename) # Comentado para mostrar en ventana
    # print(f"GA evolution plot would be saved to {filename}")
    # plt.close() # Comentado para mostrar en ventana
    plt.show() # Mostrar en ventana emergente


def plot_final_tour(cities_coords, best_tour, best_distance, problem_name=""):
    """Grafica el mejor recorrido encontrado (requires city coordinates)."""
    if cities_coords is None or len(cities_coords) == 0:
        print("Plotting: Cannot plot final tour, city coordinates not available.")
        return

    plt.figure(figsize=(10, 10))
    ordered_city_coords = np.array([cities_coords[i] for i in best_tour + [best_tour[0]]])
    
    plt.plot(ordered_city_coords[:, 0], ordered_city_coords[:, 1], 'o-', color='navy', markersize=4, zorder=2, label="Best Tour Path")
    plt.scatter(cities_coords[:, 0], cities_coords[:, 1], c='red', s=15, zorder=3, label="Cities")
    
    plt.title(f"Best Tour for {problem_name}\nDistance: {best_distance:.2f}", fontsize=16)
    plt.xlabel("X Coordinate", fontsize=12)
    plt.ylabel("Y Coordinate", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    
    # filename = f"final_tour_{problem_name}.png"
    # plt.savefig(filename) # Comentado para mostrar en ventana
    # print(f"Final tour plot would be saved to {filename}")
    # plt.close() # Comentado para mostrar en ventana
    plt.show() # Mostrar en ventana emergente


# --- Main Execution Logic ---
if __name__ == "__main__":
    PROBLEM_FILE_NAME = "C:/Users/carlo/OneDrive/Desktop/d2103.txt"
    
    POPULATION_SIZE = 150
    GENERATIONS = 1000 
    MUTATION_RATE = 0.02
    CROSSOVER_RATE = 0.9
    ELITISM_SIZE = 10 

    rat195_content = """
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
    if PROBLEM_FILE_NAME == "rat195.tsp":
        with open(PROBLEM_FILE_NAME, "w") as f:
            f.write(rat195_content)
        print(f"Content for {PROBLEM_FILE_NAME} written to disk.")

    cities_coords_for_ga = None
    dist_matrix_for_algs = None
    num_cities_for_algs = 0
    problem_type_is_euc_2d = False

    try:
        with open(PROBLEM_FILE_NAME, 'r') as f_check:
            file_head = "".join(f_check.readline() for _ in range(10)) 
            if "EUC_2D" in file_head:
                problem_type_is_euc_2d = True
            elif "EXPLICIT" in file_head: 
                problem_type_is_euc_2d = False 
            else: 
                print(f"Warning: Could not determine problem type from header of {PROBLEM_FILE_NAME}. Assuming EUC_2D if coordinates are found.")
                problem_type_is_euc_2d = True 

    except FileNotFoundError:
        print(f"Error: Problem file {PROBLEM_FILE_NAME} not found or not written correctly.")
        exit()


    if problem_type_is_euc_2d:
        print(f"Processing {PROBLEM_FILE_NAME} as EUC_2D problem.")
        cities_coords_for_ga = load_tsplib_problem_coords(PROBLEM_FILE_NAME)
        if cities_coords_for_ga is not None and len(cities_coords_for_ga) > 0:
            dist_matrix_for_algs = calculate_distance_matrix_from_coords(cities_coords_for_ga)
            num_cities_for_algs = len(cities_coords_for_ga)
        else:
            print(f"Error: Could not load coordinates for EUC_2D problem {PROBLEM_FILE_NAME}.")
            exit()
    else: 
        print(f"Processing {PROBLEM_FILE_NAME} as EXPLICIT weight problem.")
        temp_graph = parse_tsplib_file_for_christofides(PROBLEM_FILE_NAME)
        if temp_graph:
            num_cities_for_algs = temp_graph.number_of_nodes()
            node_list = sorted(temp_graph.nodes()) 
            dist_matrix_for_algs = nx.to_numpy_array(temp_graph, nodelist=node_list, weight='weight')
        else:
            print(f"Error: Could not parse EXPLICIT problem {PROBLEM_FILE_NAME}.")
            exit()

    print("\n--- Running Christofides 3/2-Approximation Algorithm ---")
    christofides_graph_input = nx.Graph()
    if num_cities_for_algs > 0:
        christofides_graph_input.add_nodes_from(range(num_cities_for_algs))
        for i in range(num_cities_for_algs):
            for j in range(i + 1, num_cities_for_algs):
                christofides_graph_input.add_edge(i, j, weight=dist_matrix_for_algs[i, j])
    
    initial_christofides_tour = []
    christofides_tour_cost = float('inf')

    if christofides_graph_input.number_of_nodes() > 0:
        initial_christofides_tour = tsp_apx_3_2(christofides_graph_input)
        if initial_christofides_tour and len(initial_christofides_tour) > 1:
            christofides_tour_cost = 0
            tour_perm = initial_christofides_tour[:-1] 
            if tour_perm : # Check if tour_perm is not empty
                for i in range(len(tour_perm)):
                    u, v = tour_perm[i], tour_perm[(i + 1) % len(tour_perm)]
                    christofides_tour_cost += dist_matrix_for_algs[u, v]
            print(f"Christofides Tour: {initial_christofides_tour}")
            print(f"Christofides Tour Cost: {christofides_tour_cost:.2f}")
        else:
            print("Christofides algorithm did not return a valid tour.")
            initial_christofides_tour = [] 
    else:
        print(f"Could not create graph for Christofides from {PROBLEM_FILE_NAME}.")

    if num_cities_for_algs > 0:
        print("\n--- Running Genetic Algorithm ---")
        ga_solver = GeneticAlgorithmTSP(
            num_cities=num_cities_for_algs,
            dist_matrix=dist_matrix_for_algs,
            initial_solution=initial_christofides_tour if initial_christofides_tour else None,
            population_size=POPULATION_SIZE,
            mutation_rate=MUTATION_RATE,
            crossover_rate=CROSSOVER_RATE,
            elitism_size=ELITISM_SIZE
        )

        best_ga_tour, best_ga_distance, ga_history = ga_solver.run(generations=GENERATIONS)

        if best_ga_tour:
            print("\n--- Genetic Algorithm Results ---")
            print(f"Best Tour from GA (permutation): {best_ga_tour}")
            print(f"Full Best Tour from GA: {best_ga_tour + [best_ga_tour[0]]}") 
            print(f"Best Tour Distance from GA: {best_ga_distance:.2f}")
            
            plot_ga_evolution(ga_history, problem_name=os.path.splitext(PROBLEM_FILE_NAME)[0], 
                              title_suffix=f"(Seeded by Christofides Cost {christofides_tour_cost:.0f})")
            
            if cities_coords_for_ga is not None and len(cities_coords_for_ga) > 0:
                plot_final_tour(cities_coords_for_ga, best_ga_tour, best_ga_distance, 
                                problem_name=os.path.splitext(PROBLEM_FILE_NAME)[0])
            
            if christofides_tour_cost < float('inf'):
                improvement = ((christofides_tour_cost - best_ga_distance) / christofides_tour_cost) * 100 if christofides_tour_cost > 0 else 0
                print(f"\nComparison ({PROBLEM_FILE_NAME}):")
                print(f"  Christofides Initial Cost: {christofides_tour_cost:.2f}")
                print(f"  GA Final Best Cost:      {best_ga_distance:.2f}")
                if best_ga_distance < christofides_tour_cost:
                    print(f"  Improvement by GA:       {improvement:.2f}%")
                elif abs(best_ga_distance - christofides_tour_cost) < 1e-5: 
                     print(f"  GA found a solution with effectively the same cost as Christofides.")
                else:
                    print(f"  GA found a solution with a higher cost ({best_ga_distance - christofides_tour_cost:.2f} more). Initial Christofides was better.")
        else:
            print("Genetic Algorithm did not find a solution.")
    else:
        print("Cannot proceed because data loading failed earlier.")

    if PROBLEM_FILE_NAME == "rat195.tsp" and os.path.exists(PROBLEM_FILE_NAME):
        try:
            os.remove(PROBLEM_FILE_NAME)
            print(f"Cleaned up temporary file: {PROBLEM_FILE_NAME}")
        except OSError as e:
            print(f"Error removing temporary file {PROBLEM_FILE_NAME}: {e}")
