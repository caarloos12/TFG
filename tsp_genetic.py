import numpy as np
import random
import matplotlib.pyplot as plt

# --- 1. Cargador de Problemas y Funciones de Utilidad ---

def load_tsplib_problem(filepath):
    """Carga las coordenadas de los nodos desde un archivo en formato TSPLIB."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    coords = []
    in_coord_section = False
    for line in lines:
        line = line.strip()
        if line.startswith("NODE_COORD_SECTION"):
            in_coord_section = True
            continue
        elif line.startswith("EOF"):
            break
        
        if in_coord_section:
            parts = line.split()
            coords.append([float(parts[1]), float(parts[2])])
            
    print(f"Datos TSPLIB cargados desde {filepath}: {len(coords)} ciudades.")
    return np.array(coords)

def calculate_distance_matrix(cities):
    """Calcula la matriz de distancias euclidianas entre todas las ciudades."""
    num_cities = len(cities)
    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(i, num_cities):
            dist = np.linalg.norm(cities[i] - cities[j])
            dist_matrix[i, j] = dist_matrix[j, i] = dist
    return dist_matrix

# --- 2. Clase Principal del Algoritmo Genético ---

class GeneticAlgorithmTSP:
    def __init__(self, cities, population_size=100, mutation_rate=0.01, crossover_rate=0.9, elitism_size=5):
        self.cities = cities
        self.dist_matrix = calculate_distance_matrix(cities)
        self.num_cities = len(cities)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_size = elitism_size
        
        # Inicializa la población
        self.population = self._initialize_population()

    def _initialize_population(self):
        """Crea una población inicial de rutas aleatorias."""
        population = []
        base_tour = list(range(self.num_cities))
        for _ in range(self.population_size):
            # Baraja la ruta para crear un individuo aleatorio
            tour = random.sample(base_tour, self.num_cities)
            population.append(tour)
        return population

    def _calculate_tour_distance(self, tour):
        """Calcula la distancia total de un tour."""
        total_distance = 0
        for i in range(self.num_cities):
            start_city = tour[i]
            end_city = tour[(i + 1) % self.num_cities] # Vuelve al inicio
            total_distance += self.dist_matrix[start_city, end_city]
        return total_distance

    def _calculate_fitness(self, tour):
        """Calcula el fitness de un tour (inverso de la distancia)."""
        distance = self._calculate_tour_distance(tour)
        # Se añade un valor pequeño para evitar la división por cero
        return 1 / (distance + 1e-6)

    def _selection(self, ranked_population):
        """Selección por torneo para elegir un padre."""
        # Elige 3 individuos al azar (tamaño del torneo)
        tournament_size = 3
        selection = random.sample(ranked_population, tournament_size)
        # El individuo con mayor fitness (menor distancia) gana
        selection.sort(key=lambda x: x[1], reverse=True) # Ordena por fitness descendente
        return selection[0][0] # Devuelve el tour del ganador

    def _crossover(self, parent1, parent2):
        """Cruce Ordenado (Ordered Crossover - OX) para crear un hijo."""
        child = [None] * self.num_cities
        
        # Elige un subconjunto aleatorio del primer padre
        start, end = sorted(random.sample(range(self.num_cities), 2))
        
        # Copia el subconjunto al hijo
        child[start:end] = parent1[start:end]
        
        # Rellena los genes restantes del segundo padre
        pointer = end
        for city in parent2:
            if city not in child:
                if pointer >= self.num_cities:
                    pointer = 0
                child[pointer] = city
                pointer += 1
        return child

    def _mutate(self, tour):
        """Mutación por intercambio (swap mutation)."""
        # Elige dos ciudades aleatorias y las intercambia
        idx1, idx2 = random.sample(range(self.num_cities), 2)
        tour[idx1], tour[idx2] = tour[idx2], tour[idx1]
        return tour

    def run(self, generations):
        """Ejecuta el ciclo evolutivo del algoritmo genético."""
        print("Iniciando evolución con Algoritmo Genético...")
        history = {'best_distance': [], 'avg_distance': []}

        for gen in range(generations):
            # 1. Evaluar la población actual
            ranked_population = []
            total_distance = 0
            for tour in self.population:
                distance = self._calculate_tour_distance(tour)
                fitness = 1 / (distance + 1e-6)
                ranked_population.append((tour, fitness, distance))
                total_distance += distance
            
            # Ordenar por fitness para encontrar el mejor
            ranked_population.sort(key=lambda x: x[1], reverse=True)
            
            # Guardar estadísticas
            best_tour_data = ranked_population[0]
            history['best_distance'].append(best_tour_data[2])
            history['avg_distance'].append(total_distance / self.population_size)

            if (gen + 1) % 50 == 0:
                print(f"  Generación {gen+1:4}/{generations} | Mejor Distancia: {best_tour_data[2]:.2f} | Distancia Promedio: {history['avg_distance'][-1]:.2f}")

            # 2. Crear la nueva generación
            new_population = []
            
            # Elitismo: pasar los mejores individuos directamente
            for i in range(self.elitism_size):
                new_population.append(ranked_population[i][0])
            
            # Crear el resto de la población mediante cruce y mutación
            while len(new_population) < self.population_size:
                parent1 = self._selection(ranked_population)
                parent2 = self._selection(ranked_population)
                
                if random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1 # Si no hay cruce, el hijo es una copia del padre
                
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)
                
                new_population.append(child)
            
            self.population = new_population

        best_overall_tour = ranked_population[0][0]
        best_overall_distance = ranked_population[0][2]
        
        return best_overall_tour, best_overall_distance, history

# --- 3. Funciones de Visualización ---

def plot_ga_evolution(history):
    """Grafica la evolución de la mejor y la distancia promedio por generación."""
    plt.figure(figsize=(12, 6))
    plt.plot(history['best_distance'], label="Mejor Distancia", color="dodgerblue", linewidth=2)
    plt.plot(history['avg_distance'], label="Distancia Promedio", color="tomato", linestyle="--")
    plt.title("Evolución del Algoritmo Genético", fontsize=16)
    plt.xlabel("Generación", fontsize=12)
    plt.ylabel("Distancia Total del Tour", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_final_tour(cities, best_tour, best_distance):
    """Grafica el mejor recorrido encontrado."""
    plt.figure(figsize=(10, 10))
    ordered_cities = np.array([cities[i] for i in best_tour + [best_tour[0]]])
    
    plt.plot(ordered_cities[:, 0], ordered_cities[:, 1], 'o-', color='navy', markersize=4, zorder=2)
    plt.scatter(cities[:, 0], cities[:, 1], c='red', s=15, zorder=3)
    
    plt.title(f"Mejor Tour Encontrado por el AG\nDistancia Total: {best_distance:.2f}", fontsize=16)
    plt.xlabel("Coordenada X", fontsize=12)
    plt.ylabel("Coordenada Y", fontsize=12)
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


# --- 4. Ejecución Principal ---
if __name__ == "__main__":
    # --- Parámetros ---
    PROBLEM_FILE = "rat195.tsp"
    
    # Parámetros del Algoritmo Genético
    POPULATION_SIZE = 150
    GENERATIONS = 10000
    MUTATION_RATE = 0.02
    CROSSOVER_RATE = 0.9
    ELITISM_SIZE = 10 # Número de individuos élite
    
    # --- Cargar datos ---
    # (El código del problema rat195 se incluye aquí para que sea autoejecutable)
    file_content = """
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
    with open(PROBLEM_FILE, "w") as f:
        # Pega el contenido completo de tu archivo rat195.tsp aquí
        # Para brevedad, lo omito, pero es necesario para que el script funcione
        f.write(file_content.replace("... (el resto de tu archivo rat195 va aquí) ...", "2 17 12\n3 23 9\n4 34 11\n5 47 11\n6 54 12\n7 66 16\n8 75 7\n9 86 6\n10 94 8\n11 107 9\n12 115 14\n13 123 15\n14 3 32\n15 15 32\n16 26 34\n17 33 34\n18 42 34\n19 53 25\n20 64 32\n21 74 32\n22 85 34\n23 95 28\n24 104 25\n25 113 31\n26 125 34\n27 3 48\n28 15 46\n29 26 50\n30 36 54\n31 48 50\n32 54 46\n33 64 54\n34 75 44\n35 88 49\n36 98 50\n37 103 54\n38 115 47\n39 127 49\n40 6 75\n41 15 75\n42 27 73\n43 36 73\n44 47 68\n45 54 72\n46 66 68\n47 74 67\n48 85 65\n49 94 74\n50 107 65\n51 117 65\n52 125 68\n53 6 84\n54 13 95\n55 25 94\n56 37 84\n57 47 87\n58 53 95\n59 63 86\n60 77 93\n61 83 89\n62 94 95\n63 103 92\n64 115 95\n65 123 93\n66 7 114\n67 15 111\n68 24 112\n69 36 108\n70 43 112\n71 56 105\n72 64 112\n73 73 112\n74 86 107\n75 98 108\n76 104 113\n77 117 115\n78 126 109\n79 6 127\n80 17 125\n81 27 134\n82 35 126\n83 44 131\n84 54 132\n85 63 124\n86 77 127\n87 82 134\n88 96 128\n89 103 126\n90 116 130\n91 126 134\n92 7 152\n93 16 147\n94 24 153\n95 35 151\n96 45 154\n97 55 146\n98 63 155\n99 75 151\n100 87 154\n101 93 156\n102 104 151\n103 117 153\n104 127 148\n105 3 164\n106 16 172\n107 25 165\n108 35 175\n109 44 169\n110 53 174\n111 64 168\n112 76 171\n113 87 173\n114 95 174\n115 106 168\n116 114 169\n117 125 169\n118 3 190\n119 16 188\n120 25 195\n121 37 186\n122 44 189\n123 54 194\n124 66 192\n125 77 192\n126 85 188\n127 93 185\n128 106 192\n129 113 193\n130 125 195\n131 5 207\n132 15 213\n133 24 209\n134 33 214\n135 43 206\n136 53 211\n137 64 213\n138 74 212\n139 84 212\n140 94 209\n141 104 215\n142 115 206\n143 127 209\n144 6 229\n145 13 227\n146 26 235\n147 34 225\n148 43 227\n149 55 225\n150 67 229\n151 75 234\n152 87 230\n153 95 235\n154 105 228\n155 117 225\n156 127 230\n157 6 249\n158 15 246\n159 26 255\n160 33 246\n161 47 248\n162 58 252\n163 65 248\n164 73 247\n165 87 249\n166 94 245\n167 104 256\n168 113 246\n169 125 253\n170 5 266\n171 16 274\n172 24 267\n173 37 266\n174 45 267\n175 54 266\n176 67 267\n177 74 265\n178 87 264\n179 95 271\n180 106 264\n181 116 271\n182 127 273\n183 7 287\n184 17 294\n185 23 287\n186 33 284\n187 43 288\n188 53 295\n189 67 288\n190 73 286\n191 87 293\n192 94 284\n193 104 291\n194 114 294\n"))

    cities_coords = load_tsplib_problem(PROBLEM_FILE)
    
    # --- Ejecutar el algoritmo ---
    ga = GeneticAlgorithmTSP(
        cities=cities_coords,
        population_size=POPULATION_SIZE,
        mutation_rate=MUTATION_RATE,
        crossover_rate=CROSSOVER_RATE,
        elitism_size=ELITISM_SIZE
    )
    
    best_tour, best_distance, history = ga.run(generations=GENERATIONS)
    
    # --- Mostrar resultados ---
    print("\n--- Resultados Finales del Algoritmo Genético ---")
    print(f"Mejor distancia encontrada: {best_distance:.2f}")
    print(f"Mejor ruta (primeros 20 nodos): {best_tour[:20]}...")
    
    # --- Graficar ---
    plot_ga_evolution(history)
    plot_final_tour(cities_coords, best_tour, best_distance)