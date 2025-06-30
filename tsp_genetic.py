"""
==============================================================================
|| TFG - Trabajo de Fin de Grado en Matemáticas de Carlos San Román Cazorla ||
==============================================================================
"""

"Este código ha sido realizado por Carlos San Román Cazorla"
"para el desarrollo de su Trabajo de Fin de Grado en Matemáticas."

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
    PROBLEM_FILE = #PONER ARCHIVO
    
    # Parámetros del Algoritmo Genético
    POPULATION_SIZE = 150
    GENERATIONS = 10000
    MUTATION_RATE = 0.02
    CROSSOVER_RATE = 0.9
    ELITISM_SIZE = 10 # Número de individuos élite
   
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
