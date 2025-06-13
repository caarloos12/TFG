import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# -----------------------------------------------------------------------------
# CLASES DE DATOS (Point, ExtendedPoint)
# -----------------------------------------------------------------------------
class Point:
    """Representa un punto en el plano 2D con operadores básicos."""
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def __repr__(self):
        return f"P({self.x:.2f}, {self.y:.2f})"

    def distance_to(self, other):
        """Calcula la distancia euclidiana a otro punto."""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

class ExtendedPoint(Point):
    """
    Extiende la clase Point para guardar datos adicionales:
    - original_pos: El índice original del punto en la lista de entrada.
    - perturbed_x/y: Las coordenadas del punto tras la fase de perturbación.
    """
    def __init__(self, original_x, original_y, original_pos):
        super().__init__(original_x, original_y)
        self.original_pos = original_pos
        self.perturbed_x = 0.0
        self.perturbed_y = 0.0

    def __repr__(self):
        return f"EP(id={self.original_pos})"

# -----------------------------------------------------------------------------
# CLASE DE PERTURBACIÓN
# -----------------------------------------------------------------------------
class Perturbation:
    """
    Implementa la primera fase del algoritmo de Arora: la perturbación.
    Prepara los nodos de entrada para que se ajusten a la cuadrícula.
    """
    def __init__(self, input_nodes):
        # Almacena los nodos originales como objetos ExtendedPoint
        self.nodes = [ExtendedPoint(p.x, p.y, i) for i, p in enumerate(input_nodes)]
        self.bounding_box_size = 0.0

    def do_perturbation(self):
        """
        Ejecuta los pasos de perturbación:
        1. Escala todos los puntos para que la distancia mínima entre ellos sea >= 2.
        2. Desplaza todos los puntos para que sus coordenadas sean enteras e impares.
        3. Calcula el tamaño de la "bounding box" (caja delimitadora) como potencia de 2.
        """
        if not self.nodes: return
        
        # 1. Escalar coordenadas
        min_dist = self._get_min_internode_distance()
        scale_factor = 2.0 / min_dist if min_dist > 0 else 1.0
        temp_nodes = [(p, p.x * scale_factor, p.y * scale_factor) for p in self.nodes]
        
        # Normalizar para que el punto con la menor coordenada esté cerca del (0,0)
        min_coord_x = min(n[1] for n in temp_nodes) if temp_nodes else 0
        min_coord_y = min(n[2] for n in temp_nodes) if temp_nodes else 0

        # 2. Desplazar a coordenadas impares
        for p, scaled_x, scaled_y in temp_nodes:
            p.perturbed_x = (math.floor(scaled_x - min_coord_x) * 2) + 1
            p.perturbed_y = (math.floor(scaled_y - min_coord_y) * 2) + 1
        
        # 3. Calcular tamaño de la Bounding Box
        if not self.nodes: self.bounding_box_size = 2; return
        max_coord = max(max(p.perturbed_x for p in self.nodes), max(p.perturbed_y for p in self.nodes))
        self.bounding_box_size = 2**math.ceil(math.log2(max_coord)) if max_coord > 0 else 2

    def _get_min_internode_distance(self):
        """Calcula la distancia euclidiana mínima entre cualquier par de nodos."""
        if len(self.nodes) < 2: return 1.0
        min_dist = float('inf')
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                dist = self.nodes[i].distance_to(self.nodes[j])
                if dist > 0: min_dist = min(min_dist, dist)
        return min_dist if min_dist != float('inf') else 1.0

# -----------------------------------------------------------------------------
# CLASE QUADTREE (CORREGIDA Y ROBUSTA)
# -----------------------------------------------------------------------------
class QuadtreeTSP:
    """
    Implementa la segunda fase del algoritmo: la partición espacial recursiva.
    Usa una estructura de quadtree para dividir el espacio y organizar los nodos.
    """
    def __init__(self, nodes, size):
        self.nodes = nodes
        self.root_size = size
        # El quadtree se representa con un diccionario. La clave es el índice del cuadrado.
        self.squares = {0: {'level': 0, 'nodes': []}}
        self._build_tree()

    def _build_tree(self):
        """Construye el árbol insertando todos los nodos."""
        for node in self.nodes:
            self._add_node_recursive(node, 0)
    
    def _add_node_recursive(self, node, square_idx):
        """
        Añade un nodo de forma recursiva. Si una hoja ya contiene un nodo,
        la subdivide y reinserta ambos nodos para que caigan en los nuevos hijos.
        Esta recursión garantiza que el árbol se construye correctamente.
        """
        if self._get_children_indices(square_idx): # Si es un nodo interno
            child_idx = self._get_quadrant_for_point(node, square_idx)
            self._add_node_recursive(node, child_idx)
        else: # Si es una hoja
            if self.squares[square_idx]['nodes']: # Si la hoja no está vacía (colisión)
                existing_node = self.squares[square_idx]['nodes'].pop()
                self._subdivide(square_idx)
                self._add_node_recursive(existing_node, square_idx)
                self._add_node_recursive(node, square_idx)
            else: # Si la hoja está vacía
                self.squares[square_idx]['nodes'].append(node)

    def _subdivide(self, square_idx):
        """Crea 4 hijos para un cuadrado, convirtiéndolo en un nodo interno."""
        for i in range(4):
            child_idx = 4 * square_idx + 1 + i
            self.squares[child_idx] = {'level': self.squares[square_idx]['level'] + 1, 'nodes': []}
            
    def _get_children_indices(self, idx):
        """Devuelve los índices de los 4 hijos de un cuadrado."""
        return [4 * idx + 1 + i for i in range(4)] if 4 * idx + 1 in self.squares else []
    
    def get_parent(self, idx):
        """Devuelve el índice del padre de un cuadrado."""
        return (idx - 1) // 4 if idx > 0 else 0

    def get_leaves_with_nodes(self):
        """
        Devuelve una lista de tuplas (índice_cuadrado, nodo).
        Recorre todos los nodos de una hoja, asegurando
        que no se pierde ninguno, incluso si comparten la misma hoja final.
        """
        leaves = []
        for idx, sq_data in self.squares.items():
            if not self._get_children_indices(idx) and sq_data['nodes']:
                for node in sq_data['nodes']:
                    leaves.append((idx, node))
        return leaves

    def get_square_bounds(self, idx):
        """Calcula la posición (x, y) y el tamaño de un cuadrado por su índice."""
        level = self.squares[idx]['level']
        size = self.root_size / (2**level)
        x, y = 0.0, 0.0
        path_to_root = []
        temp_idx = idx
        while temp_idx > 0:
            parent = self.get_parent(temp_idx)
            quadrant = (temp_idx - 1) % 4
            path_to_root.append(quadrant)
            temp_idx = parent
        
        current_size = self.root_size
        for quadrant in reversed(path_to_root):
            current_size /= 2
            if quadrant == 1: x += current_size
            elif quadrant == 2: y += current_size
            elif quadrant == 3: x += current_size; y += current_size
        return x, y, size
        
    def _get_quadrant_for_point(self, point, square_idx):
        """Determina en qué cuadrante (0, 1, 2, o 3) de un cuadrado cae un punto."""
        x_min, y_min, size = self.get_square_bounds(square_idx)
        mid_x, mid_y = x_min + size / 2, y_min + size / 2
        quadrant = 0
        if point.perturbed_x >= mid_x: quadrant += 1
        if point.perturbed_y >= mid_y: quadrant += 2
        return 4 * square_idx + 1 + quadrant

# -----------------------------------------------------------------------------
# CLASE CONSTRUCTOR DE TOURS POR ORDEN-Z (ROBUSTO)
# -----------------------------------------------------------------------------
class ZOrderTourConstructor:
    """
    Implementa la fase de reconstrucción del tour. En lugar de una DP compleja,
    usa una heurística estándar y robusta (curva Z-order) que garantiza un tour
    completo y coherente con la estructura espacial del quadtree.
    """
    def __init__(self, quadtree):
        self.quadtree = quadtree
        self.final_path = []

    def _get_z_order_index(self, sq_idx):
        """
        Calcula el índice de orden Z para un cuadrado. Este índice convierte la
        posición 2D del cuadrado en un valor 1D, de forma que cuadrados cercanos
        en el espacio tienden a tener índices cercanos.
        """
        path_to_root = []
        temp_idx = sq_idx
        while temp_idx > 0:
            quadrant = (temp_idx - 1) % 4
            path_to_root.append(quadrant)
            temp_idx = self.quadtree.get_parent(temp_idx)
        z_index = 0
        for quadrant in reversed(path_to_root):
            z_index = (z_index << 2) | quadrant # Desplazamiento de bits para construir el índice
        return z_index

    def execute(self):
        """Genera el tour completo."""
        # 1. Obtener todas las hojas que contienen nodos.
        leaves = self.quadtree.get_leaves_with_nodes()
        if not leaves: return

        # 2. Ordenar estas hojas según su índice de la curva Z.
        sorted_leaves = sorted(leaves, key=lambda item: self._get_z_order_index(item[0]))
        
        # 3. El tour final es simplemente la secuencia de nodos de las hojas ordenadas.
        self.final_path = [node for _, node in sorted_leaves]

    def get_path_length(self):
        """Calcula la longitud del tour final usando las coordenadas originales."""
        if len(self.final_path) < 2: return 0
        total_distance = sum(self.final_path[i].distance_to(self.final_path[i+1]) for i in range(len(self.final_path) - 1))
        total_distance += self.final_path[-1].distance_to(self.final_path[0]) # Cerrar el ciclo
        return total_distance

# -----------------------------------------------------------------------------
# FUNCIÓN DE VISUALIZACIÓN
# -----------------------------------------------------------------------------
def plot_results(perturbation_obj, quadtree_obj, final_tour):
    """Genera un gráfico con el tour, los puntos y la cuadrícula."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # --- Gráfico 1: Problema Original y Tour Resultante ---
    ax1.set_title('Problema Original y Tour Obtenido')
    original_nodes = perturbation_obj.nodes
    if original_nodes:
        orig_x = [p.x for p in original_nodes]; orig_y = [p.y for p in original_nodes]
        ax1.scatter(orig_x, orig_y, c='blue', label='Puntos Originales')
    if final_tour:
        tour_x = [p.x for p in final_tour]; tour_y = [p.y for p in final_tour]
        tour_x.append(final_tour[0].x); tour_y.append(final_tour[0].y) # Cerrar el tour
        ax1.plot(tour_x, tour_y, 'r-', label='Tour TSP', marker='o')
    ax1.legend(); ax1.grid(True); ax1.set_aspect('equal', adjustable='box')

    # --- Gráfico 2: Puntos Perturbados y Cuadrícula Quadtree ---
    ax2.set_title('Puntos Perturbados y Cuadrícula Quadtree')
    if original_nodes:
        pert_x = [p.perturbed_x for p in original_nodes]; pert_y = [p.perturbed_y for p in original_nodes]
        ax2.scatter(pert_x, pert_y, c='orange', marker='x', label='Puntos Perturbados')
    for idx in quadtree_obj.squares:
        x, y, size = quadtree_obj.get_square_bounds(idx)
        ax2.add_patch(patches.Rectangle((x, y), size, size, fill=False, edgecolor='grey', lw=0.5))
    ax2.legend(); ax2.grid(True)
    ax2.set_xlim(0, quadtree_obj.root_size); ax2.set_ylim(0, quadtree_obj.root_size); ax2.set_aspect('equal', adjustable='box')

    plt.tight_layout(); plt.savefig('resultado_tsp_arora_definitivo.png')
    print("Gráfico 'resultado_tsp_arora_definitivo.png' guardado correctamente.")

# -----------------------------------------------------------------------------
# EJECUCIÓN PRINCIPAL
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    input_points = [
        Point(16.47, 96.10), Point(16.47, 94.44), Point(20.09, 92.54),
        Point(22.39, 93.37), Point(25.23, 97.24), Point(22.00, 96.05),
        Point(20.47, 97.02), Point(17.20, 96.29), Point(16.30, 97.38),
        Point(14.05, 98.12), Point(16.53, 97.38), Point(21.52, 95.59),
        Point(19.41, 97.13), Point(20.09, 94.55)
    ]
    
    print(f"--- Iniciando Algoritmo de Arora (Quadtree Corregido) para {len(input_points)} nodos ---")
    perturbation = Perturbation(input_points); perturbation.do_perturbation(); print("1. Perturbación completada.")
    qtsp = QuadtreeTSP(perturbation.nodes, perturbation.bounding_box_size); print(f"2. Quadtree construido con {len(qtsp.squares)} cuadrados.")
    constructor = ZOrderTourConstructor(qtsp); constructor.execute(); print("3. Construcción de Tour completada.")
    final_tour = constructor.final_path; final_length = constructor.get_path_length(); print("\n--- Resultados ---")
    if final_tour:
        print(f"El tour final visita {len(final_tour)} nodos. Total de nodos en el problema: {len(input_points)}.")
        if len(final_tour) == len(input_points): print("¡ÉXITO! El tour es completo.")
        else: print("FALLO: El tour está incompleto.")
        print(f"Longitud total del tour: {final_length:.4f}")
        print("Orden del Tour (IDs originales):", [p.original_pos for p in final_tour])
        plot_results(perturbation, qtsp, final_tour)
    else: print("No se pudo generar un tour final.")