"""
==============================================================================
|| TFG - Trabajo de Fin de Grado en Matemáticas de Carlos San Román Cazorla ||
==============================================================================
"""

"Este código ha sido realizado por Carlos San Román Cazorla"
"para el desarrollo de su Trabajo de Fin de Grado en Matemáticas."
"Mediante este código se pretende mostrar la implementación de  "
"un algoritmo greedy para resolver el problema de MIN SET COVER, "
"cuya ratio estándar es logarítmica, luego pertence a la clase Log-APX."
"El algoritmo viene descrito en el libro de Vazirani"

def greedy_set_cover(universe, subsets, costs):
    """
    Implementación del Algoritmo 2 (página 35) para resolver el problema Set Cover
    de forma voraz.

    Args:
        universe (set): El conjunto de todos los elementos que deben ser cubiertos.
        subsets (dict): Un diccionario donde las claves son los nombres de los 
                        subconjuntos y los valores son los conjuntos de elementos.
                        Ej: {'A': {1, 2, 3}, 'B': {3, 4, 5}}
        costs (dict): Un diccionario que mapea los nombres de los subconjuntos
                      a sus costes. Ej: {'A': 5, 'B': 4}

    Returns:
        tuple: Una tupla conteniendo la lista de conjuntos de la solución y
               el coste total de dicha solución. O un error si no se puede cubrir.
    """
    # C: Conjunto de elementos ya cubiertos. Inicia vacío como en el algoritmo.
    elements_covered = set()
    
    # Lista para guardar los conjuntos seleccionados en la solución.
    solution_sets = []
    
    # El algoritmo se ejecuta mientras el conjunto de elementos cubiertos (C)
    # no sea igual al universo (U).
    while elements_covered != universe:
        best_set = None
        min_ratio = float('inf')

        # Paso 3: Encontrar el conjunto más rentable en la iteración actual.
        # La rentabilidad es la ratio coste / |elementos nuevos cubiertos|.
        for set_name, set_elements in subsets.items():
            # Calcula los elementos que este conjunto cubriría y que aún no están cubiertos.
            new_elements_covered = set_elements - elements_covered
            
            # Si el conjunto no aporta elementos nuevos, no nos interesa.
            if not new_elements_covered:
                continue

            # Calcula la ratio coste-beneficio.
            ratio = costs[set_name] / len(new_elements_covered)

            # Si esta ratio es la mejor que hemos encontrado hasta ahora,
            # guardamos este conjunto como el mejor candidato.
            if ratio < min_ratio:
                min_ratio = ratio
                best_set = set_name
        
        # Si no se encontró ningún conjunto que cubra nuevos elementos,
        # significa que el universo no puede ser cubierto con los subconjuntos dados.
        if best_set is None:
            raise ValueError("No se puede cubrir el universo con los subconjuntos proporcionados.")

        # Paso 5 y 6: Escoger el mejor conjunto y actualizar los elementos cubiertos.
        solution_sets.append(best_set)
        elements_covered.update(subsets[best_set])
        
    # Calcular el coste total de la solución.
    total_cost = sum(costs[s] for s in solution_sets)
    
    return solution_sets, total_cost


# --- Ejemplo de uso ---
if __name__ == "__main__":
    # Definimos la instancia del problema según la Definición 4.3.1.
    # Universo de elementos a cubrir.
    U = set(range(1, 11)) 

    # Colección de subconjuntos disponibles.
    S = {
        'A': {1, 2, 3, 4},
        'B': {5, 6, 7},
        'C': {1, 3, 5, 7, 9},
        'D': {2, 4, 6, 8, 10},
        'E': {8, 9, 10}
    }

    # Función de coste para cada subconjunto.
    C = {
        'A': 5,
        'B': 4,
        'C': 8,
        'D': 8,
        'E': 3
    }

    print(f"Universo a cubrir: {U}\n")
    try:
        # Ejecutamos el algoritmo
        chosen_sets, cost = greedy_set_cover(U, S, C)
        
        # Mostramos el resultado
        print(f"Los conjuntos elegidos para la solución son: {chosen_sets}")
        print(f"El coste total de la cobertura es: {cost}")
    except ValueError as e:
        print(e)
