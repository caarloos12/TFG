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

def min_set_cover(universe, subsets,costs):
    coste=0
    # Primero comprobaremos que todos los elementos del universo están cubiertos por al menos un subconjunto
    elements = set(e for s in subsets for e in s)
    if elements != universe:
        return None
    
    # Ahora, llevaremos la cuenta de los elementos que ya están cubiertos mediante un conjunto
    covered = set()

    # Aquí guardaremos los subconjuntos que vamos a seleccionar
    cover = []

    # La condición del algoritmo es que mientras no hayamos cubierto todos los elementos del universo, 
    # buscamos el siguiente subconjunto que mejor ratio de elementos no cubiertos/coste tenga
    while covered != elements:
        subset = min( (s for s in subsets if len(s - covered) > 0),  # Filtramos subconjuntos vacíos
                        key=lambda s: costs[subsets.index(s)] / len(s - covered))
        cover.append(subset)
        coste+=costs[subsets.index(subset)]
        covered |= subset
 
    return cover, coste

# Ahora, veamos un ejemplo de uso del algoritmo:

universo = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
subconjuntos = [{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10}, {2, 3, 5, 7}, {4, 6, 8, 10}]
costos = [5, 10, 3, 1, 8, 7]

cobertura, coste_total = min_set_cover(universo, subconjuntos, costos)
print("Subconjuntos seleccionados:", cobertura)
print("Coste total:", coste_total)

# Ahora, un caso más grande:
import random

# Universo de 500 elementos
universe = set(range(1, 51))

# 200 subconjuntos, cada uno con alta superposición (cubren 100-200 elementos)
subsets = [set(random.sample(list(universe), random.randint(1, 15))) for _ in range(200)]

# Costes variables
costs = [random.randint(5, 30) for _ in range(200)]

print(min_set_cover(universe, subsets, costs))

