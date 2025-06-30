"""
==============================================================================
|| TFG - Trabajo de Fin de Grado en Matemáticas de Carlos San Román Cazorla ||
==============================================================================
"""

"Este código ha sido realizado por Carlos San Román Cazorla"
"para el desarrollo de su Trabajo de Fin de Grado en Matemáticas."

def fptas_knapsack(items, capacity, epsilon):
"""
    Implementación del Algoritmo 6 (página 41), un Esquema de Aproximación
    Totalmente Polinomial en Tiempo (FPTAS) para el problema de la mochila 0/1.

    Args:
        items (list): Lista de tuplas (beneficio, peso).
        capacity (int): Capacidad máxima de la mochila.
        epsilon (float): Factor de error.

    Returns:
        tuple: (beneficio_total, peso_total, lista_de_objetos_seleccionados).
    """
    if not items or capacity <= 0 or epsilon <= 0:
        return 0, 0, []

    num_items = len(items)
    
    # --- Paso 1: Definir K ---
    max_profit = 0
    for profit, _ in items:
        if profit > max_profit:
            max_profit = profit
            
    if max_profit == 0:
        return 0, 0, []

    K = (epsilon * max_profit) / num_items

    # --- Pasos 2 y 3: Escalar los beneficios ---
    scaled_items = []
    for profit, weight in items:
        # Ignorar objetos con peso mayor a la capacidad
        if weight <= capacity:
            scaled_items.append({
                'scaled_profit': int(profit / K) if K > 0 else profit,
                'original_profit': profit,
                'weight': weight
            })
    
    num_scaled_items = len(scaled_items)
    max_possible_scaled_profit = sum(item['scaled_profit'] for item in scaled_items)

    # --- Paso 4: Programación Dinámica con tabla 2D ---
    # dp[i][p] = peso mínimo para un beneficio 'p' usando los primeros 'i' objetos.
    dp = [[float('inf')] * (max_possible_scaled_profit + 1) for _ in range(num_scaled_items + 1)]

    # Caso base: con 0 objetos, solo se puede lograr un beneficio de 0 con un peso de 0.
    for i in range(num_scaled_items + 1):
        dp[i][0] = 0

    # Llenamos la tabla de DP
    for i in range(1, num_scaled_items + 1):
        item = scaled_items[i-1]
        p_scaled = item['scaled_profit']
        w = item['weight']
        
        for p in range(1, max_possible_scaled_profit + 1):
            # Opción 1: No incluir el objeto i. El peso es el mismo que con i-1 objetos.
            weight_without_item = dp[i-1][p]
            
            # Opción 2: Incluir el objeto i.
            weight_with_item = float('inf')
            if p_scaled <= p:
                if dp[i-1][p - p_scaled] != float('inf'):
                    weight_with_item = w + dp[i-1][p - p_scaled]
            
            # Nos quedamos con la opción que nos dé el menor peso.
            dp[i][p] = min(weight_without_item, weight_with_item)

    # --- Encontrar la mejor solución ---
    best_scaled_profit = 0
    for p in range(max_possible_scaled_profit, -1, -1):
        # Si el peso para lograr el beneficio 'p' no excede la capacidad...
        if dp[num_scaled_items][p] <= capacity:
            best_scaled_profit = p
            break
            
    # --- Paso 5: Backtracking para encontrar los objetos ---
    solution_items = []
    p = best_scaled_profit
    
    for i in range(num_scaled_items, 0, -1):
        # Si el peso es diferente al de la fila anterior, el objeto i fue incluido.
        if dp[i][p] != dp[i-1][p]:
            item = scaled_items[i-1]
            solution_items.append((item['original_profit'], item['weight']))
            p -= item['scaled_profit']
            
    total_profit = sum(item[0] for item in solution_items)
    total_weight = sum(item[1] for item in solution_items)

    return total_profit, total_weight, solution_items
