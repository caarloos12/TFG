import numpy as np

def knapsack_fptas(items, capacity, epsilon):
    """
    FPTAS para el problema de la mochila 0/1
    :param items: Lista de tuplas (valor, peso)
    :param capacity: Capacidad máxima de la mochila
    :param epsilon: Parámetro de aproximación
    :return: Valor máximo aproximado y los elementos seleccionados
    """
    n = len(items)
    max_value = max(value for value, _ in items)
    
    # Escalado de valores
    scaling_factor = (epsilon * max_value) / n
    scaled_items = [(int(value / scaling_factor), weight) for value, weight in items]
    
    # Programación dinámica con los valores escalados
    V = sum(value for value, _ in scaled_items)
    dp = np.full((n + 1, V + 1), float('inf'))
    dp[0][0] = 0
    
    for i in range(1, n + 1):
        value, weight = scaled_items[i - 1]
        for v in range(V + 1):
            dp[i][v] = dp[i - 1][v]
            if v >= value:
                dp[i][v] = min(dp[i][v], dp[i - 1][v - value] + weight)
    
    # Encontrar el mayor valor posible dentro de la capacidad
    best_value = max(v for v in range(V + 1) if dp[n][v] <= capacity)
    
    # Reconstrucción de la solución
    selected_items = []
    w = capacity
    for i in range(n, 0, -1):
        if best_value > 0 and dp[i][best_value] != dp[i - 1][best_value]:
            selected_items.append(i - 1)
            best_value -= scaled_items[i - 1][0]
    
    # Devolver el valor original y los elementos seleccionados
    total_value = sum(items[i][0] for i in selected_items)
    return total_value, selected_items

# Ejemplo de uso
items = [(60, 10), (100, 20), (120, 30)]  # (valor, peso)
capacity = 50
epsilon = 0.1

max_value, selected_items = knapsack_fptas(items, capacity, epsilon)
print("Valor aproximado máximo:", max_value)
print("Índices de los elementos seleccionados:", selected_items)
