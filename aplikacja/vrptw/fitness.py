import numpy as np

def fitness_penalty_from_routes(routes, df, D, Q, alpha=1000.0, beta=100.0):
    """
    Oblicza karaną jakość rozwiązania VRPTW:
    - dystans tras
    - przekroczenia pojemności
    - opóźnienia względem okien czasowych
    """
    total_distance = 0.0
    cap_violation = 0.0
    time_violation = 0.0

    for route in routes:
        load = 0.0
        t = 0.0
        last = 0

        for nid in route:
            demand = df.loc[nid, 'demand']
            ready = df.loc[nid, 'ready']
            due = df.loc[nid, 'due']
            service = df.loc[nid, 'service']

            travel = D[last, nid]
            total_distance += travel
            t += travel

            if t < ready:
                t = ready
            if t > due:
                time_violation += (t - due)

            t += service
            load += demand

            if load > Q:
                cap_violation += (load - Q)

            last = nid

        total_distance += D[last, 0]  # powrót

    fitness = total_distance + alpha * cap_violation + beta * time_violation
    return fitness, total_distance, cap_violation, time_violation
