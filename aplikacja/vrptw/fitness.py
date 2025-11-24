import numpy as np

def fitness_penalty_from_routes(
    routes,
    df,
    D,
    Q,
    alpha: float = 1000.0,
    beta: float = 100.0,
    max_vehicles: int | None = None,
):
    """
    Funkcja dopasowania dla VRPTW:
    - minimalizujemy łączny dystans,
    - pojemność i okna czasowe karane są współczynnikami alpha i beta,
    - jeśli liczba tras > max_vehicles → fitness = nieskończoność (rozwiązanie niedozwolone).
    """

    total_distance = 0.0          # całkowity dystans
    cap_violation = 0.0           # suma przekroczeń ładunku
    time_violation = 0.0          # suma spóźnień

    for route in routes:
        load = 0.0                # aktualny ładunek
        t = 0.0                   # czas na zegarze
        last = 0                  # poprzedni wierzchołek (start z depot)

        for nid in route:
            demand = df.loc[nid, "demand"]
            ready = df.loc[nid, "ready"]
            due = df.loc[nid, "due"]
            service = df.loc[nid, "service"]

            # przejazd do klienta
            t += D[last, nid]
            total_distance += D[last, nid]

            # okno czasowe: oczekiwanie / spóźnienie
            if t < ready:
                t = ready
            if t > due:
                time_violation += (t - due)

            # obsługa klienta
            t += service
            load += demand

            # pojemność
            if load > Q:
                cap_violation += (load - Q)

            last = nid

        # powrót do depot
        total_distance += D[last, 0]

    # twardy limit liczby pojazdów (tras)
    if max_vehicles is not None and len(routes) > max_vehicles:
        fitness = np.inf
    else:
        fitness = total_distance + alpha * cap_violation + beta * time_violation

    return fitness, total_distance, cap_violation, time_violation
