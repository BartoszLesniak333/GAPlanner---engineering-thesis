import numpy as np

def fitness_penalty_from_routes(routes, df, D, Q, alpha=1000.0, beta=100.0):
    """
    Oblicza karaną jakość rozwiązania VRPTW:
    - dystans tras
    - przekroczenia pojemności
    - opóźnienia względem okien czasowych
    """
    total_distance = 0.0          # całkowity dsytans
    cap_violation = 0.0           # suma przekroczeń ładunku
    time_violation = 0.0          # suma spóźnien 

    for route in routes:
        load = 0.0                #aktualny ładunek
        t = 0.0                   #czas na zegarze
        last = 0                  #skąd jedziemy

        for nid in route:
            demand = df.loc[nid, 'demand']
            ready = df.loc[nid, 'ready']
            due = df.loc[nid, 'due']
            service = df.loc[nid, 'service']

            travel = D[last, nid]    # ruszamy z last do nid

            # aktualizacja dystansu i czasu
            total_distance += travel
            t += travel

            # przed czasem - czekasz
            if t < ready:
                t = ready

            # po czasie - liczysz spoznienie
            if t > due:
                time_violation += (t - due)

            # aktualizacja czasu i załadunku
            t += service
            load += demand

            # jezeli pojemnosc przekroczona to zapisuje sie nadwyzka
            if load > Q:
                cap_violation += (load - Q)

            last = nid       # aktualizacja wezła

        total_distance += D[last, 0]  # powrót do depot 

    fitness = total_distance + alpha * cap_violation + beta * time_violation 
    return fitness, total_distance, cap_violation, time_violation
