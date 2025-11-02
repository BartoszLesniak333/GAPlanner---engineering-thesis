# fitness_penalty_exact.py
import numpy as np

def fitness_penalty_from_routes(routes, inst, alpha=1000.0, beta=100.0):
    """
    Zwraca wartość: c~(s) = c(s) + α q(s) + β ω(s)
    - c(s): łączny dystans
    - q(s): suma nadmiaru ładunku ponad Q na trasach (soft capacity)
    - ω(s): suma spóźnień względem due (soft time windows; waiting bez kary)
    """
    df, dist, Q = inst.data, inst.distance, inst.Q

    total_dist = 0.0
    total_overload = 0.0  # q(s)
    total_lateness = 0.0  # ω(s)

    for r in routes:
        last = 0  # depot (index 0)
        load = 0
        t = 0.0

        # pętla po klientach w trasie
        for nid in r:
            demand = df.demand.iloc[nid]
            ready  = df.ready.iloc[nid]
            due    = df.due.iloc[nid]
            serv   = df.service.iloc[nid]

            # dystans i czas jazdy
            total_dist += dist[last, nid]
            t += dist[last, nid]

            # waiting bez kary; spóźnienie karzemy
            if t < ready:
                t = ready
            if t > due:
                total_lateness += (t - due)

            # serwis i aktualizacja stanu
            t += serv
            load += demand
            last = nid

        # powrót do depotu
        total_dist += dist[last, 0]

        # nadmiar ładunku na trasie (soft capacity)
        total_overload += max(0, load - Q)

    fitness = total_dist + alpha * total_overload + beta * total_lateness
    return {
        "fitness": fitness,
        "distance": total_dist,
        "q_overload": total_overload,
        "omega_lateness": total_lateness,
        "alpha": alpha, "beta": beta
    }
