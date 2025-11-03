# vrptw/fitness.py

from __future__ import annotations
from typing import Iterable
import numpy as np
from vrptw.model import Instance

def fitness_penalty_from_routes(
    routes: Iterable[Iterable[int]],
    inst: Instance,
    alpha: float = 1000.0,
    beta: float = 100.0,
) -> tuple[float, float, float, float]:
    """
    Zaimplementowana funkcja z karami:
        c̃(s) = c(s) + α q(s) + β ω(s)

    gdzie:
    - c(s)      – zwykły koszt (dystans) wszystkich tras
    - q(s)      – łączne naruszenie pojemności (ile łącznie "przeładowaliśmy")
    - ω(s)      – łączne spóźnienie względem okien czasowych
    - α, β      – wagi kar

    Zwraca krotkę:
        (fitness, total_distance, total_cap_violation, total_time_violation)
    żebyś mógł sobie to też podejrzeć w run_cli.py.
    """
    df = inst.data
    dist = inst.distance
    Q = inst.Q

    total_distance = 0.0      # c(s)
    total_cap_violation = 0.0 # q(s)
    total_time_violation = 0.0 # ω(s)

    for route in routes:
        load = 0.0
        t = 0.0
        last = 0  # start w depocie (id=0)

        for nid in route:
            nid = int(nid)

            demand = float(df.demand.iloc[nid])
            ready = float(df.ready.iloc[nid])
            due = float(df.due.iloc[nid])
            service = float(df.service.iloc[nid])

            # przejazd depot/klient -> klient
            travel = dist[last, nid]
            total_distance += travel
            t += travel

            # okno czasowe: jak przyjedziemy za wcześnie to czekamy (bez kary)
            if t < ready:
                t = ready

            # spóźnienie: to jest właśnie ω(s) (sumujemy opóźnienie)
            if t > due:
                total_time_violation += (t - due)

            # obsługa
            t += service

            # pojemność: jeśli przekroczymy Q to dodajemy nadwyżkę do q(s)
            load += demand
            if load > Q:
                total_cap_violation += (load - Q)

            last = nid

        # powrót do depotu
        total_distance += dist[last, 0]

    # penalizowana jakość rozwiązania
    fitness = (
        total_distance
        + alpha * total_cap_violation
        + beta * total_time_violation
    )
    return fitness, total_distance, total_cap_violation, total_time_violation


# def fitness_from_permutation(pi: np.ndarray, inst: Instance,
#                              alpha: float = 1000.0, beta: float = 100.0):
#     """
#     Wersja pomocnicza: jeśli masz tylko permutację (giant tour),
#     możesz najpierw użyć splitowania, a potem tej samej funkcji fitness.
#     Zostawiłem to tutaj na przyszłość – teraz w run_cli robisz split osobno.
#     """
#     from vrptw.split import split_routes

#     routes, _, _ = split_routes(pi, inst, alpha=0.0, beta=0.0)
#     return fitness_penalty_from_routes(routes, inst, alpha, beta)
