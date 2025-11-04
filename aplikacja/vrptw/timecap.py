# vrptw/timecap.py
from __future__ import annotations
from typing import Iterable
from vrptw.model import Instance

def travel_time(inst: Instance, i: int, j: int) -> float:
    """Czas/przejazd między wierzchołkami i, j (teraz = dystans)."""
    return float(inst.distance[i, j])

def eval_route(
    route: Iterable[int],
    inst: Instance,
    alpha: float = 0.0,
    beta: float = 0.0,
) -> tuple[float, float, float, float]:
    """
    Ewaluacja jednej trasy:
    - zwraca: (cost, dist, cap_violation, time_violation)
    - cost = dist + α * cap + β * time
    """
    df = inst.data
    dist_mat = inst.distance
    Q = inst.Q

    total_dist = 0.0
    cap_violation = 0.0
    time_violation = 0.0

    load = 0.0
    t = 0.0
    last = 0  # start w depocie

    for nid in route:
        nid = int(nid)
        demand = float(df.demand.iloc[nid])
        ready = float(df.ready.iloc[nid])
        due = float(df.due.iloc[nid])
        service = float(df.service.iloc[nid])

        # przejazd
        travel = dist_mat[last, nid]
        total_dist += travel
        t += travel

        # okno czasowe
        if t < ready:
            t = ready
        if t > due:
            time_violation += (t - due)

        # serwis
        t += service

        # pojemność
        load += demand
        if load > Q:
            cap_violation += (load - Q)

        last = nid

    # powrót do depotu
    total_dist += dist_mat[last, 0]

    cost = total_dist + alpha * cap_violation + beta * time_violation
    return cost, total_dist, cap_violation, time_violation


def eval_solution(
    routes: Iterable[Iterable[int]],
    inst: Instance,
    alpha: float = 0.0,
    beta: float = 0.0,
) -> tuple[float, float, float, float]:
    """
    Suma po wszystkich trasach – wygodne np. dla lokalnego przeszukiwania.
    """
    total_cost = 0.0
    total_dist = 0.0
    total_cap = 0.0
    total_time = 0.0

    for r in routes:
        c, d, cv, tv = eval_route(r, inst, alpha, beta)
        total_cost += c
        total_dist += d
        total_cap += cv
        total_time += tv

    return total_cost, total_dist, total_cap, total_time
