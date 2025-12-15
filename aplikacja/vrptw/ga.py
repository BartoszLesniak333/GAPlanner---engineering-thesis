import time
import numpy as np
from vrptw.fitness import fitness_penalty_from_routes
from vrptw.split import split_routes


def run_ga(
    df,
    D,
    Q,
    pop_size: int,
    gens: int,
    pc: float,
    pm: float,
    alpha: float,
    beta: float,
    max_vehicles: int | None = None,
    time_limit_sec: float | None = None,
    gamma: float = 0.0
):
    
    n = len(df) - 1  # pomijamy depot (id=0)

    #inicjalizacja populacji

    def init_population(size: int, n_nodes: int) -> np.ndarray:
        base = np.arange(1, n_nodes + 1)
        return np.array([np.random.permutation(base) for _ in range(size)])

    # operatory
    def tournament_selection(pop: np.ndarray, fits: np.ndarray, k: int = 3) -> np.ndarray:
        idxs = np.random.choice(len(pop), size=k, replace=False)
        best_idx = idxs[np.argmin(fits[idxs])]
        return pop[best_idx]

    def ox_crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        n = len(p1)
        c1, c2 = sorted(np.random.choice(n, size=2, replace=False))
        child = -np.ones(n, dtype=int)

        # środek z p1
        child[c1:c2 + 1] = p1[c1:c2 + 1]

        # pozostałe pozycje w kolejności z p2
        pos = (np.arange(n) + c2 + 1) % n
        p2_list = list(p2)
        idx_p2 = 0
        for p in pos:
            if child[p] != -1:
                continue
            # szukamy pierwszego genu z p2, który nie jest jeszcze w dziecku
            while p2_list[idx_p2] in child:
                idx_p2 += 1
            child[p] = p2_list[idx_p2]
            idx_p2 += 1
            if idx_p2 >= n:
                break
        return child

    def swap_mutation(ind: np.ndarray) -> None:
        i, j = np.random.choice(len(ind), size=2, replace=False)
        ind[i], ind[j] = ind[j], ind[i]

    # start GA

    pop = init_population(pop_size, n)
    fits = np.empty(pop_size)
    extra = [None] * pop_size  # (distance, overload, lateness)

    # ocena początkowej populacji
    for i in range(pop_size):
        routes, _, _ = split_routes(pop[i], df, D, Q, alpha=alpha, beta=beta,gamma=gamma)
        f, d, q, t = fitness_penalty_from_routes(
            routes,
            df,
            D,
            Q,
            alpha=alpha,
            beta=beta,
            max_vehicles=max_vehicles,
            gamma=gamma
        )
        fits[i] = f
        extra[i] = (d, q, t)

    best_idx = int(np.argmin(fits))
    best = pop[best_idx].copy()
    best_stats = extra[best_idx]
    best_fit = float(fits[best_idx])
    history = [best_fit]

    # główna pętla GA
    start_time = time.time()

    for _ in range(gens):
        # limit czasu - przerywamy jeśli przekroczony
        if time_limit_sec is not None and (time.time() - start_time) >= time_limit_sec:
            break
    
        new_pop = [best.copy()]  # elityzm

        while len(new_pop) < pop_size:
            p1 = tournament_selection(pop, fits)
            p2 = tournament_selection(pop, fits)

            # krzyżowanie
            if np.random.rand() < pc:
                child = ox_crossover(p1, p2)
            else:
                child = p1.copy()

            # mutacja
            if np.random.rand() < pm:
                swap_mutation(child)

            new_pop.append(child)

        pop = np.array(new_pop, dtype=int)

        # ocena nowej populacji
        for i in range(pop_size):
            routes, _, _ = split_routes(pop[i], df, D, Q, alpha=alpha, beta=beta,gamma=gamma)
            f, d, q, t = fitness_penalty_from_routes(
                routes,
                df,
                D,
                Q,
                alpha=alpha,
                beta=beta,
                max_vehicles=max_vehicles,
                gamma=gamma
            )
            fits[i] = f
            extra[i] = (d, q, t)

        best_idx = int(np.argmin(fits))
        if fits[best_idx] < best_fit:
            best_fit = float(fits[best_idx])
            best = pop[best_idx].copy()
            best_stats = extra[best_idx]

        history.append(best_fit)

    stats = {
        "fitness": best_fit,
        "distance": best_stats[0],
        "overload": best_stats[1],
        "lateness": best_stats[2],
    }
    return best, stats, history
