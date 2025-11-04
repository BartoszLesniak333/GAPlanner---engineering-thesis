import numpy as np
from vrptw.fitness import fitness_penalty_from_routes
from vrptw.split import split_routes

def run_ga(df, D, Q, pop_size, gens, pc, pm, alpha):
    n = len(df) - 1  # pomijamy depot (id=0)

    def init_population(size, n):
        base = np.arange(1, n + 1)
        return np.array([np.random.permutation(base) for _ in range(size)])

    def tournament_selection(pop, fits, k=3):
        idx = np.random.choice(len(pop), size=k, replace=False)
        best = min(idx, key=lambda i: fits[i])
        return pop[best].copy()

    def ox_crossover(p1, p2):
        n = len(p1)
        c = np.full(n, -1)
        a, b = sorted(np.random.choice(n, 2, replace=False))
        c[a:b+1] = p1[a:b+1]
        pos = (b + 1) % n
        for gene in p2:
            if gene not in c:
                c[pos] = gene
                pos = (pos + 1) % n
        return c

    def swap_mutation(child):
        i, j = np.random.choice(len(child), 2, replace=False)
        child[i], child[j] = child[j], child[i]

    # Inicjalizacja
    pop = init_population(pop_size, n)
    fits = np.empty(pop_size)
    extra = [None] * pop_size

    for i in range(pop_size):
        routes, _, _ = split_routes(pop[i], df, D, Q, alpha=alpha, beta=100)
        f, d, q, t = fitness_penalty_from_routes(routes, df, D, Q, alpha, 100)
        fits[i] = f
        extra[i] = (d, q, t)

    best_idx = np.argmin(fits)
    best = pop[best_idx].copy()
    best_fit = fits[best_idx]
    best_extra = extra[best_idx]
    history = [best_fit]

    for _ in range(gens):
        new_pop = [best.copy()]
        while len(new_pop) < pop_size:
            p1 = tournament_selection(pop, fits)
            p2 = tournament_selection(pop, fits)
            if np.random.rand() < pc:
                child = ox_crossover(p1, p2)
            else:
                child = p1.copy()
            if np.random.rand() < pm:
                swap_mutation(child)
            new_pop.append(child)

        pop = np.vstack(new_pop)

        for i in range(pop_size):
            routes, _, _ = split_routes(pop[i], df, D, Q, alpha=alpha, beta=100)
            f, d, q, t = fitness_penalty_from_routes(routes, df, D, Q, alpha, 100)
            fits[i] = f
            extra[i] = (d, q, t)

        gen_best_idx = np.argmin(fits)
        if fits[gen_best_idx] < best_fit:
            best_fit = fits[gen_best_idx]
            best = pop[gen_best_idx].copy()
            best_extra = extra[gen_best_idx]

        history.append(best_fit)

    stats = {
        "fitness": best_fit,
        "distance": best_extra[0],
        "overload": best_extra[1],
        "lateness": best_extra[2],
    }
    return best, stats, history
