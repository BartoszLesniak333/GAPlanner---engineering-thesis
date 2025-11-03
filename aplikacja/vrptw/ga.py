# ga.py
import numpy as np

# tu zaimportuj swoją ocenę / split.
# zakładam, że masz coś w stylu:
# from vrptw.fitness import fitness_vrptw
# i że ta funkcja zwraca: (fitness, distance, overload, lateness)
from vrptw.fitness import fitness_vrptw  # <-- dostosuj do swojej ścieżki/modułu


def _get_n_clients(inst):
    """
    W naszej implementacji Instance (model.py) mamy:
    - inst.data: DataFrame z wierszem depotu id=0
    - klienci: id 1..N
    więc liczba klientów = liczba wierszy - 1.
    """
    if hasattr(inst, "data"):
        # zakładamy że depot jest zawsze 0
        return len(inst.data) - 1

    # fallbacki gdybyś kiedyś użył innej struktury
    if hasattr(inst, "n_clients"):
        return int(inst.n_clients)
    if hasattr(inst, "n_customers"):
        return int(inst.n_customers)
    if hasattr(inst, "customers"):
        return len(inst.customers)

    raise ValueError("Nie mogę zgadnąć liczby klientów z obiektu inst.")

def _init_population(pop_size, n):
    """pop_size losowych permutacji 1..n jako ndarray (pop_size, n)."""
    base = np.arange(1, n + 1, dtype=np.int64)
    pop = np.empty((pop_size, n), dtype=np.int64)
    for i in range(pop_size):
        pop[i] = np.random.permutation(base)
    return pop


def _tournament_selection(pop, fits, k=3):
    """Zwraca *kopię* wybranego osobnika."""
    idx = np.random.randint(0, pop.shape[0], size=k)
    best = idx[0]
    for j in idx[1:]:
        if fits[j] < fits[best]:  # minimalizujemy
            best = j
    return pop[best].copy()


def _ox_crossover(p1, p2):
    """
    OX (Order Crossover) dla permutacji.
    p1, p2: ndarray 1D z tym samym n.
    """
    n = p1.shape[0]
    c = np.full(n, -1, dtype=np.int64)

    a, b = np.sort(np.random.choice(n, size=2, replace=False))
    # skopiuj środek z p1
    c[a:b+1] = p1[a:b+1]

    # uzupełniaj z p2 w kolejności
    pos = (b + 1) % n
    for gene in p2:
        if gene not in c:  # można przyspieszyć, ale tak jest czytelnie
            c[pos] = gene
            pos = (pos + 1) % n

    return c


def _swap_mutation(child):
    """Zamiana dwóch losowych pozycji w permutacji."""
    n = child.shape[0]
    i, j = np.random.randint(0, n, size=2)
    child[i], child[j] = child[j], child[i]


def run_ga(inst, pop_size, gens, pc, pm, alpha):
    """
    GA (wersja bazowa) dla VRPTW w reprezentacji giant-tour.
    :param inst: twoja instancja VRPTW
    :param pop_size: liczba osobników w populacji
    :param gens: liczba generacji (warunek stopu)
    :param pc: prawdopodobieństwo krzyżowania
    :param pm: prawdopodobieństwo mutacji (SWAP)
    :param alpha: współczynnik kary (przekazujemy do fitnessu)
    :return: (best_perm, best_stats, history)
    """
    n = _get_n_clients(inst)

    # 1) inicjalizacja
    pop = _init_population(pop_size, n)

    # 2) ocena startowa
    fits = np.empty(pop_size, dtype=float)
    extra_stats = [None] * pop_size  # żeby mieć distance, overload, lateness
    for i in range(pop_size):
        f, dist, overload, lateness = fitness_vrptw(inst, pop[i], alpha)
        fits[i] = f
        extra_stats[i] = (dist, overload, lateness)

    # najlepszy na starcie
    best_idx = int(np.argmin(fits))
    best_perm = pop[best_idx].copy()
    best_fit = float(fits[best_idx])
    best_dist, best_over, best_late = extra_stats[best_idx]

    history = [best_fit]

    # 3) główna pętla GA
    for _ in range(gens):
        new_pop = []
        # elityzm: wrzuć najlepszego na początek
        new_pop.append(best_perm.copy())

        # generuj resztę populacji
        while len(new_pop) < pop_size:
            # selekcja
            p1 = _tournament_selection(pop, fits, k=3)
            p2 = _tournament_selection(pop, fits, k=3)

            # krzyżowanie
            if np.random.rand() < pc:
                child = _ox_crossover(p1, p2)
            else:
                child = p1.copy()

            # mutacja
            if np.random.rand() < pm:
                _swap_mutation(child)

            new_pop.append(child)

        pop = np.vstack(new_pop)

        # ocena nowej populacji
        for i in range(pop_size):
            f, dist, overload, lateness = fitness_vrptw(inst, pop[i], alpha)
            fits[i] = f
            extra_stats[i] = (dist, overload, lateness)

        # aktualizacja najlepszego
        gen_best_idx = int(np.argmin(fits))
        gen_best_fit = float(fits[gen_best_idx])
        if gen_best_fit < best_fit:
            best_fit = gen_best_fit
            best_perm = pop[gen_best_idx].copy()
            best_dist, best_over, best_late = extra_stats[gen_best_idx]

        history.append(best_fit)

    # pakujemy best_stats tak, żebyś miał wszystko pod ręką
    best_stats = {
        "fitness": best_fit,
        "distance": best_dist,
        "overload": best_over,
        "lateness": best_late,
    }

    return best_perm, best_stats, history
