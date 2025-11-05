import numpy as np
from vrptw.fitness import fitness_penalty_from_routes
from vrptw.split import split_routes

def run_ga(df, D, Q, pop_size, gens, pc, pm, alpha):  
    
    # 'pop size' to liczba osobników tak dla przypomnienia
    # 'pc' to prawd. krzyżowania a 'pm' to prawd. mutacji
    
    n = len(df) - 1  # pomijamy depot (id=0) poniewaz permutacje od 1 do n 

    # inicjalizacja populacji -> losuje k indeksów, wybieram najlepszy (najm. fitness) i zwracam kopie
    def init_population(size, n):
        base = np.arange(1, n + 1)
        return np.array([np.random.permutation(base) for _ in range(size)])

    #selekcja turniejowa
    def tournament_selection(pop, fits, k=3):
        idx = np.random.choice(len(pop), size=k, replace=False)
        best = min(idx, key=lambda i: fits[i])
        return pop[best].copy()

    # krzyżowanie OX

    """
    Tutaj sobie opisze bo to ważne. 
    1, wytnij losowy fragmment z p1 i wstaw di dziecka
    2. reszte pozycji uzupełnij z p2 pomijajc powtórki
    Efektem jest dziecko zachowujace fragmenty obyu rodziców i jest poprawna permutacja
    """

    def ox_crossover(p1, p2):
        n = len(p1)
        c = np.full(n, -1)         # dziecko
        a, b = sorted(np.random.choice(n, 2, replace=False))
        c[a:b+1] = p1[a:b+1]      #wstawienie do dziecka
        pos = (b + 1) % n
        for gene in p2:           # uzupełnienie reszty genami
            if gene not in c:
                c[pos] = gene
                pos = (pos + 1) % n
        return c

    # mutacja -> zamien 2 geny ze sobą
    def swap_mutation(child):
        i, j = np.random.choice(len(child), 2, replace=False)
        child[i], child[j] = child[j], child[i]

    # Inicjalizacja
    pop = init_population(pop_size, n)     # inicjalizacja populacji
    fits = np.empty(pop_size)              # Wartosc startowa
    extra = [None] * pop_size              # Wartosc startowa

    for i in range(pop_size):
        routes, _, _ = split_routes(pop[i], df, D, Q, alpha=alpha, beta=100)     # wywołanie splitu
        f, d, q, t = fitness_penalty_from_routes(routes, df, D, Q, alpha, 100)   # wywyołanie fitnessu
        fits[i] = f    # przypisanie fitnesu
        extra[i] = (d, q, t)   # kontener metryk pomocnicztch, czyli dystans, przeciazenie, spoznienie

    best_idx = np.argmin(fits)
    best = pop[best_idx].copy()
    best_fit = fits[best_idx]
    best_extra = extra[best_idx]
    history = [best_fit]

    # Ewolucja
    for _ in range(gens):
        new_pop = [best.copy()]         # tu zachodzi Elityzm czyli przechodzi bez zmian
        while len(new_pop) < pop_size:
            p1 = tournament_selection(pop, fits) # selekcja
            p2 = tournament_selection(pop, fits)
            if np.random.rand() < pc:   # prawd. krzyzowania 
                child = ox_crossover(p1, p2)
            else:
                child = p1.copy()
            if np.random.rand() < pm:   # prawd. mutacji 
                swap_mutation(child)
            new_pop.append(child)

        pop = np.vstack(new_pop)      # przypisanie nowej populacji

        for i in range(pop_size):
            routes, _, _ = split_routes(pop[i], df, D, Q, alpha=alpha, beta=100)   # ocena split
            f, d, q, t = fitness_penalty_from_routes(routes, df, D, Q, alpha, 100)  # ocena fitness
            fits[i] = f
            extra[i] = (d, q, t)

        # porównanie najleszego w tej generacji i aktualizacja globalnie najlepszego oraz zapis wynik do history
        gen_best_idx = np.argmin(fits)
        if fits[gen_best_idx] < best_fit:
            best_fit = fits[gen_best_idx]
            best = pop[gen_best_idx].copy()
            best_extra = extra[gen_best_idx]

        history.append(best_fit)
    # metryki najnlepszego rozwiazania
    stats = {
        "fitness": best_fit,
        "distance": best_extra[0],
        "overload": best_extra[1],
        "lateness": best_extra[2],
    }
    return best, stats, history

#  NOTATKI
# czy musze wprawdzic zmienna oznaczjaca liczbe pojzdów do split? wtedy algorytm bedzie preferował mnijesza liczbe tras
# 
# 
# 
# 
# 
# 
# 
# 
# 
