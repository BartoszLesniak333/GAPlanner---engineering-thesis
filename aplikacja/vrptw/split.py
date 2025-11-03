import numpy as np

def split_routes(pi, inst, alpha=1000.0, beta=100.0):
    """
    Dekoder SPLIT – przekształca giant tour (permutację klientów) na zestaw tras VRPTW.
    Bazuje na klasycznym podejściu 'route-first, cluster-second' (Beasley 1983, Toth & Vigo 2014),
    powszechnie stosowanym w algorytmach genetycznych do VRPTW (Potvin 1996, Berger et al. 2003).
    """

    n = len(pi)
    INF = 10**18

    # dp[j] = minimalny koszt obsłużenia pierwszych j klientów
    # prv[j] = indeks i, po którym rozpoczyna się ostatnia trasa w najlepszym rozwiązaniu prefiksu 1..j
    # (Toth & Vigo 2014 – rozdz. 4.3.1 Route-first Cluster-second)
    dp = [INF]*(n+1)
    prv = [-1]*(n+1)
    dp[0] = 0

    # Funkcja kosztu łuku (i,j): obsługa segmentu klientów pi[i+1..j]
    # – inspirowane Potvin & Bengio (1996), sekcja 2.1
    # – Berger, Barkaoui & Bräysy (2003), sekcja 3.1
    def arc_cost(i, j):
        Q = inst.Q
        df = inst.data
        dist = inst.distance
        load = 0
        t = 0.0
        cost = 0.0
        late = 0.0
        viol = 0
        last = 0  # depot

        # iteracja po klientach z zakresu i+1 .. j (czyli pi[i+1..j])
        for k in range(i+1, j+1):
            nid = int(pi[k-1])  # id klienta (zgodne z indeksem w df)
            demand = df.demand.iloc[nid]
            ready = df.ready.iloc[nid]
            due = df.due.iloc[nid]
            service = df.service.iloc[nid]

            # --- Ograniczenie pojemności (Beasley 1983, Potvin 1996) ---
            if load + demand > Q:
                return INF, 0, 0  # segment niemożliwy

            # --- Czas przejazdu i serwis (Potvin 1996 – VRPTW decoding) ---
            cost += dist[last, nid]
            t += dist[last, nid]

            # --- Obsługa okien czasowych (waiting + karanie spóźnień) ---
            # Zgodnie z Bräysy & Gendreau (2005), czekanie nie jest karane.
            if t < ready:
                t = ready  # czekamy do otwarcia okna
            if t > due:
                late += (t - due)  # kara za spóźnienie
                viol += 1  # zliczanie naruszeń okien
            t += service
            load += demand
            last = nid

        # powrót do depotu (Toth & Vigo 2014)
        cost += dist[last, 0]

        # Funkcja kosztu łuku (hierarchiczna + kary)
        # Berger et al. (2003): dystans + α * opóźnienie + β * liczba naruszeń
        c = cost + alpha * late + beta * viol
        return c, late, viol

    # --- Dynamic Programming (Beasley 1983; Toth & Vigo 2014) ---
    for j in range(1, n+1):
        best = INF
        best_i = -1
        for i in range(0, j):
            c, _, _ = arc_cost(i, j)
            val = dp[i] + c
            if val < best:
                best = val
                best_i = i
        dp[j] = best
        prv[j] = best_i

    # --- Rekonstrukcja tras (standardowy krok DP) ---
    routes = []
    j = n
    while j > 0:
        i = prv[j]
        routes.append(list(pi[i:j]))
        j = i
    routes.reverse()

    # Wynik: lista tras + liczba tras (NV)
    NV = len(routes)
    return routes, NV, dp[n]
