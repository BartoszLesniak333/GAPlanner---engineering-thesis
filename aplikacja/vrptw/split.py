import numpy as np

def split_routes(pi, df, D, Q, alpha=1000.0, beta=100.0, gamma=0.0):
 
    n = len(pi)                #liczba klientów w permutacji
    INF = float('inf')

    # tablice do proggramwoania dynamicznego
    dp = [INF] * (n + 1)       # min. koszt pociecia
    prv = [-1] * (n + 1)       # miejsce ciecia
    dp[0] = 0                  # warunek poczatkowy kosztu pociecia


    #funkcja do liczenia kosztu jedenj trasy
    def arc_cost(i, j):         
        load = 0               # aktualny ładunek w aucie
        t = 0                  # czas w punkcie tarsy
        cost = 0               # dotychczasowy koszt
        late = 0               # ilość spóżnień - laczny czas
        viol = 0               # liczba naruszen okien
        last = 0               # numer ostatnio odwiedzonego wezla. start w depot

        for k in range(i + 1, j + 1):
            nid = pi[k - 1]    # id klienta z permutacji

            # pobieranie danych z dataframe
            demand = df.loc[nid, 'demand']
            ready = df.loc[nid, 'ready']
            due = df.loc[nid, 'due']
            service = df.loc[nid, 'service']

            # sprawdzenie czy po dodoaniu ładunku do aktualnego ładunku nie przekroczy ładowności
            if load + demand > Q:
                return INF

            cost += D[last, nid]    # aktualizacja kosztu
            t += D[last, nid]       # aktualizacja czasu o przyjazd

            # przyjazd za wczesnie - nie karze tutaj
            if t < ready:
                t = ready

            # przyajzd za poźno - kara
            if t > due:
                late += (t - due)
                viol += 1

            t += service           # aktualizacja czasu o serwis
            load += demand         # aktualizacja załadowania
            last = nid             # aktualizacja ostatnio owiedzonego klienta

        cost += D[last, 0]  # powrót do depotu
        total = cost + alpha * late + beta * viol + gamma
        return total

    #programowanie dynamiczne
    for j in range(1, n + 1):          # rozwazamy j elementów w permutacji od poczatku do indexu j-1
        for i in range(j):             # kandydat na poczatek ostatniej trasy
            c = arc_cost(i, j)

            # aktualizacja najlepszego kosztu i miejsca 
            if dp[i] + c < dp[j]:
                dp[j] = dp[i] + c
                prv[j] = i

    routes = []                 # pusta linia na gotowe trasy
    j = n                       # zaczynamy od konca permutacji
    while j > 0:                # pętla rozpoczynająca powrót wstecz do depot
        i = prv[j]
        routes.append([0] + list(pi[i:j]) + [0]) # dodajemy trase i depot na poczatku i końcu
        j = i                   # cofamy sie do poprzedniego ciecia 
    routes.reverse()            # odwracamy trase aby była od poczatku do konca

    # zwracamy liste tras, liczbe tras, min łączny koszt wyliczony dla całej permutacji
    return routes, len(routes), dp[n]
