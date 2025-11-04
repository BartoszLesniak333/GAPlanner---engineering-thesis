import numpy as np

def split_routes(pi, df, D, Q, alpha=1000.0, beta=100.0):
    """
    Prosty dekoder split: dzieli permutację klientów na trasy zgodne z VRPTW.
    Uwzględnia ograniczenie pojemności i karę za spóźnienia.
    """
    n = len(pi)
    INF = 1e18
    dp = [INF] * (n + 1)
    prv = [-1] * (n + 1)
    dp[0] = 0

    def arc_cost(i, j):
        load = 0
        t = 0
        cost = 0
        late = 0
        viol = 0
        last = 0  # depot

        for k in range(i + 1, j + 1):
            nid = pi[k - 1]
            demand = df.loc[nid, 'demand']
            ready = df.loc[nid, 'ready']
            due = df.loc[nid, 'due']
            service = df.loc[nid, 'service']

            if load + demand > Q:
                return INF

            cost += D[last, nid]
            t += D[last, nid]

            if t < ready:
                t = ready
            if t > due:
                late += (t - due)
                viol += 1
            t += service
            load += demand
            last = nid

        cost += D[last, 0]  # powrót do depotu
        total = cost + alpha * late + beta * viol
        return total

    for j in range(1, n + 1):
        for i in range(j):
            c = arc_cost(i, j)
            if dp[i] + c < dp[j]:
                dp[j] = dp[i] + c
                prv[j] = i

    routes = []
    j = n
    while j > 0:
        i = prv[j]
        routes.append([0] + list(pi[i:j]) + [0])
        j = i
    routes.reverse()

    return routes, len(routes), dp[n]
