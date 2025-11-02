from vrptw.data import load_instance
from vrptw.split import split_routes
from vrptw.fitness import fitness_penalty_from_routes
import numpy as np


#import danych z csv
inst = load_instance("data/data2.csv")  

# Przykładowa instancja, którą sobie wybrałem

# data1
#pi = np.array([3,5,1,4,2])  

#data2
#pi = np.array([14, 3, 7, 11, 18, 1, 9, 5, 16, 13, 19, 8, 10, 2, 15, 6, 12, 4, 17])

# data3
pi = np.array([14, 3, 7, 11, 18, 1, 9, 5, 16, 13, 19, 8, 10, 2, 15, 6, 12, 4, 17, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29])

#data4
#pi = np.array([14, 3, 7, 11, 18, 1, 9, 5, 16, 13, 19, 8, 10, 2, 15, 6, 12, 4, 17, 25, 22, 27, 20, 33, 28, 31, 24, 39, 35, 23, 36, 21, 37, 30, 26, 38, 34, 29, 32])

# SPLIT
routes, NV, total_cost = split_routes(pi, inst, alpha=0, beta=0)

#Wynik
print("Liczba tras:", NV)
for i, r in enumerate(routes):
    print(f"Trasa {i+1}: {r}")
print("Całkowity koszt:", total_cost)


# --- 4) Policz funkcję z karami: c~(s) = c + α q + β ω ---
alpha = 1000.0   # kara za overload (q)
beta  = 100.0    # kara za lateness (ω)
res = fitness_penalty_from_routes(routes, inst, alpha=1000.0, beta=100.0)

print("\n=== Wyniki funkcji z karami ===")
print("fitness  c~(s):", round(res["fitness"],2))
print("distance   c(s):", round(res["distance"],2))
print("overload   q(s):", res["q_overload"])
print("lateness ω(s):", round(res["omega_lateness"],2))

# --- 5) Sprawdzenie spójności: fitness ≈ c + αq + βω ---
lhs = res["fitness"]
rhs = res["distance"] + alpha*res["q_overload"] + beta*res["omega_lateness"]
assert abs(lhs - rhs) < 1e-6, "Rozbieżność w obliczeniach fitness!"
print("\n[OK] Spójność: c~(s) = c + αq + βω")