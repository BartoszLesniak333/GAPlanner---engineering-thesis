from vrptw.data import load_instance
from vrptw.split import split_routes
from vrptw.fitness import fitness_penalty_from_routes
import numpy as np


#import danych z csv
inst = load_instance("data/data1.csv")  

# Przykładowa instancja, którą sobie wybrałem

# data 1
pi = np.array([3,5,1,4,2])  

#data 2
#pi = np.array([14, 3, 7, 11, 18, 1, 9, 5, 16, 13, 19, 8, 10, 2, 15, 6, 12, 4, 17])

# data 3
#pi = np.array([14, 3, 7, 11, 18, 1, 9, 5, 16, 13, 19, 8, 10, 2, 15, 6, 12, 4, 17, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29])

#data 4
#pi = np.array([14, 3, 7, 11, 18, 1, 9, 5, 16, 13, 19, 8, 10, 2, 15, 6, 12, 4, 17, 25, 22, 27, 20, 33, 28, 31, 24, 39, 35, 23, 36, 21, 37, 30, 26, 38, 34, 29, 32])

#data 5
#pi = np.array([14, 3, 7, 11, 18, 1, 9, 5, 16, 13, 19, 8, 10, 2, 15, 6, 12, 4, 17, 25, 22, 27, 20, 33, 28, 31, 24, 39, 35, 23, 36, 21, 37, 30, 26, 38, 34, 29, 32, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49])

# SPLIT
routes, NV, total_cost = split_routes(pi, inst, alpha=0, beta=0)

#Wynik
print("Liczba tras:", NV)
for i, r in enumerate(routes):
    print(f"Trasa {i+1}: {r}")
print("Całkowity koszt:", total_cost)

