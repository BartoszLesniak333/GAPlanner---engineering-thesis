import argparse    #biblioteka do czytania parametrów z lini komend
import os          # biblitorka do zapisu plików 
import csv
import matplotlib.pyplot as plt

from vrptw.data import load_instance
from vrptw.split import split_routes
from vrptw.ga import run_ga


def parse_args():
    parser = argparse.ArgumentParser()
    
    # wczytanie danych
    parser.add_argument("--instance", type=str, default="data/data5.csv", help="Ścieżka do pliku CSV z instancją VRPTW")

    # parametry algorytmu genetycznego
    parser.add_argument("--pop", type=int, default=20, help="Liczba osobników w populacji")
    parser.add_argument("--gens", type=int, default=40, help="Liczba pokoleń")
    parser.add_argument("--pc", type=float, default=0.9, help="Prawdopodobieństwo krzyżowania")
    parser.add_argument("--pm", type=float, default=0.2, help="Prawdopodobieństwo mutacji")
    parser.add_argument("--alpha", type=float, default=1000.0, help="Waga kary za spóźnienie")
    parser.add_argument("--beta", type=float, default=100.0, help="Waga za każde naruszenie ograniczenia")
    
    # katalog wyjściowy
    parser.add_argument("--outdir", type=str, default="out", help="Folder do zapisu wyników")
    
    return parser.parse_args()

# funkcja do rysowania historii fitnessu 
def save_history_plot(history, out_path):
    plt.plot(range(1, len(history)+1), history)
    plt.xlabel("Generacja")
    plt.ylabel("Najlepszy fitness")
    plt.title("Historia GA")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

#funkcja do rysowania mapy tras
def save_routes_plot(routes, df, out_path):
    xs = df["x"]
    ys = df["y"]

    plt.figure()
    plt.scatter(xs, ys, s=20, c="lightgray")
    plt.scatter([xs.loc[0]], [ys.loc[0]], s=80, c="black", marker="s", label="Depot")

    for idx, r in enumerate(routes, start=1):
        route_x = [xs.loc[i] for i in r]
        route_y = [ys.loc[i] for i in r]
        plt.plot(route_x, route_y, marker="o", label=f"Trasa {idx}")

    plt.title("Trasy VRPTW")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

#funkcja main
def main():
    args = parse_args()
    df, D, Q = load_instance(args.instance)

    best_perm, stats, history = run_ga(
        df, D, Q,
        pop_size=args.pop,
        gens=args.gens,
        pc=args.pc,
        pm=args.pm,
        alpha=args.alpha,
        beta=args.beta
    )

    routes, NV, _ = split_routes(best_perm, df, D, Q, args.alpha, args.beta)

    print("=== NAJLEPSZY OSOBNIK ===")
    print("Permutacja:", best_perm.tolist())
    for i, r in enumerate(routes, start=1):
        print(f"Trasa {i}: {r}")

    print("\n=== METRYKI ===")
    print(f"Liczba tras       : {NV}")
    print(f"Dystans           : {stats['distance']:.2f}")
    print(f"Przeładowania     : {stats['overload']:.2f}")
    print(f"Spóźnienia        : {stats['lateness']:.2f}")
    print(f"Fitness           : {stats['fitness']:.2f}")
   

    os.makedirs(args.outdir, exist_ok=True)

    with open(os.path.join(args.outdir, "history.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "best_fitness"])
        for i, val in enumerate(history, 1):
            writer.writerow([i, val])

    save_history_plot(history, os.path.join(args.outdir, "history.png"))
    save_routes_plot(routes, df, os.path.join(args.outdir, "routes.png"))

if __name__ == "__main__":
    main()
