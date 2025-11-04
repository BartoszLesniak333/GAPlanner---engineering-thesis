import argparse
import os
import csv
import matplotlib.pyplot as plt

from vrptw.data import load_instance
from vrptw.split import split_routes
from vrptw.fitness import fitness_penalty_from_routes
from vrptw.ga import run_ga

def parse_args():
    p = argparse.ArgumentParser(description="GA dla VRPTW (prosta wersja)")
    p.add_argument("--instance", default="data/data2.csv")
    p.add_argument("--pop", type=int, default=50)
    p.add_argument("--gens", type=int, default=200)
    p.add_argument("--pc", type=float, default=0.9)
    p.add_argument("--pm", type=float, default=0.2)
    p.add_argument("--alpha", type=float, default=1000.0)
    p.add_argument("--beta", type=float, default=100.0)
    p.add_argument("--outdir", default="out")
    return p.parse_args()

def save_history_plot(history, out_path):
    plt.plot(range(1, len(history)+1), history)
    plt.xlabel("Generacja")
    plt.ylabel("Najlepszy fitness")
    plt.title("Historia GA")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_routes_plot(routes, df, out_path):
    xs = df["x"]
    ys = df["y"]

    plt.figure()
    plt.scatter(xs, ys, s=20, c="lightgray")
    plt.scatter([xs.loc[0]], [ys.loc[0]], s=80, c="black", marker="s", label="Depot")

    for idx, r in enumerate(routes, start=1):
        full = [0] + r + [0]
        route_x = [xs.loc[i] for i in full]
        route_y = [ys.loc[i] for i in full]
        plt.plot(route_x, route_y, marker="o", label=f"Trasa {idx}")

    plt.title("Trasy VRPTW")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    args = parse_args()
    df, D, Q = load_instance(args.instance)

    best_perm, stats, history = run_ga(
        df, D, Q,
        pop_size=20,
        gens=30,
        pc=args.pc,
        pm=args.pm,
        alpha=args.alpha
    )

    routes, NV, _ = split_routes(best_perm, df, D, Q, args.alpha, args.beta)
    fitness_val, dist, cap_v, time_v = fitness_penalty_from_routes(routes, df, D, Q, args.alpha, args.beta)

    print("=== NAJLEPSZY OSOBNIK ===")
    print("Permutacja:", best_perm.tolist())
    for i, r in enumerate(routes, start=1):
        print(f"Trasa {i}: {r}")

    print("\n=== METRYKI ===")
    print(f"Liczba tras       : {NV}")
    print(f"Dystans           : {dist:.2f}")
    print(f"Przeładowania     : {cap_v:.2f}")
    print(f"Spóźnienia        : {time_v:.2f}")
    print(f"Fitness           : {fitness_val:.2f}")

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
