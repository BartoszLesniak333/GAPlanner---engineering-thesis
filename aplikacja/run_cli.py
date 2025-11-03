#!/usr/bin/env python3
import argparse
import os
import csv

import matplotlib.pyplot as plt  # <-- nowość

from vrptw.data import load_instance
from vrptw.split import split_routes
import vrptw.fitness as fitness_mod


def make_fitness_vrptw(beta: float = 100.0):
    def fitness_vrptw(inst, perm, alpha: float):
        # używamy tych samych kar co GA
        routes, NV, _ = split_routes(perm, inst, alpha=alpha, beta=beta)
        fit, dist, cap_v, time_v = fitness_mod.fitness_penalty_from_routes(
            routes, inst, alpha=alpha, beta=beta
        )
        return fit, dist, cap_v, time_v
    return fitness_vrptw


def parse_args():
    p = argparse.ArgumentParser(description="GA dla VRPTW (route-first, cluster-second)")
    p.add_argument("--instance", default="data/data5.csv")
    p.add_argument("--pop", type=int, default=50)
    p.add_argument("--gens", type=int, default=200)
    p.add_argument("--pc", type=float, default=0.9)
    p.add_argument("--pm", type=float, default=0.2)
    p.add_argument("--alpha", type=float, default=1000.0)
    p.add_argument("--beta", type=float, default=100.0)
    p.add_argument("--outdir", default="out", help="katalog wyjściowy na csv i png")
    return p.parse_args()


def save_history_plot(history, out_path: str):
    plt.figure()
    plt.plot(range(1, len(history) + 1), history)
    plt.xlabel("Generacja")
    plt.ylabel("Najlepszy fitness")
    plt.title("Historia GA")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_routes_plot(routes, inst, out_path: str):
    """
    Rysuje mapę: punkt 0 jako depot, każda trasa w osobnym kolorze.
    trasa: depot -> klienci -> depot
    """
    df = inst.data
    xs = df["x"]
    ys = df["y"]

    plt.figure()
    # narysuj wszystkich klientów jako szare punkty
    plt.scatter(xs, ys, s=20, c="lightgray", zorder=1)

    # depot (zakładamy id=0)
    depot_x = xs.loc[0]
    depot_y = ys.loc[0]
    plt.scatter([depot_x], [depot_y], s=80, c="black", marker="s", label="Depot", zorder=3)

    # każda trasa osobno
    for idx, r in enumerate(routes, start=1):
        # pełna ścieżka z powrotem do depotu
        full = [0] + r + [0]
        route_x = [xs.loc[i] for i in full]
        route_y = [ys.loc[i] for i in full]
        # matplotlib sam pokoloruje kolejne linie
        plt.plot(route_x, route_y, marker="o", label=f"Trasa {idx}", zorder=2)

    plt.title("Trasy VRPTW")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc="best")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    args = parse_args()

    # 1) instancja
    inst = load_instance(args.instance)

    # 2) wstrzyknięcie fitnessu, którego oczekuje ga.py
    fitness_mod.fitness_vrptw = make_fitness_vrptw(beta=args.beta)

    # 3) dopiero teraz importujemy GA
    from vrptw.ga import run_ga

    # 4) GA
    best_perm, best_stats, history = run_ga(
        inst=inst,
        pop_size=20,
        gens=30,
        pc=args.pc,
        pm=args.pm,
        alpha=args.alpha,
    )

    # 5) dekodowanie NA TYCH SAMYCH KARACH co GA
    routes, NV, _ = split_routes(
        best_perm,
        inst,
        alpha=args.alpha,
        beta=args.beta,
    )

    fitness_val, dist, cap_v, time_v = fitness_mod.fitness_penalty_from_routes(
        routes,
        inst,
        alpha=args.alpha,
        beta=args.beta,
    )

    # 6) druk
    print("=== NAJLEPSZY OSOBNIK Z GA ===")
    print("Permutacja (giant tour):", best_perm.tolist())
    print(f"Liczba tras (NV): {NV}")
    for i, r in enumerate(routes, start=1):
        print(f"Trasa {i}: {r}")

    print("\n=== Metryki ===")
    print(f"dist (c(s))          : {dist:.4f}")
    print(f"viol_capacity q(s)   : {cap_v:.4f}")
    print(f"viol_time ω(s)       : {time_v:.4f}")
    print(f"fitness c̃(s)        : {fitness_val:.4f}")

    lhs = fitness_val
    rhs = dist + args.alpha * cap_v + args.beta * time_v
    if abs(lhs - rhs) < 1e-6:
        print("\n[OK] Spójność: c̃(s) = c + α q + β ω")
    else:
        print("\n[WARN] NIESPÓJNOŚĆ przy liczeniu funkcji celu!")

    # 7) zapisy
    os.makedirs(args.outdir, exist_ok=True)

    # 7a) history.csv
    hist_csv = os.path.join(args.outdir, "history.csv")
    with open(hist_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "best_fitness"])
        for gen, val in enumerate(history, start=1):
            w.writerow([gen, val])
    print(f"[OK] Zapisano historię do: {hist_csv}")

    # 7b) history.png
    hist_png = os.path.join(args.outdir, "history.png")
    save_history_plot(history, hist_png)
    print(f"[OK] Zapisano wykres historii do: {hist_png}")

    # 7c) routes.png
    routes_png = os.path.join(args.outdir, "routes.png")
    save_routes_plot(routes, inst, routes_png)
    print(f"[OK] Zapisano mapę tras do: {routes_png}")


if __name__ == "__main__":
    main()

