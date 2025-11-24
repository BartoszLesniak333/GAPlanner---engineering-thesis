from __future__ import annotations
import os
import sys
import csv
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# import modułów
from vrptw.data import load_instance
from vrptw.ga import run_ga
from vrptw.split import split_routes

# PySide6 - Interfejs
from PySide6.QtCore import QThread, Signal, Slot
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QLineEdit, QSpinBox, QDoubleSpinBox, QPushButton,
    QFileDialog, QLabel, QProgressBar, QTextEdit, QMessageBox
)

# widżet matplotlib w QT
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# kontener na przechowywanie argumentów z gui
@dataclass
class GAParams:
    instance_path: str
    outdir: str
    pop: int
    gens: int
    pc: float
    pm: float
    alpha: float
    beta: float
    max_vehicles: int


# klasa do tworzenia wykresu z matplotlib
class MplCanvas(FigureCanvas):
    def __init__(self, width: float = 5.0, height: float = 3.0, dpi: int = 100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

    def clear(self):
        self.ax.clear()
        self.draw_idle()


# główna funkcja obliczeń, dzięki dziedziczeniu po QThread obliczenia działają w tle
class GAWorker(QThread):
    finished = Signal(dict)  # wysyłanie obliczonego wyniku lub błędu

    def __init__(self, params: GAParams):
        super().__init__()
        self.params = params

    def run(self):
        try:
            # wczytanie instancji
            df, D, Q = load_instance(self.params.instance_path)

            # uruchomienie GA
            best_perm, stats, history = run_ga(
                df, D, Q,
                pop_size=self.params.pop,
                gens=self.params.gens,
                pc=self.params.pc,
                pm=self.params.pm,
                alpha=self.params.alpha,
                beta=self.params.beta,
                max_vehicles=self.params.max_vehicles,
            )

            # Uruchomienie splitu dla najlepszego osobnika
            routes, NV, _ = split_routes(
                best_perm, df, D, Q,
                self.params.alpha,
                self.params.beta,
            )

            # twarde sprawdzenie limitu liczby pojazdów
            if self.params.max_vehicles is not None and NV > self.params.max_vehicles:
                self.finished.emit({
                    "ok": False,
                    "error": (
                        f"Znalezione rozwiązanie używa {NV} pojazdów, "
                        f"a dostępne jest tylko {self.params.max_vehicles}."
                    ),
                })
                return

            # zapis wyników
            os.makedirs(self.params.outdir, exist_ok=True)
            hist_csv = os.path.join(self.params.outdir, "history.csv")
            with open(hist_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["generation", "best_fitness"])
                for i, val in enumerate(history, 1):
                    w.writerow([i, val])

            # rysowanie wyników w gui
            self.finished.emit({
                "ok": True,
                "df": df,
                "routes": routes,
                "NV": NV,
                "stats": stats,
                "history": history,
            })

        except Exception:
            # inne błędy (np. bug w kodzie) – pełny traceback do logu
            self.finished.emit({"ok": False, "error": traceback.format_exc()})


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GAPlanner - VRPTW with GA")
        self.resize(1100, 760)

        self.worker: Optional[GAWorker] = None
        self.df = None
        self.routes: Optional[List[List[int]]] = None
        self.stats: Dict[str, Any] = {}
        self.history: List[float] = []

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # lewy panel: parametry + log
        left = QVBoxLayout()
        form = QFormLayout()

        # instancja i wybór pliku
        self.le_instance = QLineEdit()
        self.le_instance.setPlaceholderText("")
        btn_browse = QPushButton("Wybierz plik…")
        btn_browse.clicked.connect(self.on_browse)
        row = QHBoxLayout()
        row.addWidget(self.le_instance)
        row.addWidget(btn_browse)
        w_row = QWidget()
        w_row.setLayout(row)
        form.addRow("Instancja:", w_row)

        # folder wyjściowy
        self.le_outdir = QLineEdit()
        self.le_outdir.setText("out")
        btn_out = QPushButton("Wybierz folder…")
        btn_out.clicked.connect(self.on_browse_outdir)
        row2 = QHBoxLayout()
        row2.addWidget(self.le_outdir)
        row2.addWidget(btn_out)
        w_row2 = QWidget()
        w_row2.setLayout(row2)
        form.addRow("Folder wyjściowy:", w_row2)

        # parametry GA
        self.sb_pop = QSpinBox()
        self.sb_pop.setRange(2, 100000)
        self.sb_pop.setValue(20)

        self.sb_gens = QSpinBox()
        self.sb_gens.setRange(1, 100000)
        self.sb_gens.setValue(40)

        self.dsb_pc = QDoubleSpinBox()
        self.dsb_pc.setRange(0.0, 1.0)
        self.dsb_pc.setSingleStep(0.01)
        self.dsb_pc.setValue(0.9)

        self.dsb_pm = QDoubleSpinBox()
        self.dsb_pm.setRange(0.0, 1.0)
        self.dsb_pm.setSingleStep(0.01)
        self.dsb_pm.setValue(0.2)

        self.dsb_alpha = QDoubleSpinBox()
        self.dsb_alpha.setRange(0.0, 1e9)
        self.dsb_alpha.setDecimals(3)
        self.dsb_alpha.setValue(1000.0)

        self.dsb_beta = QDoubleSpinBox()
        self.dsb_beta.setRange(0.0, 1e9)
        self.dsb_beta.setDecimals(3)
        self.dsb_beta.setValue(100.0)

        self.sb_vehicles = QSpinBox()
        self.sb_vehicles.setRange(1, 1000)
        self.sb_vehicles.setValue(10)

        form.addRow("Populacja:", self.sb_pop)
        form.addRow("Pokolenia:", self.sb_gens)
        form.addRow("Pc:", self.dsb_pc)
        form.addRow("Pm:", self.dsb_pm)
        form.addRow("Alpha:", self.dsb_alpha)
        form.addRow("Beta:", self.dsb_beta)
        form.addRow("Liczba pojazdów:", self.sb_vehicles)

        left.addLayout(form)

        # przyciski
        btns = QHBoxLayout()
        self.btn_run = QPushButton("Start")
        self.btn_run.clicked.connect(self.on_run)
        self.btn_open_out = QPushButton("Otwórz folder wyników")
        self.btn_open_out.clicked.connect(self.on_open_outdir)
        btns.addWidget(self.btn_run)
        btns.addWidget(self.btn_open_out)
        left.addLayout(btns)

        # status, progress, log
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.lbl_status = QLabel("Gotowy.")
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        left.addWidget(self.progress)
        left.addWidget(self.lbl_status)
        left.addWidget(QLabel("Log:"))
        left.addWidget(self.log, 1)

        # prawy panel: wykresy
        right = QVBoxLayout()
        self.canvas_conv = MplCanvas(width=6.5, height=3.8)
        self.canvas_routes = MplCanvas(width=6.5, height=3.8)
        right.addWidget(QLabel("Historia fitness"))
        right.addWidget(self.canvas_conv)
        right.addWidget(QLabel("Trasy"))
        right.addWidget(self.canvas_routes)

        root.addLayout(left, 0)
        root.addLayout(right, 1)

    # --- Handlery GUI ------------------------------------------------------------

    @Slot()
    def on_browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Wybierz plik CSV", "", "CSV (*.csv);;Wszystkie pliki (*.*)"
        )
        if path:
            self.le_instance.setText(path)

    @Slot()
    def on_browse_outdir(self):
        path = QFileDialog.getExistingDirectory(self, "Wybierz folder wyjściowy")
        if path:
            self.le_outdir.setText(path)

    @Slot()
    def on_open_outdir(self):
        outdir = self.le_outdir.text().strip() or os.getcwd()
        try:
            if sys.platform.startswith("win"):
                os.startfile(outdir)
            elif sys.platform == "darwin":
                os.system(f"open '{outdir}'")
            else:
                os.system(f"xdg-open '{outdir}'")
        except Exception:
            QMessageBox.warning(self, "Folder", f"Nie udało się otworzyć folderu: {outdir}")

    def _collect(self) -> Optional[GAParams]:
        inst = self.le_instance.text().strip()
        if not inst or not os.path.exists(inst):
            QMessageBox.warning(self, "Instancja", "Wskaż istniejący plik CSV z instancją.")
            return None
        outdir = self.le_outdir.text().strip() or "out"
        return GAParams(
            instance_path=inst,
            outdir=outdir,
            pop=self.sb_pop.value(),
            gens=self.sb_gens.value(),
            pc=self.dsb_pc.value(),
            pm=self.dsb_pm.value(),
            alpha=self.dsb_alpha.value(),
            beta=self.dsb_beta.value(),
            max_vehicles=self.sb_vehicles.value(),
        )

    @Slot()
    def on_run(self):
        p = self._collect()
        if not p:
            return
        self.btn_run.setEnabled(False)
        self.lbl_status.setText("Liczenie…")
        self.progress.setRange(0, 0)
        self.log.append("\n——— START ——–")
        self.log.append(
            f"Instancja: {p.instance_path}\n"
            f"pop={p.pop}, gens={p.gens}, pc={p.pc}, pm={p.pm}, "
            f"alpha={p.alpha}, beta={p.beta}, max_vehicles={p.max_vehicles}"
        )
        self.canvas_conv.clear()
        self.canvas_routes.clear()

        self.worker = GAWorker(p)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    @Slot(dict)
    def on_finished(self, res: Dict[str, Any]):
        self.btn_run.setEnabled(True)
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        if not res.get("ok"):
            self.lbl_status.setText("Błąd")
            err = res.get("error", "")
            self.log.append("\n!!! Błąd wykonania !!!\n" + err)
            QMessageBox.critical(
                self, "Błąd", "Wystąpił wyjątek podczas działania. Szczegóły w logu."
            )
            return

        self.df = res["df"]
        self.routes = res["routes"]
        NV = res["NV"]
        self.stats = res["stats"]
        self.history = res["history"]

        self.lbl_status.setText("Zakończono.")
        self.log.append("Zakończono.")

        st = self.stats
        self.log.append(
            f"\n=== METRYKI ===\n"
            f"Liczba tras: {NV}\n"
            f"Dystans: {st.get('distance'):.2f}\n"
            f"Przeładowania: {st.get('overload'):.2f}\n"
            f"Spóźnienia: {st.get('lateness'):.2f}\n"
            f"Fitness: {st.get('fitness'):.2f}"
        )

        # historia fitnessu
        ax = self.canvas_conv.ax
        ax.clear()
        ax.plot(range(1, len(self.history) + 1), self.history)
        ax.set_xlabel("Generacja")
        ax.set_ylabel("Najlepszy fitness")
        ax.setTitle = "Historia GA"
        ax.grid(True, alpha=0.3)
        self.canvas_conv.draw_idle()

        # trasy
        self._plot_routes()
        self._save_outputs()

    def _plot_routes(self):
        if self.df is None or not self.routes:
            return
        xs = self.df["x"]
        ys = self.df["y"]
        ax = self.canvas_routes.ax
        ax.clear()
        ax.scatter(xs, ys, s=20)
        try:
            ax.scatter([xs.loc[0]], [ys.loc[0]], s=80, marker="s", label="Depot")
        except Exception:
            pass
        for idx, r in enumerate(self.routes, start=1):
            try:
                rx = [xs.loc[i] for i in r]
                ry = [ys.loc[i] for i in r]
                ax.plot(rx, ry, marker="o", linewidth=1)
            except Exception:
                continue
        ax.set_title("Trasy VRPTW")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, alpha=0.2)
        ax.set_aspect("equal", adjustable="box")
        self.canvas_routes.draw_idle()

    def _save_outputs(self):
        outdir = self.le_outdir.text().strip() or "out"
        os.makedirs(outdir, exist_ok=True)

        # history.png
        try:
            fig_hist = Figure(figsize=(6.4, 4.0), dpi=100)
            ax = fig_hist.add_subplot(111)
            ax.plot(range(1, len(self.history) + 1), self.history)
            ax.set_xlabel("Generacja")
            ax.set_ylabel("Najlepszy fitness")
            ax.set_title("Historia GA")
            ax.grid(True, alpha=0.3)
            fig_hist.tight_layout()
            fig_hist.savefig(os.path.join(outdir, "history.png"))
        except Exception:
            pass

        # routes.png
        try:
            fig_r = Figure(figsize=(6.4, 4.8), dpi=100)
            ax = fig_r.add_subplot(111)
            xs = self.df["x"]
            ys = self.df["y"]
            ax.scatter(xs, ys, s=20)
            try:
                ax.scatter([xs.loc[0]], [ys.loc[0]], s=80, marker="s", label="Depot")
            except Exception:
                pass
            for idx, r in enumerate(self.routes or [], start=1):
                try:
                    rx = [xs.loc[i] for i in r]
                    ry = [ys.loc[i] for i in r]
                    ax.plot(rx, ry, marker="o", linewidth=1)
                except Exception:
                    continue
            ax.set_title("Trasy VRPTW")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_aspect("equal", adjustable="box")
            fig_r.tight_layout()
            fig_r.savefig(os.path.join(outdir, "routes.png"))
        except Exception:
            pass


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
