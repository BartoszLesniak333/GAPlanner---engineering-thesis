import numpy as np
import pandas as pd
from vrptw.model import Instance

def load_instance(csv_path: str, Q: int | None = None) -> Instance:
    # 1) wczytanie z poprawnym separatorem i normalizacja nazw kolumn
    df = pd.read_csv(csv_path, sep=';')
    df.columns = [c.strip().lower() for c in df.columns]

    # 2) poprawne typy
    df['id'] = pd.to_numeric(df['id'], errors='raise').astype(int)
    for c in ['x','y','demand','ready','due','service','vehicle_capacity']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # 3) indeks = id i sortowanie po id (kluczowe dla spójności z macierzą)
    df = df.set_index('id', drop=False).sort_index()

    # 4) pojemność Q
    if Q is None:
        if 'vehicle_capacity' in df.columns:
            if 0 not in df.index:
                raise ValueError("Brak depotu id=0 w danych.")
            depot_Q = df.loc[0, 'vehicle_capacity']
            if pd.isna(depot_Q):
                raise ValueError("Brak vehicle_capacity w wierszu depotu.")
            Q = int(depot_Q)
        else:
            raise ValueError("Brak kolumny 'vehicle_capacity' i nie podano Q.")
    # (opcjonalnie wypełnij NaN u klientów kopią Q, żeby nigdy nie wycinać wierszy)
    if 'vehicle_capacity' in df.columns:
        df['vehicle_capacity'] = df['vehicle_capacity'].fillna(Q)

    # 5) macierz euklidesowa indeksowana **po ID** (0..max_id)
    max_id = int(df.index.max())
    N = max_id + 1
    D = np.zeros((N, N), dtype=float)

    # mapa współrzędnych po ID
    X = df['x'].to_dict()
    Y = df['y'].to_dict()

    present_ids = set(df.index.tolist())
    for i in present_ids:
        xi, yi = X[i], Y[i]
        for j in present_ids:
            dx, dy = xi - X[j], yi - Y[j]
            D[i, j] = (dx*dx + dy*dy) ** 0.5

    return Instance(data=df, distance=D, Q=Q)
