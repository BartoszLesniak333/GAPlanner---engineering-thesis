import numpy as np
import pandas as pd 
from vrptw.model import Instance

def load_instance(csv_path: str, Q: int | None = None) -> Instance:
 
    df = pd.read_csv(csv_path, sep=";")  
    df.columns = [c.strip().lower() for c in df.columns]

    if Q is None:
        if 'vehicle_capacity' in df.columns:
            depot_row = df[df['id'] == 0].iloc[0]
            Q = int(depot_row['vehicle_capacity'])
        else: 
            return ValueError("Brak ładowności pojazdu - Q")
        
    # -- Macierz euklidesowa -- #
    cords = df[['x','y']].to_numpy()
    dist = np.linalg.norm(cords[:, None, :] - cords[None, :, :], axis=2)
    return Instance(data=df,distance=dist, Q=Q)