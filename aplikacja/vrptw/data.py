import pandas as pd
import numpy as np

def load_instance(filename, Q=None):
    df = pd.read_csv(filename, sep=';')
    df = df.sort_values('id').reset_index(drop=True)

    if Q is None:
        Q = df[df['id'] == 0]['vehicle_capacity'].values[0]

    coords = df[['x', 'y']].values
    N = len(coords)
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)

    return df, D, Q
