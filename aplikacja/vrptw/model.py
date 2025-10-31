import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class Instance:
    data: pd.DataFrame
    distance: np.ndarray
    Q: int

@dataclass
class Route:
    customers: list[int]

@dataclass
class Solutions:
    routes: list[Route]

