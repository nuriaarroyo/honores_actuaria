# portafolios/constructores/naive.py
from __future__ import annotations
import pandas as pd
from typing import Tuple, Dict, Any

class Naive:
    nombre = "naive_1_over_n"
    def optimizar(self, returns: pd.DataFrame, **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        w = pd.Series(1.0 / len(returns.columns), index=returns.columns)
        return w, {"n_assets": len(returns.columns)}