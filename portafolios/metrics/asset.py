# portafolios/metrics/asset.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional

def returns_simple(prices: pd.DataFrame) -> pd.DataFrame:
    """R_t = P_t / P_{t-1} - 1"""
    return prices.pct_change().dropna()

def returns_log(prices: pd.DataFrame) -> pd.DataFrame:
    """r_t = ln(P_t) - ln(P_{t-1})"""
    return np.log(prices).diff().dropna()

def mean_return(returns: pd.DataFrame, ann_factor: Optional[int] = None) -> pd.Series:
    """
    Media por periodo del DataFrame de rendimientos.
    - Si ann_factor is None → per-period mean (N-periodos, sin anualizar).
    - Si ann_factor es int  → anualiza: mean * ann_factor.
    """
    mu = returns.mean()
    return mu if ann_factor is None else mu * ann_factor

def volatility(returns: pd.DataFrame, ann_factor: Optional[int] = None, ddof: int = 1) -> pd.Series:
    """
    Volatilidad (desv. estándar) del DataFrame de rendimientos.
    - Si ann_factor is None → per-period std (sobre N observaciones de la ventana).
    - Si ann_factor es int  → anualiza: std * sqrt(ann_factor).
    """
    sigma = returns.std(ddof=ddof)
    return sigma if ann_factor is None else sigma * np.sqrt(ann_factor)
