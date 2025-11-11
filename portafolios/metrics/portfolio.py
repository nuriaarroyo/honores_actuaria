# portafolios/metrics/portfolio.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Tuple

# ============================================================
# 1) KPIs POR MOMENTOS (usa mu, cov, weights)
#    - Recomendado: mu = media de retornos simples (per-period)
#                   cov = covarianza (de simples o logs, sé consistente)
# ============================================================

def expected_return_from_moments(mu: pd.Series, weights: pd.Series, *,
                                 ann_factor: Optional[int] = None) -> float:
    """
    Retorno esperado: μ_p = w' μ (per-period si ann_factor=None).
    """
    w = weights.reindex(mu.index).fillna(0).values
    er = float(np.dot(w, mu.values))
    return er if ann_factor is None else er * ann_factor

def expected_volatility_from_moments(cov: pd.DataFrame, weights: pd.Series, *,
                                     ann_factor: Optional[int] = None) -> float:
    """
    Volatilidad esperada: σ_p = sqrt(w' Σ w).
    """
    cov = cov.reindex(index=weights.index, columns=weights.index).fillna(0).values
    w = weights.fillna(0).values
    var = float(w @ cov @ w)
    vol = np.sqrt(var) if var >= 0 else np.nan
    return vol if ann_factor is None else vol * np.sqrt(ann_factor)

def sharpe_from_moments(mu: pd.Series, cov: pd.DataFrame, weights: pd.Series, *,
                        rf_per_period: float = 0.0,
                        ann_factor: Optional[int] = None) -> float:
    """
    Sharpe por momentos: S = (μ_p - rf) / σ_p.
    rf_per_period en mismas unidades que μ (per-period si μ es per-period).
    """
    er = expected_return_from_moments(mu, weights, ann_factor=None)
    vol = expected_volatility_from_moments(cov, weights, ann_factor=None)
    if vol == 0 or np.isnan(vol):
        return float("nan")
    s = (er - rf_per_period) / vol
    return s if ann_factor is None else s * np.sqrt(ann_factor)

def risk_contributions_from_cov(cov: pd.DataFrame, weights: pd.Series, *,
                                ann_factor: Optional[int] = None,
                                as_fraction: bool = True) -> pd.Series:
    """
    Contribución al riesgo: RC_i = w_i * (Σ w)_i / σ_p  (fracción si as_fraction=True).
    """
    Σ = cov.reindex(index=weights.index, columns=weights.index).fillna(0).values
    if ann_factor is not None:
        Σ = Σ * ann_factor
    w = weights.fillna(0).values.reshape(-1, 1)
    port_vol = float(np.sqrt(w.T @ Σ @ w))
    if port_vol == 0 or np.isnan(port_vol):
        return pd.Series(np.nan, index=weights.index)
    mc = (Σ @ w)[:, 0]            # marginal contributions
    rc = (weights.values * mc) / (port_vol if as_fraction else 1.0)
    return pd.Series(rc, index=weights.index)

# ============================================================
# 2) KPIs POR TRAYECTORIA (requieren serie de retornos del portafolio)
#    - Útiles para MDD, Sortino, VaR/CVaR, TE/IR, Alpha/Beta
# ============================================================

def _align_rw(returns: pd.DataFrame, weights: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    returns = returns.reindex(columns=weights.index)
    weights = weights.reindex(returns.columns).fillna(0.0)
    return returns, weights

def portfolio_return_series(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """
    Serie de retornos del portafolio: r_p,t = R_t @ w  (per-period).
    """
    rets, w = _align_rw(returns, weights)
    return (rets @ w).dropna()

def max_drawdown(returns: pd.DataFrame, weights: pd.Series) -> float:
    rp = portfolio_return_series(returns, weights)
    nav = (1 + rp).cumprod()
    peak = nav.cummax()
    dd = (nav / peak - 1.0).min()
    return float(dd)

def downside_deviation(returns: pd.DataFrame, weights: pd.Series, *,
                       mar_per_period: float = 0.0,
                       ann_factor: Optional[int] = None) -> float:
    rp = portfolio_return_series(returns, weights)
    downside = np.clip(rp - mar_per_period, None, 0.0)
    dd = np.sqrt((downside ** 2).mean())
    return dd if ann_factor is None else dd * np.sqrt(ann_factor)

def sortino(returns: pd.DataFrame, weights: pd.Series, *,
            mar_per_period: float = 0.0,
            ann_factor: Optional[int] = None) -> float:
    er = portfolio_return_series(returns, weights).mean()
    dd = downside_deviation(returns, weights, mar_per_period=mar_per_period, ann_factor=None)
    if dd == 0 or np.isnan(dd):
        return float("nan")
    s = (er - mar_per_period) / dd
    return s if ann_factor is None else s * np.sqrt(ann_factor)

def calmar_from_moments(mu: pd.Series, cov: pd.DataFrame, weights: pd.Series, *,
                        ann_factor: Optional[int] = None,
                        returns_for_mdd: Optional[pd.DataFrame] = None) -> float:
    """
    Calmar = retorno (por momentos) / |MDD| (por trayectoria).
    Pasa 'returns_for_mdd' (per-period) para estimar MDD.
    """
    if returns_for_mdd is None:
        return float("nan")
    er = expected_return_from_moments(mu, weights, ann_factor=ann_factor)
    dd = max_drawdown(returns_for_mdd, weights)
    denom = abs(dd) if dd is not None else np.nan
    if denom == 0 or np.isnan(denom):
        return float("nan")
    return float(er / denom)

# VaR/CVaR (Gauss) por trayectoria
def var_gaussian(returns: pd.DataFrame, weights: pd.Series, *,
                 alpha: float = 0.95, ann_factor: Optional[int] = None, ddof: int = 1) -> float:
    from scipy.stats import norm
    rp = portfolio_return_series(returns, weights)
    mu, sigma = rp.mean(), rp.std(ddof=ddof)
    if ann_factor is not None:
        mu = mu * ann_factor
        sigma = sigma * np.sqrt(ann_factor)
    z = norm.ppf(1 - alpha)  # negativo (cola izquierda)
    var = -(mu + z * sigma)
    return float(max(var, 0.0))

def cvar_gaussian(returns: pd.DataFrame, weights: pd.Series, *,
                  alpha: float = 0.95, ann_factor: Optional[int] = None, ddof: int = 1) -> float:
    from scipy.stats import norm
    rp = portfolio_return_series(returns, weights)
    mu, sigma = rp.mean(), rp.std(ddof=ddof)
    if ann_factor is not None:
        mu = mu * ann_factor
        sigma = sigma * np.sqrt(ann_factor)
    z = norm.ppf(1 - alpha)  # negativo
    phi = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    es = -(mu + sigma * (phi / (1 - alpha)))
    return float(max(es, 0.0))

# Benchmark-based
def tracking_error(returns: pd.DataFrame, weights: pd.Series, benchmark: pd.Series, *,
                   ann_factor: Optional[int] = None, ddof: int = 1) -> float:
    rp = portfolio_return_series(returns, weights)
    joint = pd.concat([rp, benchmark], axis=1).dropna()
    if joint.shape[1] < 2 or joint.empty:
        return float("nan")
    diff = joint.iloc[:, 0] - joint.iloc[:, 1]
    te = diff.std(ddof=ddof)
    return te if ann_factor is None else te * np.sqrt(ann_factor)

def alpha_beta(returns: pd.DataFrame, weights: pd.Series, benchmark: pd.Series, *,
               rf_per_period: float = 0.0, ann_factor: Optional[int] = None, ddof: int = 1) -> Tuple[float, float]:
    rp = portfolio_return_series(returns, weights)
    joint = pd.concat([rp, benchmark], axis=1).dropna()
    if joint.shape[1] < 2 or joint.empty:
        return float("nan"), float("nan")
    y = (joint.iloc[:, 0] - rf_per_period).values
    x = (joint.iloc[:, 1] - rf_per_period).values
    if x.std(ddof=ddof) == 0:
        return float("nan"), float("nan")
    beta = float(np.cov(x, y, ddof=ddof)[0, 1] / np.var(x, ddof=ddof))
    alpha = float(y.mean() - beta * x.mean())
    if ann_factor is not None:
        alpha *= ann_factor
    return alpha, beta

def information_ratio(returns: pd.DataFrame, weights: pd.Series, benchmark: pd.Series, *,
                      ann_factor: Optional[int] = None, ddof: int = 1) -> float:
    rp = portfolio_return_series(returns, weights)
    joint = pd.concat([rp, benchmark], axis=1).dropna()
    if joint.shape[1] < 2 or joint.empty:
        return float("nan")
    diff = joint.iloc[:, 0] - joint.iloc[:, 1]
    mu, sd = diff.mean(), diff.std(ddof=ddof)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    ir = mu / sd
    return ir if ann_factor is None else ir * np.sqrt(ann_factor)

# ============================================================
# 3) WRAPPERS opcionales usando RETURNS completos
#    (por si a veces prefieres no armar mu/cov a mano)
# ============================================================

def expected_return(returns: pd.DataFrame, weights: pd.Series, *, ann_factor: Optional[int] = None) -> float:
    mu = returns.mean()
    return expected_return_from_moments(mu, weights, ann_factor=ann_factor)

def expected_volatility(returns: pd.DataFrame, weights: pd.Series, *, ann_factor: Optional[int] = None, ddof: int = 1) -> float:
    cov = returns.cov(ddof=ddof)
    return expected_volatility_from_moments(cov, weights, ann_factor=ann_factor)

def sharpe(returns: pd.DataFrame, weights: pd.Series, *, rf_per_period: float = 0.0, ann_factor: Optional[int] = None, ddof: int = 1) -> float:
    mu = returns.mean()
    cov = returns.cov(ddof=ddof)
    return sharpe_from_moments(mu, cov, weights, rf_per_period=rf_per_period, ann_factor=ann_factor)
