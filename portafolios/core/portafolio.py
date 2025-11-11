# portafolios/core/portafolio.py
from __future__ import annotations
import pandas as pd
from typing import Iterable, Optional, Callable, Any
from ..metrics import asset as am

class Portfolio:
    """
    Versión mínima 1:
      - Guarda tickers, rango de fechas y precios (close/adjclose).
      - preparar_datos() llama al loader y llena self.prices y self.tickers.
    """

    """
    Mínimo viable 2:
      - preparar_datos() -> deja prices y tickers listos.
      - construir(constructor, t0, lookback, **kwargs) -> calcula returns y weights.
      - evaluar("single_period", ...) -> evalúa buy&hold en horizonte si hay datos.
    """

    def __init__(
        self,
        *,
        tickers: Optional[Iterable[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        loader: Optional[Callable[..., pd.DataFrame]] = None,
        loader_kwargs: Optional[dict[str, Any]] = None,
        freq: Optional[str] = "B"):
        #para guaradar
        self.tickers = list(tickers) if tickers is not None else None
        self.start = pd.Timestamp(start) if start else None
        self.end = pd.Timestamp(end) if end else None
        self.freq = freq
        self.loader = loader
        self.loader_kwargs = loader_kwargs or {}
        self.prices: Optional[pd.DataFrame] = None  # SOLO close prices en esta fase
        self.asset_returns = None
        self.asset_log_returns = None
        self.weights = None
        self.asset_vol = None
        self.asset_vol_lr = None
        self.asset_mean_ret =None
        self.asset_mean_lr = None
        self.info = {}

    def preparar_datos(self) -> "Portfolio":
        """
        Llama al loader y pobla:
          - self.prices: DataFrame de precios de cierre por ticker
          - self.tickers: lista final de tickers (si no se pasó al inicio)
        """
        if self.loader is None:
            raise ValueError("Debes proporcionar un 'loader' que devuelva precios de cierre.")

        # El loader espera argumentos por nombre
        df = self.loader(
            tickers=self.tickers,
            start=(self.start.isoformat() if self.start is not None else None),
            end=(self.end.isoformat() if self.end is not None else None),
            freq=self.freq,
            **self.loader_kwargs
        )

        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("El loader debe devolver un DataFrame con DatetimeIndex.")
        if df.empty:
            raise ValueError("El loader devolvió un DataFrame vacío.")

        self.prices = df.sort_index()
        # Si no definiste tickers al crear el objeto, toma los del DF
        if self.tickers is None:
            self.tickers = list(self.prices.columns)

        self.asset_returns     = am.returns_simple(self.prices)
        self.asset_log_returns = am.returns_log(self.prices)
        self.asset_vol         = am.volatility(self.asset_returns)
        self.asset_vol_lr        = am.volatility(self.asset_log_returns)
        self.asset_mean_ret = am.mean_return(self.asset_returns)
        self.asset_mean_lr = am.mean_return(self.asset_log_returns)
        

        return self
    
        # portafolios/core/portafolio.py (fragmento de construir)


    def construir(self, constructor,**kwargs) -> "Portfolio":
        if self.asset_returns is None:
            raise RuntimeError("Llama primero a preparar_datos().")
        
        w, meta = constructor.optimizar(self.asset_returns, **kwargs)
        self.weights = w.reindex(self.asset_returns.columns).fillna(0).sort_index()

        self.info.update({
            "constructor": getattr(constructor, "nombre", str(constructor)),
            **(meta or {})
            
        })
        return self


    def evaluar(self, metodo="single_period", **kwargs):
        if self.weights is None:
            raise RuntimeError("Primero construye el portafolio (no hay weights).")
        if metodo == "single_period":
            from ..eval.single_period import evaluar_once
            return evaluar_once(self, **kwargs)
        raise ValueError(f"Método de evaluación no soportado: {metodo}")
