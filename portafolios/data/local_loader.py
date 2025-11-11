# portafolios/data/local_loader.py
from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Iterable, Optional

def local_loader(
    *,
    path: str | Path,
    tickers: Optional[Iterable[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    prefer_adj_close: bool = True,
    freq: Optional[str] = None
) -> pd.DataFrame:
    """
    Lee un CSV con formato estilo yfinance (MultiIndex: (Ticker, Field))
    y devuelve SOLO precios de cierre por ticker, con columnas = tickers.

    - Si hay 'Adj Close', lo prioriza; si no, usa 'Close'.
    - Filtra por fechas y tickers si se proporcionan.
    - Opcionalmente, remuestrea a 'freq' (ej. 'B') con forward-fill.

    Retorna: DataFrame (DatetimeIndex) con columnas = tickers y valores = precios.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    # Leemos asumiendo formato yfinance exportado (MultiIndex columnas)
    df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    df = df.sort_index()

    # Elegimos columnas de cierre por ticker
    close_cols = []
    for t, f in df.columns:
        if f == ("Adj Close" if prefer_adj_close else "Close"):
            close_cols.append((t, f))
    # Si no hubo "Adj Close", intentamos con "Close"
    if not close_cols:
        close_cols = [(t, f) for (t, f) in df.columns if f == "Close"]
        if not close_cols:
            raise ValueError("No se encontraron columnas 'Adj Close' ni 'Close' en el CSV.")

    df_close = df[close_cols].copy()
    # Dejar solo el nivel del ticker como nombre de columna
    df_close.columns = df_close.columns.get_level_values(0)

    # Filtrado por tickers (si se pide)
    if tickers is not None:
        keep = [t for t in tickers if t in df_close.columns]
        if not keep:
            raise ValueError("Ninguno de los tickers solicitados está en el CSV.")
        df_close = df_close[keep]

    # Filtrado por fechas (si se pide)
    if start:
        df_close = df_close[df_close.index >= pd.to_datetime(start)]
    if end:
        df_close = df_close[df_close.index <= pd.to_datetime(end)]

    # Rellenos simples
    df_close = df_close.ffill().bfill()

    # Remuestreo opcional
    if freq:
        df_close = df_close.asfreq(freq, method="pad")

    return df_close
