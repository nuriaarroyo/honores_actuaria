# main_smoke_naive.py
import numpy as np
import pandas as pd
from pathlib import Path

from naive import NaivePortafolio  # hereda Portafolio con tus cambios

def main():
    # 1) cargar datos
    data_path = Path(__name__).with_name('data_clean_stock_data.csv')
    df = pd.read_csv(str(data_path), header=[0,1], index_col=0, parse_dates=True)

    # 2) definir ventanas (ejemplo)
    construct_start, construct_end = '2021-01-01', '2022-12-31'
    bt_train_start, bt_train_end   = '2019-01-01', '2020-12-31'
    bt_test_start, bt_test_end     = '2023-01-01', '2023-12-31'

    # 3) instanciar
    pf = NaivePortafolio(df)

    # 4) triple split
    pf.dividir(
        construct_start, construct_end,
        bt_train_start, bt_train_end,
        bt_test_start, bt_test_end
    )

    # 5) pesos naive en el universo de construcción
    n = len(pf._construct_tickers)
    assert n > 0, "Sin tickers en el tramo de construcción."
    pesos = np.ones(n) / n

    # 6) construir SOLO sobre CONSTRUCT (devuelve serie de retornos construct)
    rs_construct = pf.construir(pesos)

    # 7) evaluar con los mismos pesos en BT-train y BT-test
    rs_bt_train = pf.bt_train(pesos)
    rs_bt_test  = pf.bt_test(pesos)

    # 8) checks básicos (sin KPIs)
    print(f"[OK] Tickers en construct: {n}")
    print(f"[OK] Suma de pesos: {pesos.sum():.6f}")
    print(f"[OK] Retornos construct: {len(rs_construct) if rs_construct is not None else 0} observaciones")
    print(f"[OK] Retornos bt_train : {len(rs_bt_train) if rs_bt_train is not None else 0} observaciones")
    print(f"[OK] Retornos bt_test  : {len(rs_bt_test)  if rs_bt_test  is not None else 0} observaciones")

if __name__ == "__main__":
    main()
