# la primogenita 
import numpy as np 
import pandas as pd

from portafolio import Portafolio 


class NaivePortafolio(Portafolio):
    def __init__(self, data: pd.DataFrame):
        super().__init__(data, "Portafolio Naive (1/N)")
    
    def construir(self,
                  start_train_date: str, end_train_date: str, start_bt_date: str, end_bt_date: str ):
        # Dividir los datos y calcular retornos esperados y volatilidad
        self.dividir(start_train_date, end_train_date, start_bt_date, end_bt_date)
        
        # Obtener tickers del período de entrenamiento
        tickers = self.data_train.columns.get_level_values(0).unique().to_list()
        n = len(tickers)
        weightseach = 1/n
        pesos = np.full(n, weightseach)
        
        # Asignar los pesos al portafolio
        super().construir(pesos)
        
        return pesos
    

