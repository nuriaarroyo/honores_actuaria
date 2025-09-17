#el papá 
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


class Portafolio:
    def __init__(self, data: pd.DataFrame, nombre: str = "Portafolio"):  #df
        self.data = data.copy()
        self.nombre = nombre
        self.weights = None
        self.expected_returns = None
        self.volatility = None
        self.portfolioreturns = None
        
    def dividir(self,
                start_train_date: str, end_train_date: str, #train data 
                start_bt_date: str, end_bt_date: str ): #backtesting data
        # loc
        self.data_train = self.data.loc[start_train_date:end_train_date]
        self.data_bt = self.data.loc[start_bt_date:end_bt_date]
        
        # Calcular retornos esperados y volatilidad del período de entrenamiento
        self.expected_returns = self.calculate_expected_returns(self.data_train)
        self.volatility = self.calculate_volatility(self.data_train)
        
        return self.data_train

    def calculate_expected_returns(self, data):
        closeprices = data.xs('Close', axis=1, level=1)
        returns = closeprices.pct_change().dropna()
        return returns.mean()

    def calculate_volatility(self, data):
        closeprices = data.xs('Close', axis=1, level=1)
        returns = closeprices.pct_change().dropna()
        return returns.std()

    def construir(self, pesos):
        closeprices = self.data.xs('Close', axis=1, level=1)
        if len(pesos) != len(closeprices.columns):
            raise ValueError("Length of weights must match number of assets.")
        self.weights = np.array(pesos)
        self.portfolio_returns()

    def portfolio_returns(self):
        closeprices = self.data.xs('Close', axis=1, level=1)
        returns = closeprices.pct_change().dropna()
        self.portfolioreturns = returns @ self.weights

    def portfolio_return(self):
        closeprices = self.data.xs('Close', axis=1, level=1)
        returns = closeprices.pct_change().dropna()
        portfolio_ret = (returns @ self.weights).mean()
        return portfolio_ret

    def portfolio_volatility(self):
        closeprices = self.data.xs('Close', axis=1, level=1)
        returns = closeprices.pct_change().dropna()
        portfolio_vol = (returns @ self.weights).std()
        return portfolio_vol

    def mc(self,pesos):
        closeprices = self.data.xs('Close', axis=1, level=1)
        logreturns = np.log(closeprices / closeprices.shift(1)).dropna()

        mean_returns = logreturns.mean()
        cov_matrix = logreturns.cov()
        n_simulations = 5000
        n_days = 252
        simulations = np.zeros((n_simulations, n_days))

        for i in range(n_simulations):
            daily_returns = np.random.multivariate_normal(mean_returns, cov_matrix, size=n_days)
            price_paths = np.exp(np.cumsum(daily_returns, axis=0))
            price_paths = price_paths * np.array([1] * len(self.tickers))
            portfolio_value = price_paths @ pesos
            simulations[i, :] = portfolio_value

        fig = go.Figure()
        for i in range(n_simulations):
            fig.add_trace(go.Scatter(
                x=np.arange(n_days),
                y=simulations[i, :],
                mode='lines',
                line=dict(color='blue'),
                opacity=0.1
            ))

        fig.update_layout(
            title='Simulación Monte Carlo del Portafolio',
            xaxis_title='Días',
            yaxis_title='Valor del Portafolio',
            showlegend=False
        )
        fig.show()

    def bt(self,pesos):
        bt_returns = self.data_bt.xs('Close', axis=1, level=1).pct_change().dropna() @ pesos
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=bt_returns.index,
            y=(1 + bt_returns).cumprod(),
            mode='lines',
            name='Backtest Portfolio'
        ))
        fig.update_layout(title='Crecimiento del Portafolio en el Backtest')
        fig.show()
    
    def risk(self,pesos):
        pass

    def mostrar_pesos(self,pesos):
        tickers = self.data.columns.get_level_values(0).unique().to_list()
        for ticker, peso in zip(tickers, pesos):
            print(f"acción: {ticker}, peso asignado: {peso:.4f}")
        pass
    
    def pastel(self,pesos):
        tickers = self.data.columns.get_level_values(0).unique().to_list()
        n_activos = len(tickers)
        
        # Ajustar tamaño del gráfico según número de activos
        if n_activos <= 20:
            width, height = 1200, 800
            text_size = 12
            legend_size = 10
        elif n_activos <= 40:
            width, height = 1400, 900
            text_size = 11
            legend_size = 9
        else:
            width, height = 1600, 1000
            text_size = 10
            legend_size = 8
        
        # Crear gráfico de pastel con Plotly
        # Filtrar solo activos con peso > 0 para mejor visualización
        pesos_filtrados = []
        tickers_filtrados = []
        
        for ticker, peso in zip(tickers, pesos):
            if peso > 0.001:  # Solo activos con peso > 0.1%
                pesos_filtrados.append(peso)
                tickers_filtrados.append(ticker)
        
        # Detectar si es un portfolio tipo Naive (todos con peso similar)
        pesos_unicos = set([round(p, 4) for p in pesos_filtrados])
        es_naive = len(pesos_unicos) == 1 and len(pesos_filtrados) > 20
        
        # Si hay muchos activos con peso 0, ajustar el agujero central
        if len(pesos_filtrados) < len(tickers) * 0.5:
            hole_size = 0.1  # Agujero más pequeño para mejor visualización
        elif es_naive:
            hole_size = 0.2  # Agujero intermedio para portfolios muy diversificados
        else:
            hole_size = 0.3
        
        # Configurar colores y efectos especiales para portfolios Naive
        if es_naive:
            # Para portfolios equiponderados, usar colores alternados para mejor distinción
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] * (len(tickers_filtrados) // 5 + 1)
            colors = colors[:len(tickers_filtrados)]
            pull_effect = [0.05] * len(tickers_filtrados)  # Ligero efecto pull para todos
        else:
            colors = None  # Usar colores por defecto
            pull_effect = [0.1 if p > 0.05 else 0 for p in pesos_filtrados]  # Destacar activos importantes
        
        fig = go.Figure(data=[go.Pie(
            labels=tickers_filtrados,
            values=pesos_filtrados,
            hole=hole_size,
            textinfo='label+percent',
            textposition='outside',
            textfont=dict(size=text_size),
            marker=dict(
                line=dict(color='white', width=1),
                colors=colors
            ),
            rotation=45,  # Rotar para mejor distribución
            pull=pull_effect
        )])
        
        # Crear título con información sobre activos excluidos
        activos_excluidos = len(tickers) - len(tickers_filtrados)
        if activos_excluidos > 0:
            title_text = f'Pesos del {self.nombre}<br><sub>Se muestran {len(tickers_filtrados)} de {len(tickers)} activos (excluidos {activos_excluidos} con peso ≤ 0.1%)</sub>'
        elif es_naive:
            peso_uniforme = pesos_filtrados[0] if pesos_filtrados else 0
            title_text = f'Pesos del {self.nombre}<br><sub>Portfolio Equiponderado: {len(tickers_filtrados)} activos con peso uniforme de {peso_uniforme:.1%} cada uno</sub>'
        else:
            title_text = f'Pesos del {self.nombre}'
        
        fig.update_layout(
            title=title_text,
            showlegend=True,
            width=width,
            height=height,
            title_font_size=20,
            font=dict(size=legend_size),
            margin=dict(l=50, r=50, t=100, b=50),  # Aumentar margen superior para el subtítulo
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02,
                font=dict(size=legend_size)
            )
        )
        
        # Guardar como HTML para abrir en navegador
        safe_name = self.nombre.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("\\", "_")
        plots_dir = Path(__file__).resolve().parents[1] / 'plots'
        plots_dir.mkdir(exist_ok=True)
        out_path = plots_dir / f'portfolio_pastel_{safe_name}.html'
        fig.write_html(str(out_path))
        
        # Mostrar en navegador
        fig.show()
        pass

    def barras(self,pesos):
        tickers = self.data.columns.get_level_values(0).unique().to_list()
        
        # Crear gráfico de barras con Plotly
        fig = go.Figure(data=[go.Bar(
            x=tickers,
            y=pesos,
            marker_color='cornflowerblue',
            text=[f'{p:.1%}' for p in pesos],
            textposition='auto'
        )])
        
        fig.update_layout(
            title=f'Distribución de Pesos del {self.nombre}',
            xaxis_title='Activos',
            yaxis_title='Peso',
            width=1400,
            height=700,
            title_font_size=20,
            xaxis_title_font_size=16,
            yaxis_title_font_size=16,
            showlegend=False
        )
        
        # Guardar como HTML
        safe_name = self.nombre.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("\\", "_")
        plots_dir = Path(__file__).resolve().parents[1] / 'plots'
        plots_dir.mkdir(exist_ok=True)
        out_path = plots_dir / f'portfolio_barras_{safe_name}.html'
        fig.write_html(str(out_path))
        
        # Mostrar en navegador
        fig.show()
        pass

    def bubbleplot_matplotlib(self, weights):
        tickers = self.data.columns.get_level_values(0).unique()
        closeprices = self.data.xs('Close', axis=1, level=1)
        returns = closeprices.pct_change().dropna()

        ereturns = returns.mean()
        vols = returns.std()

        # Crear gráfico de burbujas con Plotly
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=ereturns.values,
            y=vols.values,
            mode='markers+text',
            marker=dict(
                size=weights * 2000,
                color='cornflowerblue',
                opacity=0.7,
                line=dict(color='black', width=1)
            ),
            text=tickers,
            textposition='middle center',
            name='Activos'
        ))
        
        fig.update_layout(
            title=f'Bubble Plot: Returns vs Volatility - {self.nombre}',
            xaxis_title='Expected Returns',
            yaxis_title='Volatility',
            width=1200,
            height=900,
            title_font_size=20,
            xaxis_title_font_size=16,
            yaxis_title_font_size=16,
            showlegend=False
        )
        
        # Guardar como HTML
        safe_name = self.nombre.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("\\", "_")
        plots_dir = Path(__file__).resolve().parents[1] / 'plots'
        plots_dir.mkdir(exist_ok=True)
        out_path = plots_dir / f'bubble_plot_{safe_name}.html'
        fig.write_html(str(out_path))
        
        # Mostrar en navegador
        fig.show()

    def correlacion_y_dendograma(self):
        close = self.data.xs('Close', axis=1, level=1)
        corr_original = close.pct_change().dropna().corr()

        # Matriz de correlaciones original con Plotly
        # Solo mostrar valores de correlación para activos importantes o en hover
        # Crear matriz de texto solo para correlaciones extremas
        text_matrix = corr_original.copy()
        text_matrix = text_matrix.applymap(lambda x: f'{x:.2f}' if abs(x) > 0.7 else '')  # Solo mostrar correlaciones > 0.7 o < -0.7
        
        fig1 = go.Figure(data=go.Heatmap(
            z=corr_original.values,
            x=corr_original.columns,
            y=corr_original.index,
            colorscale='RdBu',
            zmid=0,
            text=text_matrix.values,  # Mostrar solo correlaciones importantes
            texttemplate="%{text}",
            textfont={"size": 9, "color": "black"},
            hovertemplate='<b>%{y} vs %{x}</b><br>' +
                         'Correlación: %{z:.3f}<br>' +
                         '<extra></extra>',
            hoverongaps=False
        ))
        
        fig1.update_layout(
            title="Matriz de Correlaciones (Original)<br><sub>Hover sobre las celdas para ver valores exactos</sub>",
            width=1000,
            height=1000,
            xaxis=dict(
                tickangle=45,
                tickfont=dict(size=8),
                showticklabels=True
            ),
            yaxis=dict(
                tickfont=dict(size=8),
                showticklabels=True
            )
        )
        
        # Guardar y mostrar
        plots_dir = Path(__file__).resolve().parents[1] / 'plots'
        plots_dir.mkdir(exist_ok=True)
        out_path1 = plots_dir / 'correlacion_original.html'
        fig1.write_html(str(out_path1))
        fig1.show()

        # Reordenar usando cuasidiagonalización
        dist = 1 - corr_original
        linked = linkage(squareform(dist), method='single')
        dendro = dendrogram(linked, no_plot=True)
        order = dendro['leaves']
        corr_reordered = corr_original.iloc[order, :].T.iloc[order, :]

        # Matriz de correlaciones reorganizada
        # Crear matriz de texto solo para correlaciones extremas
        text_matrix_reordered = corr_reordered.copy()
        text_matrix_reordered = text_matrix_reordered.applymap(lambda x: f'{x:.2f}' if abs(x) > 0.7 else '')  # Solo mostrar correlaciones > 0.7 o < -0.7
        
        fig2 = go.Figure(data=go.Heatmap(
            z=corr_reordered.values,
            x=corr_reordered.columns,
            y=corr_reordered.index,
            colorscale='RdBu',
            zmid=0,
            text=text_matrix_reordered.values,  # Mostrar solo correlaciones importantes
            texttemplate="%{text}",
            textfont={"size": 9, "color": "black"},
            hovertemplate='<b>%{y} vs %{x}</b><br>' +
                         'Correlación: %{z:.3f}<br>' +
                         '<extra></extra>',
            hoverongaps=False
        ))
        
        fig2.update_layout(
            title="Matriz de Correlaciones (Reordenada)<br><sub>Después de la cuasidiagonalización - Hover sobre las celdas para ver valores exactos</sub>",
            width=1000,
            height=1000,
            xaxis=dict(
                tickangle=45,
                tickfont=dict(size=8),
                showticklabels=True
            ),
            yaxis=dict(
                tickfont=dict(size=8),
                showticklabels=True
            )
        )
        
        # Guardar y mostrar
        plots_dir = Path(__file__).resolve().parents[1] / 'plots'
        plots_dir.mkdir(exist_ok=True)
        out_path2 = plots_dir / 'correlacion_reordenada.html'
        fig2.write_html(str(out_path2))
        fig2.show()

        # Dendograma con Plotly usando scipy.dendrogram
        # Crear el dendograma real usando scipy
        from scipy.cluster.hierarchy import dendrogram
        
        # Crear figura para el dendograma
        fig3 = go.Figure()
        
        # Obtener las coordenadas del dendograma
        dendro_data = dendrogram(linked, no_plot=True, labels=close.columns.tolist())
        
        # Extraer las coordenadas de las líneas del dendograma
        icoord = dendro_data['icoord']  # Coordenadas x de las líneas
        dcoord = dendro_data['dcoord']  # Coordenadas y de las líneas
        
        # Agregar cada línea del dendograma
        for i in range(len(icoord)):
            # Convertir coordenadas de scipy a coordenadas de plotly
            x_coords = icoord[i]
            y_coords = dcoord[i]
            
            # Agregar línea horizontal
            fig3.add_trace(go.Scatter(
                x=x_coords,
                y=[y_coords[1], y_coords[1]],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Agregar líneas verticales
            fig3.add_trace(go.Scatter(
                x=[x_coords[0], x_coords[0]],
                y=[y_coords[0], y_coords[1]],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig3.add_trace(go.Scatter(
                x=[x_coords[2], x_coords[2]],
                y=[y_coords[2], y_coords[1]],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Agregar etiquetas de los activos en el eje X
        leaf_positions = dendro_data['leaves']
        leaf_labels = [close.columns[i] for i in leaf_positions]
        
        fig3.add_trace(go.Scatter(
            x=list(range(len(leaf_labels))),
            y=[0] * len(leaf_labels),
            mode='text',
            text=leaf_labels,
            textposition='bottom center',
            textfont=dict(size=8, color='black'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig3.update_layout(
            title="Dendograma de Clustering Jerárquico<br><sub>Muestra la estructura de agrupación de activos por correlación</sub>",
            xaxis_title="Activos (ordenados por clustering)",
            yaxis_title="Distancia de Correlación",
            width=1400,
            height=800,
            showlegend=False,
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-1, len(leaf_labels)]
            ),
            yaxis=dict(
                showgrid=True,
                zeroline=True,
                zerolinecolor='lightgray',
                gridcolor='lightgray'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Guardar y mostrar
        plots_dir = Path(__file__).resolve().parents[1] / 'plots'
        plots_dir.mkdir(exist_ok=True)
        out_path3 = plots_dir / 'dendograma.html'
        fig3.write_html(str(out_path3))
        fig3.show()

    @property
    def retornos(self):
        closeprices = self.data.xs('Close', axis=1, level=1)
        return closeprices.pct_change().dropna()

    @property
    def tickers(self):
        return self.data.columns.get_level_values(0).unique().to_list() 