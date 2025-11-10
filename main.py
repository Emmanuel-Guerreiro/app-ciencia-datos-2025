import streamlit as st

st.markdown("# Aplicación ciencia de datos")


st.image("static/pato.png", width=400)


st.markdown(
    """
### Autores
Guerreiro, Emmanuel 47262

Izuel, Tomás 47700
"""
)

st.markdown(
    """
## Descripción técnica

Sistema de predicción de variación de precio de Bitcoin basado en análisis de sentimiento de tweets por Donald Trump.

### Objetivo
Predecir la variación del precio de BTC en las próximas 24 horas (`btc_delta_24h`) mediante análisis de contenido textual de tweets.

### Modelo de clasificación

#### Algoritmo: XGBoost (Extreme Gradient Boosting)
Ensemble de árboles de decisión basado en gradient boosting. Construcción secuencial de árboles donde cada uno corrige errores del anterior.

#### Configuración del modelo
- **Tipo:** Clasificación multiclase (4 clases)
- **Función objetivo:** `multi:softmax`
  - Clasificación multiclase con salida directa de clase
  - Minimiza pérdida logarítmica multiclase
- **Métrica de evaluación:** `mlogloss` (multi-class log loss)

#### Hiperparámetros principales
- **`n_estimators: 1000`**
  - Número de árboles en el ensemble
  - Valor alto para capturar patrones complejos en datos de mercado
  - Mitiga underfitting en dataset con features multidimensionales (16+ categorías de moderación)

- **`max_depth: 10`**
  - Profundidad máxima de cada árbol
  - Permite capturar interacciones no lineales entre features textuales y temporales
  - Balance entre complejidad (evitar overfitting) y capacidad de modelado

- **`learning_rate: 0.1`**
  - Tasa de aprendizaje para shrinkage
  - Valor moderado: compromiso entre velocidad de convergencia y estabilidad
  - Reduce contribución de cada árbol para regularización implícita

- **`random_state: 42`**
  - Semilla para reproducibilidad
  - Garantiza consistencia en splits y muestreo

- **`n_jobs: -1`**
  - Paralelización completa (usa todos los cores)
  - Acelera entrenamiento e inferencia

#### Clases predichas (discretización de `btc_delta_24h`)
- **Clase 0:** `< -0.06` → Caída fuerte (>6%)
- **Clase 1:** `[-0.06, 0]` → Caída leve a estable
- **Clase 2:** `(0, 0.002]` → Subida leve (<0.2%)
- **Clase 3:** `> 0.002` → Subida fuerte (>0.2%)

### Datasets
- **Tweets procesados:** `tweets-processed.csv` (~25,935 entradas)
  - Incluye análisis de moderación de texto (Google Cloud Language API)
  - 16 categorías de moderación: Toxic, Insult, Profanity, Derogatory, Sexual, Death/Harm, Violent, Firearms, Public Safety, Health, Religion, Drugs, War, Politics, Finance, Legal
- **Datos históricos BTC:** `btc_15m_data_2018_to_2025.csv`
  - Datos de precio cada 15 minutos (2018-2025)
  - Filtrado a 16:00 hrs para análisis diario

### Features
- **Textuales:** Scores de moderación por categoría (valores 0-1)
- **Engagement:** favorites, retweets
- **Temporales:** hora normalizada, día de semana, is_weekend
- **Bitcoin contextuales:** btc_delta_24h, btc_delta_48h, btc_tweet_day
- **Engineered:** btc_delta_24h_pct, btc_delta_48h_pct

### Pipeline de preprocesamiento
1. Normalización de nombres de columnas a snake_case
2. Feature engineering manual (temporales + ratios BTC)
3. Transformación con ColumnTransformer (sklearn)
4. Eliminación de columnas target (btc_24h_after, btc_48h_after)

### Visualizaciones
- Scatter plot: Análisis de correlación entre variables de moderación y variación BTC
- Boxplot: Distribución de variables por clase real
- Serie temporal: Comparación de pendiente predicha vs real para tweets específicos

"""
)
