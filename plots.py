import streamlit as st
import pandas as pd
import cloudpickle
import xgboost as xgb
from lib import load_inference_model
from plotslib import plot_text_analysis_scatter, calculate_quadrant_stats, plot_btc_price_comparison, plot_variable_by_class_boxplot

st.markdown("# Página de Gráficos 🎈")

# Define available variables
text_analysis_vars = [
    "Toxic",
    "Insult",
    "Profanity",
    "Derogatory",
    "Sexual",
    "Death, Harm & Tragedy",
    "Violent",
    "Firearms & Weapons",
    "Public Safety",
    "Health",
    "Religion & Belief",
    "Illicit Drugs",
    "War & Conflict",
    "Politics",
    "Finance",
    "Legal",
    "favorites",
    "retweets"
]

# Variables that are probabilities (0-1 range)
probability_vars = [
    "Toxic",
    "Insult",
    "Profanity",
    "Derogatory",
    "Sexual",
    "Death, Harm & Tragedy",
    "Violent",
    "Firearms & Weapons",
    "Public Safety",
    "Health",
    "Religion & Belief",
    "Illicit Drugs",
    "War & Conflict",
    "Politics",
    "Finance",
    "Legal"
]

# Create two columns for variable selection
col1, col2 = st.columns(2)

with col1:
    x_var = st.selectbox(
        "Seleccionar variable del eje X",
        options=text_analysis_vars,
        index=0  # Default to "Toxic"
    )

with col2:
    y_var = st.selectbox(
        "Seleccionar variable del eje Y",
        options=text_analysis_vars,
        index=1  # Default to "Insult"
    )

# Sample size selector
st.markdown("### Muestreo de Datos")
sample_size = st.slider(
    "Número de entradas aleatorias a mostrar",
    min_value=100,
    max_value=5000,
    value=1000,
    step=100,
    help="Selecciona un tamaño de muestra más pequeño para mejorar el rendimiento. El dataset contiene ~25,935 entradas."
)

# Create two columns for threshold selection
st.markdown("### Configuración de Umbrales")
col3, col4 = st.columns(2)

# Determine threshold ranges based on variable type
x_is_prob = x_var in probability_vars
y_is_prob = y_var in probability_vars

# Load data once if needed for non-probability variables
df_thresholds = None
if not x_is_prob or not y_is_prob:
    df_thresholds = pd.read_csv("static/tweets-processed.csv")

with col3:
    if x_is_prob:
        x_threshold = st.number_input(
            f"Umbral del eje X ({x_var})",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Umbral de línea vertical para crear cuadrantes"
        )
    else:
        # For favorites/retweets, use a larger range
        x_max = float(df_thresholds[x_var].max())
        x_threshold = st.number_input(
            f"Umbral del eje X ({x_var})",
            min_value=0.0,
            max_value=x_max,
            value=float(df_thresholds[x_var].quantile(0.5)),  # Default to median
            step=max(1.0, x_max / 100),
            help="Umbral de línea vertical para crear cuadrantes"
        )

with col4:
    if y_is_prob:
        y_threshold = st.number_input(
            f"Umbral del eje Y ({y_var})",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Umbral de línea horizontal para crear cuadrantes"
        )
    else:
        # For favorites/retweets, use a larger range
        y_max = float(df_thresholds[y_var].max())
        y_threshold = st.number_input(
            f"Umbral del eje Y ({y_var})",
            min_value=0.0,
            max_value=y_max,
            value=float(df_thresholds[y_var].quantile(0.5)),  # Default to median
            step=max(1.0, y_max / 100),
            help="Umbral de línea horizontal para crear cuadrantes"
        )

# Display the scatter plot with selected variables and thresholds
chart, plot_df = plot_text_analysis_scatter(
    x_var=x_var,
    y_var=y_var,
    x_threshold=x_threshold,
    y_threshold=y_threshold,
    sample_size=sample_size
)
st.altair_chart(chart, use_container_width=True)

# Calculate and display quadrant statistics
if x_threshold is not None and y_threshold is not None:
    st.markdown("### Estadísticas de Cuadrantes")
    quadrant_stats = calculate_quadrant_stats(
        plot_df, x_var, y_var, x_threshold, y_threshold
    )
    st.dataframe(quadrant_stats, use_container_width=True, hide_index=True)
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rojo (≤0)", int(quadrant_stats['Rojo (≤0)'].sum()))
    with col2:
        st.metric("Total Verde (>0)", int(quadrant_stats['Verde (>0)'].sum()))
    with col3:
        st.metric("Total de Puntos", int(quadrant_stats['Total'].sum()))
    with col4:
        if quadrant_stats['Total'].sum() > 0:
            green_pct = (quadrant_stats['Verde (>0)'].sum() / quadrant_stats['Total'].sum()) * 100
            st.metric("% Verde", f"{green_pct:.1f}%")

# --- Boxplot by Actual Class ---
st.markdown("---")
st.markdown("## Distribución de Variables por Clase Real")

# Variable selector
selected_var_boxplot = st.selectbox(
    "Seleccionar variable para análisis",
    options=text_analysis_vars,
    index=0,
    key="boxplot_var"
)

# Sample size selector for boxplot
sample_size_boxplot = st.slider(
    "Número de entradas aleatorias a analizar",
    min_value=100,
    max_value=2000,
    value=500,
    step=100,
    key="boxplot_sample",
    help="Selecciona un tamaño de muestra para el análisis. Las clases se calculan directamente desde btc_delta_24h."
)

if st.button("Generar Boxplot", key="boxplot_button"):
    with st.spinner("Generando boxplot..."):
        try:
            chart_boxplot = plot_variable_by_class_boxplot(
                variable=selected_var_boxplot,
                sample_size=sample_size_boxplot
            )
            st.altair_chart(chart_boxplot, use_container_width=True)
        except Exception as e:
            st.error(f"Error al generar boxplot: {e}")

# --- New BTC Price Comparison Plot ---
st.markdown("---")
st.markdown("## Comparación de Predicción BTC vs Realidad")

# Load model and preprocessor
try:
    booster, params = load_inference_model()
    with open("static/preprocessor.pkl", "rb") as f:
        preprocessor = cloudpickle.load(f)
    
    # Load tweets for selection
    tweets_df = pd.read_csv("static/tweets-processed.csv")
    tweets_df['date'] = pd.to_datetime(tweets_df['date'])
    
    # Create tweet selector
    st.markdown("### Seleccionar Tweet")
    
    # Create a searchable/filterable tweet selector
    tweet_options = []
    for idx, row in tweets_df.iterrows():
        tweet_display = f"{row['date']} - {row['text'][:100]}..."
        tweet_options.append((idx, tweet_display))
    
    # Use selectbox with formatted display
    selected_tweet_idx = st.selectbox(
        "Selecciona un tweet para analizar",
        options=range(len(tweets_df)),
        format_func=lambda x: f"{tweets_df.iloc[x]['date']} - {tweets_df.iloc[x]['text'][:100]}...",
        index=0 if len(tweets_df) > 0 else None
    )
    
    if selected_tweet_idx is not None:
        selected_tweet = tweets_df.iloc[selected_tweet_idx]
        tweet_date = str(selected_tweet['date'])
        tweet_text = selected_tweet['text']
        
        # Display selected tweet info
        st.info(f"**Tweet seleccionado:** {tweet_text[:200]}...")
        
        # Plot BTC price comparison
        chart_btc = plot_btc_price_comparison(
            tweet_date=tweet_date,
            tweet_text=tweet_text,
            booster=booster,
            preprocessor=preprocessor
        )
        st.altair_chart(chart_btc, use_container_width=True)
        
        # Show prediction details
        try:
            from lib import process_input, CLASS_RANGES
            import json
            
            input_data = {
                "id": str(selected_tweet.get('id', '1')),
                "text": tweet_text,
                "device": "web",
                "favorites": int(selected_tweet.get('favorites', 0)),
                "retweets": int(selected_tweet.get('retweets', 0)),
                "date": tweet_date,
                "btc_delta_24h": float(selected_tweet.get('btc_delta_24h', 0)),
                "btc_tweet_day": int(selected_tweet.get('btc_tweet_day', 0)),
                "btc_delta_48h": float(selected_tweet.get('btc_delta_48h', 0)),
                "btc_24h_after": float(selected_tweet.get('btc_24h_after', 0)),
                "btc_48h_after": float(selected_tweet.get('btc_48h_after', 0))
            }
            
            data = process_input(json.dumps(input_data), preprocessor)
            feature_names = preprocessor.named_steps["column_transformer"].get_feature_names_out().tolist()
            dmatrix = xgb.DMatrix(data, feature_names=feature_names)
            pred_class = int(booster.predict(dmatrix)[0])
            
            # Calculate actual class from btc_delta_24h
            btc_delta = float(selected_tweet.get('btc_delta_24h', 0))
            if btc_delta < -0.06:
                actual_class = 0
            elif -0.06 <= btc_delta <= 0:
                actual_class = 1
            elif 0 < btc_delta <= 0.002:
                actual_class = 2
            else:  # btc_delta > 0.002
                actual_class = 3
            
            # Display both predicted and actual classes
            col_pred, col_actual = st.columns(2)
            with col_pred:
                st.success(f"**Clase Predicha:** {pred_class} - {CLASS_RANGES.get(pred_class, 'Desconocido')}")
            with col_actual:
                st.info(f"**Clase Real:** {actual_class} - {CLASS_RANGES.get(actual_class, 'Desconocido')} (Δ: {btc_delta:.4f})")
            
            # Display color reference for the plot
            st.markdown("### Referencia de Colores en el Gráfico")
            col_red, col_green = st.columns(2)
            with col_red:
                st.markdown("🔴 **Rojo (línea punteada)**: Pendiente Predicha")
            with col_green:
                st.markdown("🟢 **Verde (línea punteada)**: Pendiente Real")
            
        except Exception as e:
            st.error(f"Error al obtener predicción: {e}")
            
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.info("Asegúrate de que los archivos del modelo estén disponibles en la carpeta static/")

