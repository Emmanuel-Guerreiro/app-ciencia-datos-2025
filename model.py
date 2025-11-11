import streamlit as st
import cloudpickle
import xgboost as xgb
import pandas as pd
import json
from datetime import datetime
from lib import CLASS_RANGES, load_inference_model, process_input, render_feature_importance
from plotslib import plot_btc_price_comparison

st.set_page_config(page_title="Predicciones BTC", layout="wide")

# Load model and preprocessor first
try:
    booster, params = load_inference_model()
    with open("static/preprocessor.pkl", "rb") as f:
        preprocessor = cloudpickle.load(f)
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# Compact header
col_title, col_info = st.columns([3, 1])
with col_title:
    st.title("Predicciones de Bitcoin")
with col_info:
    with st.expander("Info"):
        st.caption("**XGBoost**")
        st.caption("4 clases")

# Create tabs for different input modes
tab1, tab2 = st.tabs(["Seleccionar Tweet Existente", "Agregar Nuevo Tweet"])

# Tab 1: Select existing tweet
with tab1:
    try:
        # Load tweets
        tweets_df = pd.read_csv("static/tweets-processed.csv")
        tweets_df['date'] = pd.to_datetime(tweets_df['date'])
        tweets_df = tweets_df.sort_values('date', ascending=False).reset_index(drop=True)
        
        # Compact search and selection in columns
        col_search, col_analyze = st.columns([3, 1])
        
        with col_search:
            search_term = st.text_input(
                "Buscar",
                placeholder="Palabras clave...",
                key="search_existing",
                label_visibility="collapsed"
            )
        
        # Filter tweets based on search
        filtered_df = tweets_df.copy()
        if search_term:
            filtered_df = tweets_df[
                tweets_df['text'].str.contains(search_term, case=False, na=False)
            ].reset_index(drop=True)
            st.caption(f"✓ {len(filtered_df)} tweets encontrados")
        
        if len(filtered_df) == 0:
            st.warning("Sin resultados")
        else:
            # Create a better display format for tweets
            def format_tweet_display(idx):
                row = filtered_df.iloc[idx]
                date_str = row['date'].strftime('%Y-%m-%d %H:%M')
                text_preview = row['text'][:60].replace('\n', ' ')
                return f"{date_str} | ❤️ {int(row.get('favorites', 0))} 🔁 {int(row.get('retweets', 0))} | {text_preview}..."
            
            # Select tweet
            selected_idx = st.selectbox(
                "Tweet:",
                options=range(len(filtered_df)),
                format_func=format_tweet_display,
                key="tweet_selector",
                label_visibility="collapsed"
            )
            
            if selected_idx is not None:
                selected_tweet = filtered_df.iloc[selected_idx]
                
                # Compact display with expander
                with st.expander("Ver detalles completos del tweet", expanded=False):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.caption("📅 **Fecha**")
                        st.text(selected_tweet['date'].strftime('%Y-%m-%d %H:%M'))
                    with col2:
                        st.caption("❤️ **Favoritos**")
                        st.text(int(selected_tweet.get('favorites', 0)))
                    with col3:
                        st.caption("🔁 **Retweets**")
                        st.text(int(selected_tweet.get('retweets', 0)))
                    with col4:
                        st.caption("📊 **Δ BTC 24h**")
                        st.text(f"{float(selected_tweet.get('btc_delta_24h', 0)):.4f}")
                    
                    st.text_area(
                        "Texto:",
                        value=selected_tweet['text'],
                        height=80,
                        disabled=True,
                        key="tweet_display",
                        label_visibility="collapsed"
                    )
                
                # Run prediction - Compact button
                if st.button("🚀 Analizar", key="analyze_existing", type="primary", use_container_width=True):
                    with st.spinner("Analizando..."):
                        try:
                            tweet_date = str(selected_tweet['date'])
                            tweet_text = selected_tweet['text']
                            
                            # Prepare input data
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
                            
                            # Process and predict
                            data = process_input(json.dumps(input_data), preprocessor)
                            feature_names = preprocessor.named_steps["column_transformer"].get_feature_names_out().tolist()
                            dmatrix = xgb.DMatrix(data, feature_names=feature_names)
                            pred_class = int(booster.predict(dmatrix)[0])
                            
                            # Calculate actual class
                            btc_delta = float(selected_tweet.get('btc_delta_24h', 0))
                            if btc_delta < -0.06:
                                actual_class = 0
                            elif -0.06 <= btc_delta <= 0:
                                actual_class = 1
                            elif 0 < btc_delta <= 0.002:
                                actual_class = 2
                            else:
                                actual_class = 3
                            
                            # Compact results display
                            st.markdown("---")
                            col1, col2, col3 = st.columns([1, 1, 1])
                            
                            with col1:
                                st.metric(
                                    "🔮 Predicción", 
                                    f"Clase {pred_class}",
                                    CLASS_RANGES.get(pred_class, 'Desconocido')
                                )
                            
                            with col2:
                                st.metric(
                                    "✅ Real", 
                                    f"Clase {actual_class}",
                                    f"Δ: {btc_delta:.4f}"
                                )
                            
                            with col3:
                                is_correct = pred_class == actual_class
                                st.metric(
                                    "Resultado",
                                    "✓ Correcto" if is_correct else "✗ Error",
                                    f"Δ {abs(pred_class - actual_class)} clases" if not is_correct else "Exacto"
                                )
                            
                            # Compact chart
                            chart_btc = plot_btc_price_comparison(
                                tweet_date=tweet_date,
                                tweet_text=tweet_text,
                                booster=booster,
                                preprocessor=preprocessor
                            )
                            st.altair_chart(chart_btc, width='stretch')
                            
                            # Compact legend
                            st.caption("🔵 Línea azul: momento del tweet | 🔴 Roja: predicción | 🟢 Verde: real")
                            
                        except Exception as e:
                            st.error(f"Error al analizar el tweet: {e}")
                            import traceback
                            st.code(traceback.format_exc())
    
    except Exception as e:
        st.error(f"Error al cargar tweets: {e}")

# Tab 2: Add new tweet
with tab2:
    st.caption("💡 Completa los campos para crear un tweet y predecir su impacto")
    
    with st.form("new_tweet_form"):
        # More compact layout with 4 columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            new_date = st.date_input("📅 Fecha", value=datetime.now())
        with col2:
            new_time = st.time_input("🕐 Hora", value=datetime.now().time())
        with col3:
            new_favorites = st.number_input("❤️ Favs", min_value=0, value=0, step=1)
        with col4:
            new_retweets = st.number_input("🔁 RTs", min_value=0, value=0, step=1)
        
        new_text = st.text_area(
            "📝 Texto",
            placeholder="Contenido del tweet...",
            height=100
        )
        
        submit_button = st.form_submit_button("🚀 Analizar", type="primary", use_container_width=True)
    
    if submit_button:
        if not new_text.strip():
            st.error("⚠️ Ingresa texto")
        else:
            with st.spinner("Analizando..."):
                try:
                    # Combine date and time
                    new_datetime = datetime.combine(new_date, new_time)
                    new_datetime_str = new_datetime.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Prepare input data (with defaults for BTC fields)
                    input_data = {
                        "id": "new_tweet",
                        "text": new_text,
                        "device": "web",
                        "favorites": int(new_favorites),
                        "retweets": int(new_retweets),
                        "date": new_datetime_str,
                        "btc_delta_24h": 0.0,
                        "btc_tweet_day": 0,
                        "btc_delta_48h": 0.0,
                        "btc_24h_after": 0.0,
                        "btc_48h_after": 0.0
                    }
                    
                    # Process and predict
                    data = process_input(json.dumps(input_data), preprocessor)
                    feature_names = preprocessor.named_steps["column_transformer"].get_feature_names_out().tolist()
                    dmatrix = xgb.DMatrix(data, feature_names=feature_names)
                    pred_class = int(booster.predict(dmatrix)[0])
                    
                    # Compact display
                    st.markdown("---")
                    
                    interpretation_icons = {
                        0: "📉 Caída Fuerte",
                        1: "📊 Caída Moderada",
                        2: "📈 Crecimiento Leve",
                        3: "🚀 Crecimiento Fuerte"
                    }
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.metric("🔮 Predicción", f"Clase {pred_class}")
                    with col2:
                        st.info(f"{interpretation_icons.get(pred_class, '')} - {CLASS_RANGES.get(pred_class, 'Desconocido')}")
                    
                    # Plot with expander for details
                    with st.expander("📖 Ver interpretación detallada", expanded=False):
                        interpretation = {
                            0: "El modelo predice una caída significativa del Bitcoin (más del 6% en 24h). Sentimiento muy negativo.",
                            1: "El modelo predice una caída moderada del Bitcoin (entre 0% y 6% en 24h). Sentimiento ligeramente negativo.",
                            2: "El modelo predice un crecimiento pequeño del Bitcoin (entre 0% y 0.2% en 24h). Sentimiento neutral a ligeramente positivo.",
                            3: "El modelo predice un crecimiento significativo del Bitcoin (más de 0.2% en 24h). Sentimiento muy positivo."
                        }
                        st.write(interpretation.get(pred_class, ""))
                    
                    # Plot comparison chart
                    try:
                        chart_btc = plot_btc_price_comparison(
                            tweet_date=new_datetime_str,
                            tweet_text=new_text,
                            booster=booster,
                            preprocessor=preprocessor
                        )
                        st.altair_chart(chart_btc, width='stretch')
                        st.caption("🔴 Línea roja: predicción del modelo para las próximas 24h")
                    except Exception as e:
                        st.warning(f"⚠️ No se pudo generar gráfico: {e}")
                    
                except Exception as e:
                    st.error(f"Error al procesar el nuevo tweet: {e}")
                    import traceback
                    st.code(traceback.format_exc())

# --- Compact Sidebar ---
with st.sidebar:
    st.markdown("### ℹ️ Modelo")
    st.caption("**XGBoost Classifier**")
    st.caption("4 clases de predicción")
    
    with st.expander("📚 Clases", expanded=False):
        for class_id, class_range in CLASS_RANGES.items():
            st.caption(f"**{class_id}:** {class_range}")
    
    with st.expander("⚙️ Hiperparámetros", expanded=False):
        st.json(params)
    
    with st.expander("📊 Importancia", expanded=False):
        render_feature_importance(booster)