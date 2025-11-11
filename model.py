import streamlit as st
import cloudpickle
import xgboost as xgb
import pandas as pd
import json
from datetime import datetime
from lib import CLASS_RANGES, load_inference_model, process_input, render_feature_importance
from plotslib import plot_btc_price_comparison

st.set_page_config(page_title="Predicciones BTC", layout="wide")
st.title("🔮 Predicciones de Bitcoin")

# Load model and preprocessor
try:
    booster, params = load_inference_model()
    with open("static/preprocessor.pkl", "rb") as f:
        preprocessor = cloudpickle.load(f)
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# --- Main Content ---
st.markdown("## Analizar Tweet y Predecir Comportamiento de BTC")

# Create tabs for different input modes
tab1, tab2 = st.tabs(["📋 Seleccionar Tweet Existente", "✍️ Agregar Nuevo Tweet"])

# Tab 1: Select existing tweet
with tab1:
    st.markdown("### Seleccionar Tweet del Dataset")
    
    try:
        # Load tweets
        tweets_df = pd.read_csv("static/tweets-processed.csv")
        tweets_df['date'] = pd.to_datetime(tweets_df['date'])
        tweets_df = tweets_df.sort_values('date', ascending=False).reset_index(drop=True)
        
        # Create search functionality
        search_term = st.text_input(
            "🔍 Buscar tweet (por texto)",
            placeholder="Escribe palabras clave para filtrar tweets...",
            key="search_existing"
        )
        
        # Filter tweets based on search
        filtered_df = tweets_df.copy()
        if search_term:
            filtered_df = tweets_df[
                tweets_df['text'].str.contains(search_term, case=False, na=False)
            ].reset_index(drop=True)
            st.info(f"Se encontraron {len(filtered_df)} tweets que coinciden con la búsqueda")
        
        if len(filtered_df) == 0:
            st.warning("No se encontraron tweets. Intenta con otros términos de búsqueda.")
        else:
            # Create a better display format for tweets
            def format_tweet_display(idx):
                row = filtered_df.iloc[idx]
                date_str = row['date'].strftime('%Y-%m-%d %H:%M')
                text_preview = row['text'][:80].replace('\n', ' ')
                return f"{date_str} | ❤️ {int(row.get('favorites', 0))} 🔁 {int(row.get('retweets', 0))} | {text_preview}..."
            
            # Select tweet
            selected_idx = st.selectbox(
                "Selecciona un tweet",
                options=range(len(filtered_df)),
                format_func=format_tweet_display,
                key="tweet_selector"
            )
            
            if selected_idx is not None:
                selected_tweet = filtered_df.iloc[selected_idx]
                
                # Display tweet details
                st.markdown("---")
                st.markdown("### 📝 Detalles del Tweet Seleccionado")
                
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.markdown(f"**Fecha:** {selected_tweet['date'].strftime('%Y-%m-%d %H:%M:%S')}")
                with col2:
                    st.metric("❤️ Favoritos", int(selected_tweet.get('favorites', 0)))
                with col3:
                    st.metric("🔁 Retweets", int(selected_tweet.get('retweets', 0)))
                
                st.text_area(
                    "Contenido del Tweet:",
                    value=selected_tweet['text'],
                    height=100,
                    disabled=True,
                    key="tweet_display"
                )
                
                # Run prediction
                if st.button("🚀 Analizar Tweet", key="analyze_existing", type="primary"):
                    with st.spinner("Analizando tweet y generando predicción..."):
                        try:
                            tweet_date = str(selected_tweet['date'])
                            tweet_text = selected_tweet['text']
                            
                            st.info(f"📝 Procesando tweet: {tweet_text[:100]}...")
                            
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
                            with st.spinner("🔍 Analizando contenido del tweet con IA..."):
                                data = process_input(json.dumps(input_data), preprocessor)
                            
                            st.success("✅ Análisis de contenido completado")
                            
                            with st.spinner("🎯 Generando predicción..."):
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
                            
                            # Display results
                            st.markdown("---")
                            st.markdown("### 📊 Resultados de la Predicción")
                            
                            col_pred, col_actual = st.columns(2)
                            with col_pred:
                                st.success(f"**🔮 Clase Predicha:** {pred_class}")
                                st.info(CLASS_RANGES.get(pred_class, 'Desconocido'))
                            with col_actual:
                                st.success(f"**✅ Clase Real:** {actual_class}")
                                st.info(f"{CLASS_RANGES.get(actual_class, 'Desconocido')}\n\nΔ Real: {btc_delta:.4f}")
                            
                            # Accuracy indicator
                            if pred_class == actual_class:
                                st.success("✅ ¡Predicción correcta!")
                            else:
                                st.warning(f"⚠️ Predicción incorrecta. Error: {abs(pred_class - actual_class)} clases de diferencia")
                            
                            # Plot comparison chart
                            st.markdown("### 📈 Comparación Visual: Predicción vs Realidad")
                            chart_btc = plot_btc_price_comparison(
                                tweet_date=tweet_date,
                                tweet_text=tweet_text,
                                booster=booster,
                                preprocessor=preprocessor
                            )
                            st.altair_chart(chart_btc, width='stretch')
                            
                            # Legend
                            st.markdown("#### Referencia de Colores")
                            col_legend1, col_legend2 = st.columns(2)
                            with col_legend1:
                                st.markdown("🔴 **Línea Roja (punteada)**: Pendiente Predicha por el Modelo")
                            with col_legend2:
                                st.markdown("🟢 **Línea Verde (punteada)**: Pendiente Real del Bitcoin")
                            
                        except Exception as e:
                            st.error(f"Error al analizar el tweet: {e}")
                            import traceback
                            st.code(traceback.format_exc())
    
    except Exception as e:
        st.error(f"Error al cargar tweets: {e}")

# Tab 2: Add new tweet
with tab2:
    st.markdown("### Crear Nuevo Tweet para Análisis")
    st.info("💡 Completa los campos para simular un tweet nuevo y predecir su impacto en Bitcoin")
    
    with st.form("new_tweet_form"):
        col_form1, col_form2 = st.columns(2)
        
        with col_form1:
            new_date = st.date_input(
                "📅 Fecha del Tweet",
                value=datetime.now(),
                help="Selecciona la fecha del tweet"
            )
            new_time = st.time_input(
                "🕐 Hora del Tweet",
                value=datetime.now().time(),
                help="Selecciona la hora del tweet"
            )
        
        with col_form2:
            new_favorites = st.number_input(
                "❤️ Favoritos",
                min_value=0,
                value=0,
                step=1,
                help="Número de favoritos/likes del tweet"
            )
            new_retweets = st.number_input(
                "🔁 Retweets",
                min_value=0,
                value=0,
                step=1,
                help="Número de retweets"
            )
        
        new_text = st.text_area(
            "📝 Texto del Tweet",
            placeholder="Escribe el contenido del tweet aquí...",
            height=150,
            help="Contenido del tweet a analizar"
        )
        
        submit_button = st.form_submit_button("🚀 Analizar Nuevo Tweet", type="primary")
    
    if submit_button:
        if not new_text.strip():
            st.error("⚠️ Por favor ingresa el texto del tweet")
        else:
            with st.spinner("Analizando nuevo tweet..."):
                try:
                    # Combine date and time
                    new_datetime = datetime.combine(new_date, new_time)
                    new_datetime_str = new_datetime.strftime('%Y-%m-%d %H:%M:%S')
                    
                    st.info(f"📝 Procesando nuevo tweet...")
                    
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
                    with st.spinner("🔍 Analizando contenido del tweet con IA..."):
                        data = process_input(json.dumps(input_data), preprocessor)
                    
                    st.success("✅ Análisis de contenido completado")
                    
                    with st.spinner("🎯 Generando predicción..."):
                        feature_names = preprocessor.named_steps["column_transformer"].get_feature_names_out().tolist()
                        dmatrix = xgb.DMatrix(data, feature_names=feature_names)
                        pred_class = int(booster.predict(dmatrix)[0])
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("### 📊 Resultado de la Predicción")
                    
                    st.success(f"**🔮 Clase Predicha:** {pred_class}")
                    st.info(f"**Interpretación:** {CLASS_RANGES.get(pred_class, 'Desconocido')}")
                    
                    # Interpretation guide
                    st.markdown("### 📖 ¿Qué significa esta predicción?")
                    
                    interpretation = {
                        0: ("📉 **Caída Fuerte**", "El modelo predice una caída significativa del Bitcoin (más del 6% en 24h). Este tweet tiene un sentimiento muy negativo."),
                        1: ("📊 **Caída Moderada**", "El modelo predice una caída moderada del Bitcoin (entre 0% y 6% en 24h). Sentimiento ligeramente negativo."),
                        2: ("📈 **Crecimiento Leve**", "El modelo predice un crecimiento pequeño del Bitcoin (entre 0% y 0.2% en 24h). Sentimiento neutral a ligeramente positivo."),
                        3: ("🚀 **Crecimiento Fuerte**", "El modelo predice un crecimiento significativo del Bitcoin (más de 0.2% en 24h). Este tweet tiene un sentimiento muy positivo.")
                    }
                    
                    if pred_class in interpretation:
                        title, description = interpretation[pred_class]
                        st.markdown(f"#### {title}")
                        st.write(description)
                    
                    # Plot comparison chart (without actual data for new tweets)
                    try:
                        st.markdown("### 📈 Visualización de Predicción en Contexto Histórico")
                        chart_btc = plot_btc_price_comparison(
                            tweet_date=new_datetime_str,
                            tweet_text=new_text,
                            booster=booster,
                            preprocessor=preprocessor
                        )
                        st.altair_chart(chart_btc, width='stretch')
                        st.caption("🔴 La línea roja punteada muestra la predicción del modelo para las próximas 24 horas")
                    except Exception as e:
                        st.warning(f"No se pudo generar el gráfico histórico: {e}")
                    
                except Exception as e:
                    st.error(f"Error al procesar el nuevo tweet: {e}")
                    import traceback
                    st.code(traceback.format_exc())

# --- Sidebar: Model Info ---
with st.sidebar:
    st.markdown("## ℹ️ Información del Modelo")
    
    with st.expander("⚙️ Hiperparámetros del Modelo"):
        st.json(params)
    
    with st.expander("📊 Feature Importance"):
        render_feature_importance(booster)
    
    st.markdown("---")
    st.markdown("### 📚 Guía de Clases")
    for class_id, class_range in CLASS_RANGES.items():
        st.markdown(f"**Clase {class_id}:** `{class_range}`")