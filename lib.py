from google.cloud import language_v2
import xgboost as xgb
import json
import re
import streamlit as st
import pandas as pd
import altair as alt
from pydantic import BaseModel, Field, ValidationError
from typing import Any, Dict, List, Union
import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Get API key and language code from Streamlit secrets
try:
    LANGUAGE_CODE = st.secrets["google_cloud_language"]["language_code"]
    API_KEY = st.secrets["google_cloud_language"]["api_key"]
except (KeyError, AttributeError):
    # Fallback for cases where secrets are not available (e.g., non-Streamlit contexts)
    # In production, ensure secrets are properly configured
    import os
    LANGUAGE_CODE = os.getenv("GOOGLE_LANGUAGE_CODE", "en")
    API_KEY = os.getenv("GOOGLE_API_KEY", "")
    if not API_KEY:
        raise ValueError(
            "API_KEY not found. Please configure it in .streamlit/secrets.toml or "
            "set GOOGLE_API_KEY environment variable."
        )

client = language_v2.LanguageServiceClient(
    client_options={"api_key": API_KEY}
)


def moderate_text(text_content) -> Dict[str, float]:
    """
    Moderate text using Google Cloud Language API.
    Returns default values (0.0) if API call fails.
    """
    try:
        document = {
            "content": text_content,
            "type_": "PLAIN_TEXT",
            "language_code": LANGUAGE_CODE,
        }
        response = client.moderate_text(
            request={"document": document}
        )

        # Create dictionary with category name and confidence pairs using comprehension
        return {cat.name: cat.confidence for cat in response.moderation_categories}
    except Exception as e:
        # Log the error and return default values
        print(f"Error en moderate_text: {e}")
        # Return default values for all expected categories
        default_categories = {
            "Toxic": 0.0,
            "Insult": 0.0,
            "Profanity": 0.0,
            "Derogatory": 0.0,
            "Sexual": 0.0,
            "Death, Harm & Tragedy": 0.0,
            "Violent": 0.0,
            "Firearms & Weapons": 0.0,
            "Public Safety": 0.0,
            "Health": 0.0,
            "Religion & Belief": 0.0,
            "Illicit Drugs": 0.0,
            "War & Conflict": 0.0,
            "Politics": 0.0,
            "Finance": 0.0,
            "Legal": 0.0
        }
        return default_categories
    

def load_inference_model():
    # --- Load booster ---
    booster = xgb.Booster()
    booster.load_model("static/app_model.json")

    # --- Load hyperparams ---
    with open("static/app_model_params.json") as f:
        params = json.load(f)

    return booster, params


def render_feature_importance(booster):
    # --- Feature importance ---
    importance = booster.get_score(importance_type="gain")

    if importance:
        imp_df = pd.DataFrame(
            {"feature": list(importance.keys()), "importance": list(importance.values())}
        ).sort_values("importance", ascending=False)

        chart = (
            alt.Chart(imp_df)
            .mark_bar()
            .encode(
                x=alt.X("importance:Q", title="Gain"),
                y=alt.Y("feature:N", sort="-x", title="Feature"),
                tooltip=["feature", "importance"]
            )
            .properties(width=600, height=400)
        )
        st.altair_chart(chart, width='stretch')
    else:
        st.info("No importance data available.")


class InputRecord(BaseModel):
    """Pydantic model for validating user input records."""
    id: str = Field(..., description="Record identifier")
    text: str = Field(..., description="Text content")
    device: str = Field(..., description="Device/platform name")
    favorites: int = Field(..., description="Number of favorites", ge=0)
    retweets: int = Field(..., description="Number of retweets", ge=0)
    date: str = Field(..., description="Date string in format like '8/2/2011 18:07'")
    btc_delta_24h: float = Field(default=0.0, description="BTC delta in the last 24 hours")
    btc_tweet_day: int = Field(default=0, description="Number of BTC tweets in the last 24 hours")
    btc_delta_48h: float = Field(default=0.0, description="BTC delta in the last 48 hours")
    btc_24h_after: float = Field(default=0.0, description="BTC value 24 hours after")
    btc_48h_after: float = Field(default=0.0, description="BTC value 48 hours after") 


def _validate_and_parse_json(json_str: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Parse JSON string and validate input structure with Pydantic."""
    parsed = json.loads(json_str)
    
    # Convert id to string if it exists (JSON may parse it as int)
    def convert_id_to_string(record):
        record = record.copy()  # Avoid mutating original
        if 'id' in record and not isinstance(record['id'], str):
            record['id'] = str(record['id'])
        return record
    
    # Handle both single dict and list of dicts
    # Pydantic will automatically apply default values for missing fields
    if isinstance(parsed, list):
        records = [InputRecord(**convert_id_to_string(record)).model_dump() for record in parsed]
        return records
    else:
        record = InputRecord(**convert_id_to_string(parsed)).model_dump()
        return record


class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        import sys
        print("=== DropColumns.transform() iniciado ===")
        sys.stdout.flush()
        print(f"Columnas a eliminar configuradas: {self.columns_to_drop}")
        sys.stdout.flush()
        
        # Only drop columns that actually exist in the DataFrame
        columns_to_drop = [col for col in self.columns_to_drop if col in X.columns]
        print(f"Columnas que realmente se eliminarán: {columns_to_drop}")
        sys.stdout.flush()
        
        if columns_to_drop:
            result = X.drop(columns=columns_to_drop)
            print(f"Columnas después de eliminar: {result.columns.tolist()}")
            print("=== DropColumns.transform() completado ===")
            sys.stdout.flush()
            return result
        
        print("No hay columnas para eliminar")
        print("=== DropColumns.transform() completado ===")
        sys.stdout.flush()
        return X


class NormalizeData(BaseEstimator, TransformerMixin):
    """Transformer that normalizes column names to snake_case."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    @staticmethod
    def to_snake_case(name):
        """Convert a string to snake_case."""
        s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
        s2 = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1)
        s3 = re.sub(r'[-\s]+', '_', s2)
        s4 = re.sub(r'_+', '_', s3)
        return s4.strip('_').lower()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [self.to_snake_case(col) for col in df.columns]
        return df

class ManualFeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
        

    def extract_hour(self, df):
        print("  -> Iniciando extract_hour")
        try:
            print(f"  -> Parseando fecha (sample: {df['date'].iloc[0] if len(df) > 0 else 'N/A'})")
            df["tmp_date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
            print("  -> Extrayendo hora...")
            df["hour"] = df["tmp_date"].dt.hour
            print("  -> Normalizando hora...")
            df["normalized_hour"] = (df["hour"] - 6) % 24
            print("  -> Eliminando columna temporal...")
            df.drop(columns=["tmp_date"], inplace=True)
            print("  -> extract_hour completado")
            return df
        except Exception as e:
            print(f"  -> ERROR en extract_hour: {e}")
            raise

    def extract_day_of_week(self, df):
        print("  -> Iniciando extract_day_of_week")
        try:
            print(f"  -> Parseando fecha (sample: {df['date'].iloc[0] if len(df) > 0 else 'N/A'})")
            date_parsed = pd.to_datetime(df["date"], utc=True, errors="coerce")
            print("  -> Obteniendo nombre del día...")
            df["day_name"] = date_parsed.dt.day_name().astype("category")
            print("  -> Obteniendo día de la semana numérico...")
            df["day_of_week"] = date_parsed.dt.dayofweek
            print("  -> Calculando is_weekend...")
            df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
            print("  -> extract_day_of_week completado")
            return df
        except Exception as e:
            print(f"  -> ERROR en extract_day_of_week: {e}")
            raise

    def transform(self, df):
        import sys
        print("=== ManualFeatureEngineering.transform() iniciado ===")
        sys.stdout.flush()
        df = df.copy()
        print(f"Columnas de entrada: {df.columns.tolist()}")
        sys.stdout.flush()
        
        # Fill missing columns with default 0
        print("Llenando columnas faltantes...")
        for col in ['btc_delta_24h', 'btc_delta_48h', 'btc_tweet_day']:
            if col not in df.columns:
                df[col] = 0
        
        # Safe division with zero handling
        print("Calculando porcentajes BTC...")
        btc_tweet_day_safe = df['btc_tweet_day'].replace(0, 1)  # Replace 0 with 1 to avoid division by zero
        df['btc_delta_24h_pct'] = df['btc_delta_24h'] / btc_tweet_day_safe
        df['btc_delta_48h_pct'] = df['btc_delta_48h'] / btc_tweet_day_safe
        
        # Set to 0 if original btc_tweet_day was 0
        df.loc[df['btc_tweet_day'] == 0, 'btc_delta_24h_pct'] = 0.0
        df.loc[df['btc_tweet_day'] == 0, 'btc_delta_48h_pct'] = 0.0
        print("Porcentajes BTC calculados.")

        print("Extrayendo hora...")
        df = self.extract_hour(df)
        print("Hora extraída.")
        
        print("Extrayendo día de la semana...")
        df = self.extract_day_of_week(df)
        print("Día de la semana extraído.")
        
        print(f"Columnas de salida: {df.columns.tolist()}")
        print("=== ManualFeatureEngineering.transform() completado ===")
        return df

def process_input(input_json: str, preprocessor):
    """Process user input JSON string: validate with Pydantic, then convert to DataFrame and apply preprocessor."""
    try:
        print("=== Iniciando process_input ===")
        validated_data = _validate_and_parse_json(input_json)
        # Convert single dict to list format for DataFrame
        data_list = [validated_data] if isinstance(validated_data, dict) else validated_data
        X_new = pd.DataFrame(data_list)
        print(f"DataFrame creado con columnas: {X_new.columns.tolist()}")
        
        # Apply moderation to text field and add columns
        if 'text' in X_new.columns:
            print(f"Moderando {len(X_new)} texto(s)...")
            moderation_results = X_new['text'].apply(moderate_text)
            
            # Get all unique keys from all moderation results
            all_keys = set()
            for result in moderation_results:
                all_keys.update(result.keys())
            
            print(f"Categorías de moderación encontradas: {all_keys}")
            
            # Create new columns for each moderation category
            for key in all_keys:
                X_new[key] = moderation_results.apply(lambda x: x.get(key, 0.0))
            print("Moderación completada.")
            print(f"Columnas después de moderación: {X_new.columns.tolist()}")
        
        # Add missing columns that might be expected by the preprocessor (will be dropped if needed)
        missing_cols = ['btc_24h_after', 'btc_48h_after']
        for col in missing_cols:
            if col not in X_new.columns:
                X_new[col] = 0
        
        print(f"Columnas finales antes de preprocesamiento: {X_new.columns.tolist()}")
        print(f"Shape del DataFrame: {X_new.shape}")
        print(f"Tipos de datos:\n{X_new.dtypes}")
        
        # Apply same preprocessing as training
        print("Aplicando preprocesamiento...")
        
        # Inspect pipeline steps
        print(f"Pipeline steps: {preprocessor.steps if hasattr(preprocessor, 'steps') else 'No steps attribute'}")
        
        # CRITICAL FIX: Replace transformers with fresh instances
        if hasattr(preprocessor, 'steps'):
            for i, (name, transformer) in enumerate(preprocessor.steps):
                if name == 'feature_engineer' or isinstance(transformer, ManualFeatureEngineering):
                    print(f"⚠️  Reemplazando {name} con una instancia fresca de ManualFeatureEngineering")
                    preprocessor.steps[i] = (name, ManualFeatureEngineering())
                elif name == 'drop_columns' or isinstance(transformer, DropColumns):
                    print(f"⚠️  Reemplazando {name} con una instancia fresca de DropColumns")
                    # Preserve the columns_to_drop configuration
                    columns_to_drop = transformer.columns_to_drop if hasattr(transformer, 'columns_to_drop') else []
                    preprocessor.steps[i] = (name, DropColumns(columns_to_drop=columns_to_drop))
        
        try:
            # Apply each step manually to see where it fails
            if hasattr(preprocessor, 'steps'):
                X_temp = X_new
                for i, (name, transformer) in enumerate(preprocessor.steps):
                    import sys
                    print(f"\n--- Aplicando paso {i+1}: {name} ({type(transformer).__name__}) ---")
                    print(f"Shape antes: {X_temp.shape if hasattr(X_temp, 'shape') else 'N/A'}")
                    sys.stdout.flush()
                    try:
                        print(f"Llamando a {name}.transform()...")
                        sys.stdout.flush()
                        X_temp = transformer.transform(X_temp)
                        print(f"✓ Paso {name} completado exitosamente")
                        print(f"Shape después: {X_temp.shape if hasattr(X_temp, 'shape') else 'N/A'}")
                        sys.stdout.flush()
                    except Exception as step_error:
                        print(f"✗ ERROR en paso {name}: {step_error}")
                        print(f"Tipo de error: {type(step_error).__name__}")
                        import traceback
                        traceback.print_exc()
                        raise
                X_prepared = X_temp
            else:
                # Fallback to normal transform
                X_prepared = preprocessor.transform(X_new)
            
            print(f"\n✓ Preprocesamiento completado. Shape resultante: {X_prepared.shape}")
            return X_prepared
        except Exception as prep_error:
            print(f"\n✗ ERROR GENERAL en preprocessor.transform(): {prep_error}")
            print(f"Tipo de error: {type(prep_error).__name__}")
            import traceback
            print("Traceback completo:")
            traceback.print_exc()
            raise
            
    except Exception as e:
        print(f"Error en process_input: {e}")
        print(f"Tipo de error: {type(e).__name__}")
        import traceback
        print("Traceback completo:")
        traceback.print_exc()
        raise

CLASS_RANGES = {
    0: "btc_delta_24h < -0.06",
    1: "-0.06 ≤ btc_delta_24h ≤ 0",
    2: "0 < btc_delta_24h ≤ 0.002",
    3: "btc_delta_24h > 0.002",
}
