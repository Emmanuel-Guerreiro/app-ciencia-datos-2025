from google.cloud import language_v2
import xgboost as xgb
import json
import re
import streamlit as st
import pandas as pd
import altair as alt
from pydantic import BaseModel, Field, ValidationError
from typing import Any, Dict, List, Union
import joblib
import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

LANGUAGE_CODE = "en"

API_KEY = "AIzaSyAOD6UW_TpeD3sUHDo7ig2kuhw9O31m2Fo"

client = language_v2.LanguageServiceClient(
    client_options={"api_key": API_KEY}
)


def moderate_text(text_content) -> Dict[str, float]:
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
        st.altair_chart(chart, use_container_width=True)
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
        # Only drop columns that actually exist in the DataFrame
        columns_to_drop = [col for col in self.columns_to_drop if col in X.columns]
        if columns_to_drop:
            return X.drop(columns=columns_to_drop)
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
        df["tmp_date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df["hour"] = df["tmp_date"].dt.hour
        df["normalized_hour"] = (df["hour"] - 6) % 24
        df.drop(columns=["tmp_date"], inplace=True)
        return df

    def extract_day_of_week(self, df):
        date_parsed = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df["day_name"] = date_parsed.dt.day_name().astype("category")
        df["day_of_week"] = date_parsed.dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        return df

    def transform(self, df):
        print("I PASSED THROUGH THE TRANSFORM")
        df = df.copy()
        
        # Fill missing columns with default 0
        for col in ['btc_delta_24h', 'btc_delta_48h', 'btc_tweet_day']:
            if col not in df.columns:
                df[col] = 0
        
        # Safe division with zero handling
        btc_tweet_day_safe = df['btc_tweet_day'].replace(0, 1)  # Replace 0 with 1 to avoid division by zero
        df['btc_delta_24h_pct'] = df['btc_delta_24h'] / btc_tweet_day_safe
        df['btc_delta_48h_pct'] = df['btc_delta_48h'] / btc_tweet_day_safe
        
        # Set to 0 if original btc_tweet_day was 0
        df.loc[df['btc_tweet_day'] == 0, 'btc_delta_24h_pct'] = 0.0
        df.loc[df['btc_tweet_day'] == 0, 'btc_delta_48h_pct'] = 0.0

        df = self.extract_hour(df)
        df = self.extract_day_of_week(df)

        return df

def process_input(input_json: str, preprocessor):

    """Process user input JSON string: validate with Pydantic, then convert to DataFrame and apply preprocessor."""
    validated_data = _validate_and_parse_json(input_json)
    # Convert single dict to list format for DataFrame
    data_list = [validated_data] if isinstance(validated_data, dict) else validated_data
    X_new = pd.DataFrame(data_list)
    
    # Apply moderation to text field and add columns
    if 'text' in X_new.columns:
        moderation_results = X_new['text'].apply(moderate_text)
        
        # Get all unique keys from all moderation results
        all_keys = set()
        for result in moderation_results:
            all_keys.update(result.keys())
        
        # Create new columns for each moderation category
        for key in all_keys:
            X_new[key] = moderation_results.apply(lambda x: x.get(key, 0.0))
    
    # Add missing columns that might be expected by the preprocessor (will be dropped if needed)
    missing_cols = ['btc_24h_after', 'btc_48h_after']
    for col in missing_cols:
        if col not in X_new.columns:
            X_new[col] = 0
    # Apply same preprocessing as training
    X_prepared = preprocessor.transform(X_new)
    return X_prepared

CLASS_RANGES = {
    0: "btc_delta_24h < -0.06",
    1: "-0.06 ≤ btc_delta_24h ≤ 0",
    2: "0 < btc_delta_24h ≤ 0.002",
    3: "btc_delta_24h > 0.002",
}
