import streamlit as st
import cloudpickle
import xgboost as xgb
from lib import CLASS_RANGES, load_inference_model, process_input, render_feature_importance

st.set_page_config(page_title="Model Inspector", layout="centered")
st.title("🧠 Model Inference and Inspection")

booster, params = load_inference_model()
st.subheader("⚙️ Model Hyperparameters")
st.json(params)

render_feature_importance(booster)

# --- Inference ---
st.subheader("🔮 Try Inference")

json_input = st.text_area("Input JSON", value='{"feature1": 1.2, "feature2": 3.4}')

if st.button("Run Inference"):
    try:
        with open("static/preprocessor.pkl", "rb") as f:
            preprocessor = cloudpickle.load(f)

        data = process_input(json_input, preprocessor)

        # get feature names as a list, not numpy array
        feature_names = preprocessor.named_steps["column_transformer"].get_feature_names_out().tolist()

        # create DMatrix for booster inference
        dmatrix = xgb.DMatrix(data, feature_names=feature_names)

        preds = booster.predict(dmatrix)
        
        st.success(f"Prediction (class {int(preds[0])}): {CLASS_RANGES.get(int(preds[0]), "Unknown range")}")
    except Exception as e:
        st.error(f"Error during inference: {e}")