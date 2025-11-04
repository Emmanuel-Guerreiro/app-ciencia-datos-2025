"""
Script to pre-compute predictions for all tweets.
Run this script to generate static/tweets-with-predictions.csv
which will speed up the boxplot visualization.
"""
import pandas as pd
from lib import load_inference_model
from plotslib import precompute_predictions
import cloudpickle

if __name__ == "__main__":
    print("Loading model and preprocessor...")
    booster, params = load_inference_model()
    
    with open("static/preprocessor.pkl", "rb") as f:
        preprocessor = cloudpickle.load(f)
    
    print("Starting pre-computation...")
    precompute_predictions(
        tweets_csv_path="static/tweets-processed.csv",
        output_path="static/tweets-with-predictions.csv",
        booster=booster,
        preprocessor=preprocessor,
        batch_size=100
    )
    
    print("Done! Predictions saved to static/tweets-with-predictions.csv")

