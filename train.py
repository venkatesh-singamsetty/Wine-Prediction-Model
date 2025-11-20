 import argparse
 from pathlib import Path
 import joblib
 import json
 from sklearn.linear_model import LinearRegression
 from sklearn.metrics import mean_squared_error, mean_absolute_error
 from utils import load_data, features_and_target
 
 def train(data_path, model_out="models/model.joblib", metrics_out="models/metrics.json"):
     df = load_data(data_path)
     X, y = features_and_target(df)
 
     # Fit a very simple model (Linear Regression) — easy to understand
     model = LinearRegression()
     model.fit(X, y)
 
     # Predictions on same data (simple demo, not a proper evaluation)
     preds = model.predict(X)
     mse = mean_squared_error(y, preds)
     mae = mean_absolute_error(y, preds)
 
     # Ensure output dir
     Path(model_out).parent.mkdir(parents=True, exist_ok=True)
     # Save model
     joblib.dump(model, model_out)
 
     metrics = {
         "n_rows": int(len(df)),
         "mse": float(mse),
         "mae": float(mae)
     }
     Path(metrics_out).parent.mkdir(parents=True, exist_ok=True)
     with open(metrics_out, "w") as f:
         json.dump(metrics, f, indent=2)
 
     print("Saved model to:", model_out)
     print("Saved metrics to:", metrics_out)
     print("Metrics:", metrics)
 
 if __name__ == "__main__":
     p = argparse.ArgumentParser(description="Train a tiny wine-quality model (for beginners).")
     p.add_argument("--data-path", type=str, default="wine_data_sample.csv", help="CSV with a 'quality' column")
     p.add_argument("--model-out", type=str, default="models/model.joblib")
     p.add_argument("--metrics-out", type=str, default="models/metrics.json")
     args = p.parse_args()
     train(args.data_path, args.model_out, args.metrics_out)
