
import os
import json
import joblib

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from utils.preprocessing import load_and_preprocess_data


def train_model():

    # Load processed features and target
    X, y = load_and_preprocess_data()

    # Chronological train-test split (80:20)
    split_index = int(len(X) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print("Training Records:", len(X_train))
    print("Testing Records:", len(X_test))

    # Initialize XGBoost
    model = XGBRegressor(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    # Train model
    print("\nTraining XGBoost model...")
    model.fit(X_train, y_train)

    # Generate predictions
    predictions = model.predict(X_test)

    # Evaluate model
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print("\nXGBoost Model Performance")
    print("-------------------------")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))

    # Create saved models directory
    os.makedirs("models/saved", exist_ok=True)

    # Save trained model
    joblib.dump(
        model,
        "models/saved/xgboost_model.pkl"
    )

    print("\nXGBoost model saved successfully!")

    # Prepare evaluation metrics
    metrics = {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4)
    }

    # Load existing metrics to preserve other models
    metrics_path = "models/model_metrics.json"

    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as file:
            all_metrics = json.load(file)
    else:
        all_metrics = {}

    # Update only XGBoost metrics
    all_metrics["xgboost"] = metrics

    # Save updated metrics
    with open(metrics_path, "w") as file:
        json.dump(all_metrics, file, indent=4)

    print("\nXGBoost metrics saved successfully!")
    print("Metrics:", metrics)


if __name__ == "__main__":
    train_model()
