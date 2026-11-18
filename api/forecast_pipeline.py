
import joblib
import pandas as pd
import numpy as np

from optimization.transport_optimizer import optimize_transportation


def run_forecast_pipeline():

    print("\nGLOBAL TRANSPORTATION FORECASTING PIPELINE")
    print("==========================================")

    # 1. Load trained XGBoost model
    model = joblib.load("models/saved/xgboost_model.pkl")

    # 2. Load historical dataset
    df = pd.read_csv("data/raw/hour.csv")
    df["dteday"] = pd.to_datetime(df["dteday"])

    df = df.sort_values(["dteday", "hr"])

    last_row = df.iloc[-1]

    # 3. Generate next-hour input features
    next_time = last_row["dteday"] + pd.Timedelta(hours=1)

    hour = next_time.hour
    month = next_time.month
    weekday = next_time.weekday()

    season = (month % 12 // 3) + 1

    features = pd.DataFrame([{
        "season": season,
        "yr": min(1, next_time.year - 2011),
        "mnth": month,
        "hr": hour,
        "holiday": 0,
        "weekday": weekday,
        "workingday": int(weekday < 5),
        "weathersit": int(last_row["weathersit"]),
        "temp": float(last_row["temp"]),
        "atemp": float(last_row["atemp"]),
        "hum": float(last_row["hum"]),
        "windspeed": float(last_row["windspeed"]),
        "day": next_time.day,
        "dayofyear": next_time.day_of_year
    }])

    # 4. Predict total transportation demand
    predicted_demand = max(
        0,
        int(round(float(model.predict(features)[0])))
    )

    print("\nNext Hour:", next_time)
    print("Predicted Total Demand:", predicted_demand)

    # 5. Distribute demand across simulated zones
    zone_shares = {
        "Zone_A": 0.40,
        "Zone_B": 0.27,
        "Zone_C": 0.20,
        "Zone_D": 0.13
    }

    zone_demand = {
        zone: int(round(predicted_demand * share))
        for zone, share in zone_shares.items()
    }

    # Correct rounding so zone totals equal predicted demand
    difference = predicted_demand - sum(zone_demand.values())

    zone_demand["Zone_A"] += difference

    print("\nPredicted Zone Demand:")
    for zone, demand in zone_demand.items():
        print(f"{zone}: {demand}")

    # 6. Optimize vehicle allocation
    print("\nRunning Transportation Optimization...")

    results = optimize_transportation(
        zone_demand,
        total_fleet=25,
        vehicle_capacity=20
    )

    print("\nPIPELINE EXECUTION COMPLETED!")

    return {
        "predicted_demand": predicted_demand,
        "zone_demand": zone_demand,
        "optimization_results": results
    }


if __name__ == "__main__":
    run_forecast_pipeline()
