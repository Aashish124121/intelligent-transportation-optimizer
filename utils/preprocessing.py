import pandas as pd


def load_and_preprocess_data():

    # Load dataset
    df = pd.read_csv("data/raw/hour.csv")

    # Convert date column
    df["dteday"] = pd.to_datetime(df["dteday"])

    # Sort chronologically
    df = df.sort_values(["dteday", "hr"])

    # Extract additional date features
    df["day"] = df["dteday"].dt.day
    df["dayofyear"] = df["dteday"].dt.day_of_year

    # Remove unnecessary and leakage columns
    df = df.drop(
        columns=["instant", "dteday", "casual", "registered"]
    )

    # Separate features and target
    X = df.drop(columns=["cnt"])
    y = df["cnt"]

    return X, y


if __name__ == "__main__":

    X, y = load_and_preprocess_data()

    print("Feature Engineering Completed!")

    print("\nFeature Shape:", X.shape)
    print("Target Shape:", y.shape)

    print("\nFeatures:")
    print(X.columns.tolist())

    print("\nFirst 5 Records:")
    print(X.head())