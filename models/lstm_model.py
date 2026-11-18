
import os
import json
import joblib
import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from utils.preprocessing import load_and_preprocess_data


# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cpu")
SEQUENCE_LENGTH = 24
EPOCHS = 15
BATCH_SIZE = 128


# LSTM architecture
class DemandLSTM(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])


def create_sequences(X, y, sequence_length):

    sequences = []
    targets = []

    for i in range(sequence_length, len(X)):
        sequences.append(X[i-sequence_length:i])
        targets.append(y[i])

    return np.array(sequences), np.array(targets)


def train_model():

    # Load chronological dataset
    X, y = load_and_preprocess_data()

    split_index = int(len(X) * 0.8)

    # Fit scalers only on training data
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_train_raw = x_scaler.fit_transform(
        X.iloc[:split_index]
    )

    X_test_raw = x_scaler.transform(
        X.iloc[split_index:]
    )

    y_train_raw = y_scaler.fit_transform(
        y.iloc[:split_index].to_numpy().reshape(-1, 1)
    ).flatten()

    y_test_raw = y_scaler.transform(
        y.iloc[split_index:].to_numpy().reshape(-1, 1)
    ).flatten()

    # Keep final training observations as test context
    X_test_context = np.concatenate(
        [X_train_raw[-SEQUENCE_LENGTH:], X_test_raw]
    )

    y_test_context = np.concatenate(
        [y_train_raw[-SEQUENCE_LENGTH:], y_test_raw]
    )

    # Generate sequences
    X_train_seq, y_train_seq = create_sequences(
        X_train_raw,
        y_train_raw,
        SEQUENCE_LENGTH
    )

    X_test_seq, y_test_seq = create_sequences(
        X_test_context,
        y_test_context,
        SEQUENCE_LENGTH
    )

    # Convert to tensors
    X_train_tensor = torch.tensor(
        X_train_seq,
        dtype=torch.float32
    ).to(DEVICE)

    y_train_tensor = torch.tensor(
        y_train_seq,
        dtype=torch.float32
    ).unsqueeze(1).to(DEVICE)

    X_test_tensor = torch.tensor(
        X_test_seq,
        dtype=torch.float32
    ).to(DEVICE)

    # DataLoader
    dataset = torch.utils.data.TensorDataset(
        X_train_tensor,
        y_train_tensor
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Initialize model
    model = DemandLSTM(
        input_size=X_train_seq.shape[2]
    ).to(DEVICE)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    print("Training LSTM model...")

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0

        for batch_X, batch_y in loader:

            optimizer.zero_grad()

            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{EPOCHS}, "
            f"Loss: {total_loss / len(loader):.5f}"
        )

    # Evaluation
    model.eval()

    with torch.no_grad():

        predictions_scaled = model(
            X_test_tensor
        ).cpu().numpy()

    predictions = y_scaler.inverse_transform(
        predictions_scaled
    ).flatten()

    actual = y_scaler.inverse_transform(
        y_test_seq.reshape(-1, 1)
    ).flatten()

    mae = mean_absolute_error(actual, predictions)

    rmse = np.sqrt(
        mean_squared_error(actual, predictions)
    )

    r2 = r2_score(actual, predictions)

    print("\nLSTM Model Performance")
    print("----------------------")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))

    # Save model artifacts
    os.makedirs("models/saved", exist_ok=True)

    torch.save(
        model.state_dict(),
        "models/saved/lstm_model.pth"
    )

    joblib.dump(
        x_scaler,
        "models/saved/x_scaler.pkl"
    )

    joblib.dump(
        y_scaler,
        "models/saved/y_scaler.pkl"
    )

    print("\nLSTM model and scalers saved successfully!")

    # Prepare evaluation metrics
    metrics = {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4)
    }

    # Preserve existing model metrics
    metrics_path = "models/model_metrics.json"

    if os.path.exists(metrics_path):

        with open(metrics_path, "r") as file:
            all_metrics = json.load(file)

    else:
        all_metrics = {}

    # Update only LSTM metrics
    all_metrics["lstm"] = metrics

    # Save updated metrics
    with open(metrics_path, "w") as file:

        json.dump(
            all_metrics,
            file,
            indent=4
        )

    print("\nLSTM metrics saved successfully!")
    print("Metrics:", metrics)


if __name__ == "__main__":
    train_model()
