
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Actual evaluation results
results = {
    "Model": ["XGBoost", "LSTM"],
    "MAE": [43.81, 66.21],
    "RMSE": [67.56, 97.34],
    "R2 Score": [0.9061, 0.8051]
}

df = pd.DataFrame(results)

print("\nMODEL PERFORMANCE COMPARISON")
print("=============================")
print(df.to_string(index=False))

# Save comparison report
df.to_csv(
    "data/processed/model_comparison.csv",
    index=False
)

# Visualize metrics
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = ["MAE", "RMSE", "R2 Score"]

for ax, metric in zip(axes, metrics):

    bars = ax.bar(df["Model"], df[metric])

    ax.set_title(metric)
    ax.set_ylabel("Score")

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom"
        )

plt.suptitle("XGBoost vs LSTM Performance")
plt.tight_layout()

plt.savefig(
    "data/processed/model_comparison.png",
    dpi=300
)

plt.show()

print("\nComparison report and graph saved successfully!")
