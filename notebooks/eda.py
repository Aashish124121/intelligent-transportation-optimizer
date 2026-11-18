import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/raw/hour.csv")

# Average demand by weather condition
weather_demand = df.groupby("weathersit")["cnt"].mean()

print("\nAverage Demand by Weather:")
print(weather_demand)

# Visualization
plt.figure(figsize=(10, 6))

sns.barplot(
    x=weather_demand.index,
    y=weather_demand.values
)

plt.title("Transportation Demand by Weather Condition")
plt.xlabel("Weather Condition")
plt.ylabel("Average Rentals")

plt.xticks(
    [0, 1, 2, 3],
    ["Clear", "Mist/Cloudy", "Light Rain/Snow", "Heavy Rain/Snow"]
)

plt.savefig("data/processed/weather_demand.png")
plt.show()