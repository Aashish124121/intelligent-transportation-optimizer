
# Intelligent Transportation Demand Forecasting & Resource Optimization System

<p align="center">
  <strong>Machine Learning • Predictive Analytics • Operations Research • Backend Engineering</strong>
</p>

An end-to-end intelligent transportation planning platform that combines machine learning-based demand forecasting with constraint-based resource optimization to make smarter vehicle allocation decisions across multiple service zones.

The system compares **XGBoost and LSTM forecasting models**, exposes forecasting and optimization capabilities through a **FastAPI backend**, and provides an interactive **Streamlit dashboard** for scenario simulation and business decision-making.

Rather than simply predicting demand, the platform takes the next step: **translating predictions into resource allocation decisions under fleet constraints.**

### Project Highlights

- **Demand Forecasting:** Trained and evaluated XGBoost and LSTM models using historical hourly transportation demand data.
- **Model Evaluation:** Achieved an R² score of **0.9061 with XGBoost** on a chronological test split.
- **Resource Optimization:** Used Google OR-Tools CP-SAT to allocate a limited vehicle fleet across service zones.
- **Priority-Based Decisions:** Incorporated business priority weights and shortage penalties into the optimization objective.
- **Scenario Simulation:** Evaluated how changes in fleet size, vehicle capacity, demand, and business priorities affect allocation decisions.
- **REST API:** Built endpoints using FastAPI to connect forecasting, optimization, and scenario simulation.
- **Interactive Dashboard:** Developed a Streamlit interface for model comparison, demand visualization, scenario analysis, and cost breakdowns.
- **Automated Testing:** Implemented API and optimizer tests, with all 10 tests currently passing.

### Technology Stack


| Area                 | Technologies                         |
| -------------------- | ------------------------------------ |
| Programming          | Python                               |
| Machine Learning     | XGBoost, PyTorch, LSTM, Scikit-learn |
| Data Processing      | Pandas, NumPy                        |
| Optimization         | Google OR-Tools (CP-SAT)             |
| Backend              | FastAPI, REST APIs                   |
| Frontend / Dashboard | Streamlit, Plotly                    |
| Visualization        | Matplotlib, Seaborn                  |
| Testing              | Pytest, FastAPI TestClient           |
| Model Persistence    | Joblib, PyTorch state dictionaries   |

---

## System Architecture

The platform follows a modular architecture that separates demand forecasting, resource optimization, API services, and dashboard visualization.


flowchart TD
    A["Historical Bike Sharing Dataset"] --> B["Data Preprocessing"]
    B --> C["Feature Engineering"]
    C --> D["Model Training & Evaluation"]

    D --> E["XGBoost Model"]
    D --> F["LSTM Model"]

    E --> G["Next-Hour Demand Forecast"]
    F --> H["Comparative Model Evaluation"]

    G --> I["Illustrative Zone Demand Distribution"]
    I --> J["FastAPI Backend"]
    J --> K["OR-Tools CP-SAT Optimizer"]

    K --> L["Fleet Allocation Results"]
    L --> M["Cost & Performance Metrics"]

    M --> N["Streamlit Dashboard"]
    N --> O["Scenario Simulation"]
    N --> P["Visual Analytics"]


### Core Components

| Component | Responsibility |
|---|---|
| Data Pipeline | Loads historical data, handles preprocessing, and prepares model features. |
| Forecasting Engine | Trains and evaluates XGBoost and LSTM models to predict hourly demand. |
| Zone Distribution | Distributes the forecast across illustrative service zones. |
| Optimization Engine | Uses OR-Tools CP-SAT to allocate vehicles under fleet and capacity constraints. |
| FastAPI Backend | Exposes forecasting, allocation, and scenario simulation endpoints. |
| Streamlit Dashboard | Displays predictions, model comparisons, allocation decisions, and cost analytics. |

**Execution Flow:**

1. Historical transportation data is processed and transformed into model-ready features.
2. Machine learning models generate demand forecasts.
3. The forecast is distributed across illustrative service zones.
4. The optimization engine calculates vehicle allocation based on fleet availability, capacity, business priorities, and shortage penalties.
5. The API returns allocation results and performance metrics.
6. The dashboard visualizes the results and allows users to experiment with different scenarios.

**Model Note:** XGBoost is used by the current forecasting API pipeline. LSTM is trained and evaluated as a comparative deep learning model.
```

**4.** Press `Ctrl + S` to save.

<text color="secondary" size="sm">Tip: GitHub automatically renders Mermaid diagrams inside README files, so the architecture will appear as a visual flowchart when recruiters view your repository.</text>

Once you've added this section, tell me **Done**. We'll move on to the model performance and optimization methodology.

---

## Model Performance & Evaluation

Two machine learning approaches were implemented and evaluated using a chronological 80:20 train-test split.

### Forecasting Model Comparison

| Metric | XGBoost | LSTM |
|---|---:|---:|
| Mean Absolute Error (MAE) | **43.81** | 66.21 |
| Root Mean Squared Error (RMSE) | **67.56** | 97.34 |
| R² Score | **0.9061** | 0.8051 |

### Key Observations

- **XGBoost achieved the best overall performance**, with an R² score of 0.9061.
- The XGBoost model achieved lower MAE and RMSE than the LSTM model.
- LSTM was implemented to evaluate the effectiveness of sequence-based deep learning for hourly demand forecasting.
- Based on the evaluation results, XGBoost was selected for the current API forecasting pipeline.

### Model Evaluation Methodology

- Chronological train-test split to preserve temporal ordering.
- Feature engineering using date and time-related attributes.
- MinMax scaling for LSTM inputs and target values.
- Evaluation using MAE, RMSE, and R².
- Saved trained models and preprocessing artifacts for reuse.

---

## Resource Optimization Strategy

The project uses **Google OR-Tools CP-SAT** to solve a constrained vehicle allocation problem.

Instead of distributing vehicles equally, the optimizer considers demand, business priorities, fleet availability, and the cost of unmet demand.

### Optimization Inputs

| Parameter | Description |
|---|---|
| Predicted Demand | Estimated transportation demand in each zone. |
| Fleet Size | Total number of available vehicles. |
| Vehicle Capacity | Maximum demand served by one vehicle. |
| Business Priority | Relative importance assigned to each zone. |
| Deployment Cost | Cost of deploying one vehicle. |
| Shortage Penalty | Financial penalty for unmet demand. |

### Optimization Objective

The optimizer minimizes the combined deployment and weighted shortage costs:

\[
\min \sum_{i=1}^{n}
\left[
C_d V_i + (C_s W_i) U_i
\right]
\]

Subject to:

\[
V_i \times K + U_i \geq D_i
\]

\[
\sum_{i=1}^{n} V_i \leq F
\]

Where:

- \(V_i\) = Vehicles assigned to zone \(i\)
- \(U_i\) = Unmet demand in zone \(i\)
- \(D_i\) = Predicted demand in zone \(i\)
- \(K\) = Capacity per vehicle
- \(F\) = Total available fleet
- \(C_d\) = Deployment cost per vehicle
- \(C_s\) = Shortage penalty per unit
- \(W_i\) = Business priority weight of zone \(i\)

### Why This Matters

When fleet resources are limited, the optimizer balances deployment costs against the business impact of unmet demand. Higher-priority zones receive greater consideration through weighted shortage penalties.

This makes resource allocation a cost-aware optimization problem rather than a simple equal-distribution strategy.

---

---

## Getting Started

Follow these instructions to run the project locally.

### Prerequisites

Make sure you have installed:

- Python 3.10 or later
- Git
- Visual Studio Code (recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/transportation-optimizer.git
cd transportation-optimizer
```

### 2. Create a Virtual Environment

**Windows:**

```powershell
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI Backend

```bash
python -m uvicorn api.main:app --reload
```

Backend:

http://127.0.0.1:8000

Interactive API documentation:

http://127.0.0.1:8000/docs

### 5. Start the Streamlit Dashboard

Open a second terminal, activate the virtual environment, and run:

```bash
streamlit run app.py
```

The dashboard will be available at:

http://localhost:8501

### 6. Run Tests

```bash
python -m pytest -v
```

The test suite validates API endpoints, scenario simulation, input validation, and optimization behavior.

---

## Dataset

This project uses the **UCI Bike Sharing Dataset**, which contains historical bike rental records from the Capital Bikeshare system in Washington, D.C.

### Dataset Overview

| Attribute | Details |
|---|---|
| Source | UCI Machine Learning Repository |
| Location | Washington, D.C., USA |
| Data Period | 2011–2012 |
| Hourly Records | 17,379 |
| Target Variable | `cnt` (total rental count) |
| Granularity | Hourly |

### Features Used

- **Temporal:** Hour, weekday, month, year, season, and working-day indicators.
- **Weather:** Temperature, humidity, wind speed, and weather conditions.
- **Calendar:** Holiday and working-day indicators.

### Data Preparation

- Converted date fields into usable datetime features.
- Sorted observations chronologically.
- Removed irrelevant identifier columns.
- Excluded `casual` and `registered` to prevent target leakage because they directly contribute to the target variable `cnt`.
- Preserved chronological ordering during model evaluation.

Dataset source: [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)

---

## Assumptions & Limitations

This project is a transportation demand forecasting and resource allocation prototype. The following assumptions are important when interpreting its results:

1. **Demand Proxy:** Bike rental counts are used as a proxy for transportation demand. The dataset does not represent freight or delivery operations.
2. **Illustrative Zones:** Zone-level demand is distributed using predefined illustrative percentages because the original dataset does not contain the four service-zone identifiers used by the optimizer.
3. **Forecasting Scope:** The current API uses XGBoost for next-hour forecasting. The LSTM model is included for comparative evaluation.
4. **Forecast Assumptions:** The demonstration forecasting pipeline uses simplified assumptions for future calendar and weather features.
5. **Optimization Scope:** Vehicle allocation is based on predicted demand, fleet capacity, deployment costs, and weighted shortage penalties.

These limitations provide opportunities for future improvements, including real-world zone-level data, richer forecasting features, and deployment with live transportation datasets.

---

---

## Project Structure

```text
transportation-optimizer/
│
├── api/
│   ├── forecast_pipeline.py
│   └── main.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── xgboost_model.py
│   ├── lstm_model.py
│   ├── saved/
│   └── model_metrics.json
│
├── notebooks/
│   ├── verify_data.py
│   └── eda.py
│
├── optimization/
│   └── transport_optimizer.py
│
├── utils/
│   └── preprocessing.py
│
├── tests/
│   ├── test_optimizer.py
│   └── test_api.py
│
├── app.py
├── requirements.txt
└── README.md
```

### Engineering Practices

- Modular separation of data processing, model training, API services, and optimization logic.
- Chronological model evaluation to preserve temporal ordering.
- Input validation and meaningful API error responses.
- Configurable optimization parameters for scenario analysis.
- Automated testing of API endpoints and optimization behavior.
- Persistent storage of trained models and evaluation metrics.

---

## Future Improvements

- Integrate real-time transportation demand data through external APIs.
- Replace illustrative zone distribution with a trained zone-level forecasting model.
- Experiment with advanced forecasting architectures such as Transformers.
- Introduce dynamic vehicle routing and route optimization.
- Add Docker containerization and cloud deployment.
- Implement model monitoring, prediction drift detection, and automated retraining.
- Improve forecasting robustness using additional historical and contextual features.

## Key Engineering Concepts Demonstrated

- Supervised Machine Learning
- Time-Series Forecasting
- Deep Learning with LSTM
- Mathematical Optimization
- Constraint Programming
- REST API Development
- Cost-Aware Resource Allocation
- Model Evaluation and Persistence
- Automated Testing
- Interactive Data Visualization

## Author

**Ashish Patel**

B.Tech – Computer Science and Engineering

**Technologies:** Python | Machine Learning | XGBoost | PyTorch | FastAPI | OR-Tools | Streamlit

---

<p align="center">
  <strong>Built to demonstrate how machine learning predictions can drive practical optimization decisions.</strong>
</p>
