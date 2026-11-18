
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import json
import os


API_URL = "http://127.0.0.1:8000"
METRICS_PATH = "models/model_metrics.json"


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Transportation Optimizer",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Transportation Forecasting & Optimization")

st.caption(
    "Machine Learning-Powered Demand Forecasting "
    "and Intelligent Vehicle Allocation"
)

st.divider()


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def load_model_metrics():

    if not os.path.exists(METRICS_PATH):
        return {}

    try:
        with open(METRICS_PATH, "r") as file:
            return json.load(file)

    except (json.JSONDecodeError, OSError):
        return {}


def build_allocation_dataframe(optimization):

    rows = []

    for zone, values in optimization.items():

        rows.append({
            "Zone": zone,
            "Predicted Demand": values.get("predicted_demand", 0),
            "Business Priority": values.get("business_priority", 1.0),
            "Vehicles Assigned": values.get("vehicles_assigned", 0),
            "Capacity": values.get("capacity", 0),
            "Served Demand": values.get(
                "served_demand",
                values.get("predicted_demand", 0)
                - values.get("unmet_demand", 0)
            ),
            "Unmet Demand": values.get("unmet_demand", 0),
            "Fleet Utilization (%)": values.get(
                "fleet_utilization", 0
            ),
            "Shortage Cost": values.get("shortage_cost", 0),
            "Explanation": values.get(
                "explanation",
                "No explanation provided by the optimizer."
            )
        })

    return pd.DataFrame(rows)


def display_allocation_explanations(optimization, title):

    st.subheader(title)

    for zone, values in optimization.items():

        priority = values.get("business_priority", 1.0)
        vehicles = values.get("vehicles_assigned", 0)
        demand = values.get("predicted_demand", 0)
        served = values.get(
            "served_demand",
            demand - values.get("unmet_demand", 0)
        )
        unmet = values.get("unmet_demand", 0)
        shortage_cost = values.get("shortage_cost", 0)

        explanation = values.get(
            "explanation",
            "No explanation provided by the optimizer."
        )

        with st.expander(
            f"📍 {zone} — {vehicles} vehicle(s) assigned"
        ):

            col1, col2, col3 = st.columns(3)

            col1.metric("Business Priority", priority)
            col2.metric("Demand Served", served)
            col3.metric("Unmet Demand", unmet)

            st.write("**Optimizer's Decision:**")
            st.info(explanation)

            st.write(
                f"**Priority-weighted shortage cost:** "
                f"{shortage_cost:,}"
            )

            if vehicles > 0 and unmet == 0:
                st.success(
                    "Demand fully covered for this zone."
                )

            elif vehicles > 0 and unmet > 0:
                st.warning(
                    "Partial demand coverage due to limited capacity."
                )

            else:
                st.warning(
                    "No vehicle was allocated to this zone."
                )


def calculate_overall_utilization(df):

    total_demand = df["Predicted Demand"].sum()
    total_unmet = df["Unmet Demand"].sum()
    total_capacity = df["Capacity"].sum()

    return (
        ((total_demand - total_unmet) / total_capacity) * 100
        if total_capacity > 0 else 0
    )


# --------------------------------------------------
# BACKEND CONNECTION
# --------------------------------------------------

try:

    response = requests.get(
        f"{API_URL}/health",
        timeout=5
    )

    if response.status_code == 200:
        st.success("Backend API Connected")

    else:
        st.error("Backend API Error")

except requests.exceptions.RequestException:

    st.error(
        "Cannot connect to FastAPI. Start your backend first."
    )

    st.stop()


# --------------------------------------------------
# MODEL PERFORMANCE COMPARISON
# --------------------------------------------------

st.header("🤖 Machine Learning Model Performance")

metrics = load_model_metrics()

if metrics:

    model_data = []

    for model_name, values in metrics.items():

        model_data.append({
            "Model": model_name.upper(),
            "MAE": values.get("mae"),
            "RMSE": values.get("rmse"),
            "R² Score": values.get("r2")
        })

    metrics_df = pd.DataFrame(model_data)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Model Evaluation Metrics")

        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True
        )

    with col2:

        st.subheader("Prediction Error Comparison")

        error_data = metrics_df.melt(
            id_vars="Model",
            value_vars=["MAE", "RMSE"],
            var_name="Metric",
            value_name="Error"
        )

        fig = px.bar(
            error_data,
            x="Model",
            y="Error",
            color="Metric",
            barmode="group",
            text_auto=".2f",
            title="MAE vs RMSE"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    valid_r2 = metrics_df.dropna(
        subset=["R² Score"]
    )

    if not valid_r2.empty:

        st.subheader("Model R² Score Comparison")

        fig_r2 = px.bar(
            valid_r2,
            x="Model",
            y="R² Score",
            color="Model",
            text_auto=".4f",
            title="Coefficient of Determination"
        )

        fig_r2.update_yaxes(range=[0, 1])

        st.plotly_chart(
            fig_r2,
            use_container_width=True
        )

else:

    st.warning(
        "Model evaluation metrics are unavailable."
    )


st.divider()


# --------------------------------------------------
# DEMAND FORECASTING
# --------------------------------------------------

st.header("📈 Demand Forecasting")

if st.button(
    "Run Forecast & Optimization",
    type="primary"
):

    with st.spinner(
        "Running ML forecasting and optimization..."
    ):

        try:

            response = requests.post(
                f"{API_URL}/forecast-allocate",
                timeout=120
            )

            response.raise_for_status()

            st.session_state["results"] = response.json()

            st.success(
                "Forecast completed successfully!"
            )

        except requests.exceptions.RequestException as e:

            st.error(f"Error: {e}")


# --------------------------------------------------
# DISPLAY FORECAST RESULTS
# --------------------------------------------------

if "results" in st.session_state:

    result = st.session_state["results"]

    optimization = result.get(
        "optimization_results",
        {}
    )

    if optimization:

        df = build_allocation_dataframe(optimization)

        st.divider()

        st.header("📊 Transportation Analytics")

        # KPI metrics

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Demand",
            int(df["Predicted Demand"].sum())
        )

        col2.metric(
            "Vehicles Deployed",
            int(df["Vehicles Assigned"].sum())
        )

        col3.metric(
            "Total Capacity",
            int(df["Capacity"].sum())
        )

        col4.metric(
            "Unmet Demand",
            int(df["Unmet Demand"].sum())
        )

        overall_utilization = calculate_overall_utilization(df)

        st.metric(
            "Overall Fleet Utilization",
            f"{overall_utilization:.2f}%"
        )

        st.divider()

        # Zone allocation table

        st.subheader("Zone-wise Allocation")

        st.dataframe(
            df.drop(columns=["Explanation"]),
            use_container_width=True,
            hide_index=True
        )

        # Baseline optimization explanations

        display_allocation_explanations(
            optimization,
            "🧠 Baseline Optimization Explanations"
        )

        # Demand chart

        st.subheader("Predicted Transportation Demand")

        fig1 = px.bar(
            df,
            x="Zone",
            y="Predicted Demand",
            color="Zone",
            text_auto=True
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

        # Vehicle allocation chart

        st.subheader("Vehicle Allocation")

        fig2 = px.bar(
            df,
            x="Zone",
            y="Vehicles Assigned",
            color="Zone",
            text_auto=True
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        # Demand vs capacity chart

        st.subheader("Demand vs Capacity")

        chart_data = df.melt(
            id_vars="Zone",
            value_vars=[
                "Predicted Demand",
                "Capacity"
            ],
            var_name="Metric",
            value_name="Value"
        )

        fig3 = px.bar(
            chart_data,
            x="Zone",
            y="Value",
            color="Metric",
            barmode="group"
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

        # Fleet utilization chart

        st.subheader("Fleet Utilization by Zone")

        fig4 = px.bar(
            df,
            x="Zone",
            y="Fleet Utilization (%)",
            color="Zone",
            text_auto=".2f",
            title="Percentage of Allocated Capacity Used"
        )

        fig4.update_yaxes(
            range=[0, 100],
            ticksuffix="%"
        )

        st.plotly_chart(
            fig4,
            use_container_width=True
        )

        # Download report

        st.download_button(
            "📥 Download Allocation Report",
            df.drop(columns=["Explanation"]).to_csv(index=False),
            file_name="transportation_report.csv",
            mime="text/csv"
        )

    else:

        st.warning(
            "No optimization results received."
        )

        st.json(result)

else:

    st.info(
        "Click the button above to generate transportation forecasts."
    )


# --------------------------------------------------
# INTERACTIVE SCENARIO SIMULATOR
# --------------------------------------------------

st.divider()

st.header("🎛️ Interactive Scenario Simulator")

st.write(
    "Experiment with different fleet sizes, vehicle capacities, "
    "demand conditions and business priorities. The backend will "
    "recalculate the allocation using the OR-Tools optimization engine."
)

with st.container(border=True):

    st.subheader("Scenario Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:

        scenario_fleet = st.slider(
            "Available Vehicles",
            min_value=1,
            max_value=100,
            value=25,
            step=1,
            key="scenario_fleet"
        )

    with col2:

        scenario_capacity = st.slider(
            "Vehicle Capacity",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            key="scenario_capacity"
        )

    with col3:

        scenario_multiplier = st.slider(
            "Demand Multiplier",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            key="scenario_multiplier"
        )

    st.caption(
        f"Scenario demand adjustment: "
        f"{scenario_multiplier:.1f}x "
        f"({(scenario_multiplier - 1) * 100:+.0f}%)"
    )

    st.divider()

    # Business priority configuration

    st.subheader("🎯 Business Priority Configuration")

    st.write(
        "Adjust the importance of each zone. Higher priority "
        "increases the penalty for unmet demand in that zone."
    )

    p1, p2, p3, p4 = st.columns(4)

    with p1:

        priority_a = st.slider(
            "Zone A Priority",
            min_value=1.0,
            max_value=10.0,
            value=3.0,
            step=0.5,
            key="priority_a"
        )

    with p2:

        priority_b = st.slider(
            "Zone B Priority",
            min_value=1.0,
            max_value=10.0,
            value=2.0,
            step=0.5,
            key="priority_b"
        )

    with p3:

        priority_c = st.slider(
            "Zone C Priority",
            min_value=1.0,
            max_value=10.0,
            value=1.5,
            step=0.5,
            key="priority_c"
        )

    with p4:

        priority_d = st.slider(
            "Zone D Priority",
            min_value=1.0,
            max_value=10.0,
            value=1.0,
            step=0.5,
            key="priority_d"
        )

    st.caption(
        "Priority weights are illustrative business inputs, "
        "not values learned by the machine learning model."
    )

    st.divider()

    # Optimization cost configuration

    st.subheader("💰 Optimization Cost Configuration")

    cost_col1, cost_col2 = st.columns(2)

    with cost_col1:

        scenario_deployment_cost = st.number_input(
            "Vehicle Deployment Cost",
            min_value=1,
            max_value=100000,
            value=10,
            step=10,
            help="Cost assigned to deploying each vehicle."
        )

    with cost_col2:

        scenario_shortage_penalty = st.number_input(
            "Shortage Penalty per Unit",
            min_value=1,
            max_value=1000000,
            value=1000,
            step=100,
            help="Base penalty for each unmet demand unit, before applying zone priority."
        )

    st.caption(
        "The optimizer minimizes vehicle deployment costs plus "
        "priority-weighted unmet-demand penalties."
    )

    run_simulation = st.button(
        "🚀 Run Scenario Simulation",
        type="primary",
        use_container_width=True
    )


# --------------------------------------------------
# EXECUTE SCENARIO SIMULATION
# --------------------------------------------------

if run_simulation:

    payload = {
        "total_fleet": scenario_fleet,
        "vehicle_capacity": scenario_capacity,
        "demand_multiplier": scenario_multiplier,
        "deployment_cost": int(scenario_deployment_cost),
        "shortage_penalty": int(scenario_shortage_penalty),
        "priority_weights": {
            "Zone_A": priority_a,
            "Zone_B": priority_b,
            "Zone_C": priority_c,
            "Zone_D": priority_d
        }
    }

    with st.spinner(
        "Running priority-aware scenario simulation using OR-Tools..."
    ):

        try:

            response = requests.post(
                f"{API_URL}/simulate-scenario",
                json=payload,
                timeout=120
            )

            response.raise_for_status()

            st.session_state["scenario_results"] = response.json()

            st.success(
                "Priority-aware scenario simulation completed successfully!"
            )

        except requests.exceptions.RequestException as e:

            st.error(
                f"Scenario simulation failed: {e}"
            )

            st.session_state.pop(
                "scenario_results",
                None
            )


# --------------------------------------------------
# DISPLAY SCENARIO RESULTS
# --------------------------------------------------

if "scenario_results" in st.session_state:

    scenario = st.session_state["scenario_results"]

    scenario_optimization = scenario.get(
        "optimization_results",
        {}
    )

    if scenario_optimization:

        st.divider()

        st.header("📊 Scenario Results")

        st.caption(
            "These results are generated using your selected "
            "scenario parameters and business priorities."
        )

        # Scenario KPI metrics

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Scenario Demand",
            scenario["scenario_demand"],
            delta=(
                scenario["scenario_demand"]
                - scenario["baseline_predicted_demand"]
            )
        )

        col2.metric(
            "Vehicles Deployed",
            scenario["vehicles_deployed"]
        )

        col3.metric(
            "Total Capacity",
            scenario["total_capacity"]
        )

        col4.metric(
            "Unmet Demand",
            scenario["unmet_demand"]
        )

        st.metric(
            "Overall Fleet Utilization",
            f'{scenario["overall_fleet_utilization"]:.2f}%'
        )

        # Display the cost assumptions used for this scenario

        cost_col1, cost_col2 = st.columns(2)

        cost_col1.metric(
            "Deployment Cost per Vehicle",
            f'{scenario.get("deployment_cost", 10):,}'
        )

        cost_col2.metric(
            "Base Shortage Penalty per Unit",
            f'{scenario.get("shortage_penalty", 1000):,}'
        )

        # Read the aggregate summary returned by the API.
        # Fallback calculations keep the dashboard compatible with older API responses.
        optimization_summary = scenario.get("optimization_summary", {})

        total_deployment_cost = optimization_summary.get(
            "total_deployment_cost",
            sum(
                zone.get("deployment_cost", 0)
                for zone in scenario_optimization.values()
            )
        )

        total_weighted_shortage_cost = optimization_summary.get(
            "total_weighted_shortage_cost",
            scenario.get(
                "total_shortage_cost",
                sum(
                    zone.get("shortage_cost", 0)
                    for zone in scenario_optimization.values()
                )
            )
        )

        total_optimization_objective = optimization_summary.get(
            "total_optimization_objective",
            total_deployment_cost + total_weighted_shortage_cost
        )

        st.subheader("💰 Optimization Cost Breakdown")

        summary_col1, summary_col2, summary_col3 = st.columns(3)

        summary_col1.metric(
            "Total Deployment Cost",
            f"{total_deployment_cost:,}"
        )

        summary_col2.metric(
            "Weighted Shortage Cost",
            f"{total_weighted_shortage_cost:,}"
        )

        summary_col3.metric(
            "Total Optimization Objective",
            f"{total_optimization_objective:,}"
        )

        cost_breakdown_df = pd.DataFrame({
            "Cost Component": [
                "Vehicle Deployment",
                "Unmet-Demand Shortage"
            ],
            "Cost": [
                total_deployment_cost,
                total_weighted_shortage_cost
            ]
        })

        cost_fig = px.bar(
            cost_breakdown_df,
            x="Cost",
            y="Cost Component",
            orientation="h",
            text_auto=",",
            title="Deployment Cost vs Priority-Weighted Shortage Cost"
        )

        cost_fig.update_layout(
            xaxis_title="Cost",
            yaxis_title="",
            showlegend=False
        )

        st.plotly_chart(
            cost_fig,
            use_container_width=True
        )

        # Scenario allocation dataframe

        scenario_df = build_allocation_dataframe(
            scenario_optimization
        )

        st.subheader("Scenario Zone-wise Allocation")

        st.dataframe(
            scenario_df.drop(columns=["Explanation"]),
            use_container_width=True,
            hide_index=True
        )

        # --------------------------------------------------
        # OPTIMIZATION EXPLAINABILITY
        # --------------------------------------------------

        st.divider()

        st.header("🧠 Optimization Explainability")

        st.write(
            "Understand why the optimization engine assigned "
            "vehicles to particular zones and how business priorities "
            "influenced the allocation."
        )

        display_allocation_explanations(
            scenario_optimization,
            "Zone-wise Decision Explanations"
        )

        # Overall shortage cost

        total_shortage_cost = scenario_df["Shortage Cost"].sum()

        total_served = scenario_df["Served Demand"].sum()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Demand Served",
            int(total_served)
        )

        col2.metric(
            "Total Unmet Demand",
            int(scenario_df["Unmet Demand"].sum())
        )

        col3.metric(
            "Total Weighted Shortage Cost",
            f"{total_shortage_cost:,}"
        )

        st.caption(
            "Shortage cost is calculated using the configured "
            "business priority weights and unmet demand penalties."
        )

        # Scenario demand vs capacity

        st.subheader("Scenario Demand vs Capacity")

        scenario_chart_data = scenario_df.melt(
            id_vars="Zone",
            value_vars=[
                "Predicted Demand",
                "Capacity"
            ],
            var_name="Metric",
            value_name="Value"
        )

        scenario_fig = px.bar(
            scenario_chart_data,
            x="Zone",
            y="Value",
            color="Metric",
            barmode="group",
            title="Scenario Demand and Vehicle Capacity"
        )

        st.plotly_chart(
            scenario_fig,
            use_container_width=True
        )

        # Scenario fleet utilization

        st.subheader("Scenario Fleet Utilization")

        utilization_fig = px.bar(
            scenario_df,
            x="Zone",
            y="Fleet Utilization (%)",
            color="Zone",
            text_auto=".2f"
        )

        utilization_fig.update_yaxes(
            range=[0, 100],
            ticksuffix="%"
        )

        st.plotly_chart(
            utilization_fig,
            use_container_width=True
        )

        # Download complete scenario report

        st.download_button(
            "📥 Download Complete Scenario Report",
            scenario_df.to_csv(index=False),
            file_name="transportation_scenario_report.csv",
            mime="text/csv"
        )

    else:

        st.warning(
            "Scenario optimization results are unavailable."
        )
