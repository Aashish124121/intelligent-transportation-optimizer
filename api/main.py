
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.forecast_pipeline import run_forecast_pipeline
from optimization.transport_optimizer import optimize_transportation


app = FastAPI(
    title="Transportation Forecasting & Optimization API",
    description="ML-powered demand forecasting and priority-aware resource allocation",
    version="1.4.0"
)


# --------------------------------------------------
# DEFAULT BUSINESS PRIORITIES
# --------------------------------------------------

DEFAULT_PRIORITIES = {
    "Zone_A": 3.0,
    "Zone_B": 2.0,
    "Zone_C": 1.5,
    "Zone_D": 1.0
}


# --------------------------------------------------
# SCENARIO REQUEST MODEL
# --------------------------------------------------

class ScenarioRequest(BaseModel):

    total_fleet: int = Field(
        default=25,
        ge=1,
        le=1000
    )

    vehicle_capacity: int = Field(
        default=20,
        ge=1,
        le=10000
    )

    demand_multiplier: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0
    )

    priority_weights: dict[str, float] = Field(
        default_factory=lambda: DEFAULT_PRIORITIES.copy()
    )

    deployment_cost: int = Field(
        default=10,
        ge=1,
        le=100000
    )

    shortage_penalty: int = Field(
        default=1000,
        ge=1,
        le=1000000
    )


# --------------------------------------------------
# HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Transportation Forecasting API is running",
        "status": "healthy"
    }


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "UP",
        "service": "Transportation Optimizer"
    }


# --------------------------------------------------
# FORECAST AND DEFAULT OPTIMIZATION
# --------------------------------------------------

@app.post("/forecast-allocate")
def forecast_and_allocate():

    try:

        result = run_forecast_pipeline()

        return {
            "status": "success",
            "message": "Forecasting and optimization completed",
            "predicted_demand": result["predicted_demand"],
            "zone_demand": result["zone_demand"],
            "optimization_results": result["optimization_results"]
        }

    except Exception:

        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during forecasting and optimization."
        )


# --------------------------------------------------
# INTERACTIVE SCENARIO SIMULATION
# --------------------------------------------------

@app.post("/simulate-scenario")
def simulate_scenario(request: ScenarioRequest):

    try:

        # Generate baseline demand forecast
        forecast = run_forecast_pipeline()

        original_zone_demand = forecast["zone_demand"]

        # Apply demand multiplier
        scenario_demand = {
            zone: max(
                0,
                int(round(demand * request.demand_multiplier))
            )
            for zone, demand in original_zone_demand.items()
        }

        scenario_total_demand = sum(scenario_demand.values())

        # --------------------------------------------------
        # VALIDATE BUSINESS PRIORITIES
        # --------------------------------------------------

        if set(request.priority_weights.keys()) != set(scenario_demand.keys()):

            raise HTTPException(
                status_code=422,
                detail="Priority weights must be provided for all demand zones."
            )

        if any(
            not 0 < weight <= 10
            for weight in request.priority_weights.values()
        ):

            raise HTTPException(
                status_code=422,
                detail="Priority weights must be greater than 0 and at most 10."
            )

        # --------------------------------------------------
        # RUN OPTIMIZATION
        # --------------------------------------------------

        optimization_results = optimize_transportation(
            scenario_demand,
            total_fleet=request.total_fleet,
            vehicle_capacity=request.vehicle_capacity,
            priority_weights=request.priority_weights,
            deployment_cost=request.deployment_cost,
            shortage_penalty=request.shortage_penalty
        )

        if optimization_results is None:

            raise HTTPException(
                status_code=503,
                detail="No feasible allocation found."
            )

        # --------------------------------------------------
        # AGGREGATE ALLOCATION METRICS
        # --------------------------------------------------

        total_vehicles_deployed = sum(
            zone["vehicles_assigned"]
            for zone in optimization_results.values()
        )

        total_capacity = sum(
            zone["capacity"]
            for zone in optimization_results.values()
        )

        total_served_demand = sum(
            zone["served_demand"]
            for zone in optimization_results.values()
        )

        total_unmet_demand = sum(
            zone["unmet_demand"]
            for zone in optimization_results.values()
        )

        overall_utilization = (
            total_served_demand / total_capacity * 100
            if total_capacity > 0 else 0
        )

        # --------------------------------------------------
        # OPTIMIZATION COST BREAKDOWN
        # --------------------------------------------------

        total_deployment_cost = sum(
            zone["deployment_cost"]
            for zone in optimization_results.values()
        )

        total_shortage_cost = sum(
            zone["shortage_cost"]
            for zone in optimization_results.values()
        )

        total_optimization_cost = (
            total_deployment_cost + total_shortage_cost
        )

        # --------------------------------------------------
        # RESPONSE
        # --------------------------------------------------

        return {
            "status": "success",
            "message": "Priority-aware scenario simulation completed",

            "baseline_predicted_demand": forecast["predicted_demand"],
            "scenario_demand": scenario_total_demand,

            "total_fleet": request.total_fleet,
            "vehicle_capacity": request.vehicle_capacity,
            "demand_multiplier": request.demand_multiplier,

            "deployment_cost": request.deployment_cost,
            "shortage_penalty": request.shortage_penalty,

            "priority_weights": request.priority_weights,

            "vehicles_deployed": total_vehicles_deployed,
            "total_capacity": total_capacity,

            "total_served_demand": total_served_demand,
            "unmet_demand": total_unmet_demand,

            "overall_fleet_utilization": round(
                overall_utilization, 2
            ),

            "total_shortage_cost": total_shortage_cost,

            "optimization_summary": {
                "total_predicted_demand": scenario_total_demand,
                "total_served_demand": total_served_demand,
                "total_unmet_demand": total_unmet_demand,

                "total_vehicles_deployed": total_vehicles_deployed,
                "total_fleet_available": request.total_fleet,
                "total_capacity": total_capacity,

                "overall_fleet_utilization": round(
                    overall_utilization, 2
                ),

                "total_deployment_cost": total_deployment_cost,
                "total_weighted_shortage_cost": total_shortage_cost,
                "total_optimization_objective": total_optimization_cost
            },

            "zone_demand": scenario_demand,

            "optimization_results": optimization_results
        }

    # --------------------------------------------------
    # ERROR HANDLING
    # --------------------------------------------------

    except HTTPException:
        raise

    except ValueError as e:

        raise HTTPException(
            status_code=422,
            detail=str(e)
        )

    except RuntimeError as e:

        raise HTTPException(
            status_code=503,
            detail=str(e)
        )

    except Exception:

        raise HTTPException(
            status_code=500,
            detail="An unexpected internal server error occurred."
        )
