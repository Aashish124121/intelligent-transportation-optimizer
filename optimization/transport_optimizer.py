
from ortools.sat.python import cp_model


def optimize_transportation(
    demand,
    total_fleet=25,
    vehicle_capacity=20,
    priority_weights=None,
    deployment_cost=10,
    shortage_penalty=1000,
    time_limit=10
):
    # --------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------

    if not isinstance(demand, dict) or not demand:
        raise ValueError("Demand must be a non-empty dictionary.")

    if any(
        not isinstance(zone, str) or not zone.strip()
        for zone in demand
    ):
        raise ValueError("Every zone must have a valid name.")

    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in demand.values()
    ):
        raise ValueError("Demand values must be non-negative integers.")

    if not isinstance(total_fleet, int) or isinstance(total_fleet, bool) or total_fleet < 1:
        raise ValueError("Total fleet must be a positive integer.")

    if not isinstance(vehicle_capacity, int) or isinstance(vehicle_capacity, bool) or vehicle_capacity < 1:
        raise ValueError("Vehicle capacity must be a positive integer.")

    if not isinstance(deployment_cost, int) or isinstance(deployment_cost, bool) or deployment_cost < 1:
        raise ValueError("Deployment cost must be a positive integer.")

    if not isinstance(shortage_penalty, int) or isinstance(shortage_penalty, bool) or shortage_penalty < 1:
        raise ValueError("Shortage penalty must be a positive integer.")

    if not isinstance(time_limit, (int, float)) or isinstance(time_limit, bool) or time_limit <= 0:
        raise ValueError("Solver time limit must be positive.")

    if priority_weights is None:
        priority_weights = {
            "Zone_A": 3.0,
            "Zone_B": 2.0,
            "Zone_C": 1.5,
            "Zone_D": 1.0
        }

    if not isinstance(priority_weights, dict):
        raise ValueError("Priority weights must be a dictionary.")

    if set(priority_weights.keys()) != set(demand.keys()):
        raise ValueError("Priority weights must be provided for every demand zone.")

    for zone, weight in priority_weights.items():
        if (
            not isinstance(weight, (int, float))
            or isinstance(weight, bool)
            or not 0 < weight <= 10
        ):
            raise ValueError(
                f"Priority for {zone} must be greater than 0 and at most 10."
            )

    # --------------------------------------------------
    # CREATE OPTIMIZATION MODEL
    # --------------------------------------------------

    model = cp_model.CpModel()
    zones = list(demand.keys())

    vehicles = {}
    unmet_demand = {}

    for zone in zones:

        vehicles[zone] = model.new_int_var(
            0, total_fleet, f"vehicles_{zone}"
        )

        unmet_demand[zone] = model.new_int_var(
            0, demand[zone], f"unmet_{zone}"
        )

        model.add(
            vehicles[zone] * vehicle_capacity
            + unmet_demand[zone] >= demand[zone]
        )

    model.add(
        sum(vehicles.values()) <= total_fleet
    )

    # --------------------------------------------------
    # OPTIMIZATION OBJECTIVE
    # --------------------------------------------------

    zone_shortage_rates = {
        zone: int(shortage_penalty * priority_weights[zone])
        for zone in zones
    }

    model.minimize(
        sum(
            deployment_cost * vehicles[zone]
            + zone_shortage_rates[zone] * unmet_demand[zone]
            for zone in zones
        )
    )

    # --------------------------------------------------
    # SOLVE
    # --------------------------------------------------

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)

    status = solver.solve(model)
    status_name = solver.status_name(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):

        raise RuntimeError(
            f"Transportation optimization failed. "
            f"Solver status: {status_name}"
        )

    # --------------------------------------------------
    # PREPARE RESULTS
    # --------------------------------------------------

    results = {}

    total_deployed = 0
    total_demand = sum(demand.values())
    total_capacity = 0
    total_unmet = 0
    total_served = 0

    total_deployment_cost = 0
    total_shortage_cost = 0

    for zone in zones:

        assigned = solver.value(vehicles[zone])
        shortage = solver.value(unmet_demand[zone])

        capacity = assigned * vehicle_capacity
        served_demand = demand[zone] - shortage
        priority = priority_weights[zone]

        zone_deployment_cost = deployment_cost * assigned
        shortage_rate = zone_shortage_rates[zone]
        zone_shortage_cost = shortage_rate * shortage

        zone_total_cost = (
            zone_deployment_cost + zone_shortage_cost
        )

        total_deployed += assigned
        total_capacity += capacity
        total_unmet += shortage
        total_served += served_demand

        total_deployment_cost += zone_deployment_cost
        total_shortage_cost += zone_shortage_cost

        # --------------------------------------------------
        # DECISION EXPLANATIONS
        # --------------------------------------------------

        if assigned > 0 and shortage == 0:

            explanation = (
                f"{assigned} vehicle(s) allocated. "
                f"All {demand[zone]} demand units are served. "
                f"Deployment cost: {zone_deployment_cost}. "
                f"Business priority: {priority}."
            )

        elif assigned > 0 and shortage > 0:

            explanation = (
                f"{assigned} vehicle(s) allocated, serving "
                f"{served_demand} of {demand[zone]} demand units. "
                f"{shortage} units remain unmet because available "
                f"vehicle capacity is limited. "
                f"Deployment cost: {zone_deployment_cost}. "
                f"Shortage cost: {zone_shortage_cost}."
            )

        else:

            explanation = (
                "No vehicles allocated. The optimizer selected "
                "other allocations to minimize the total objective "
                "under the fleet constraint. "
                f"Unmet demand: {shortage}. "
                f"Business priority: {priority}. "
                f"Resulting shortage cost: {zone_shortage_cost}."
            )

        results[zone] = {
            "predicted_demand": demand[zone],
            "business_priority": priority,
            "vehicles_assigned": assigned,
            "capacity": capacity,
            "unmet_demand": shortage,
            "served_demand": served_demand,

            "fleet_utilization": round(
                served_demand / capacity * 100, 2
            ) if capacity > 0 else 0,

            "deployment_cost": zone_deployment_cost,
            "shortage_rate": shortage_rate,
            "shortage_cost": zone_shortage_cost,
            "total_zone_cost": zone_total_cost,
            "explanation": explanation
        }

    # --------------------------------------------------
    # OVERALL SUMMARY
    # --------------------------------------------------

    total_optimization_cost = (
        total_deployment_cost + total_shortage_cost
    )

    overall_utilization = (
        total_served / total_capacity * 100
        if total_capacity > 0 else 0
    )

    # --------------------------------------------------
    # SOLVER DIAGNOSTICS
    # --------------------------------------------------

    print("\nTRANSPORTATION OPTIMIZATION SUMMARY")
    print("-----------------------------------")
    print("Solver status:", status_name)
    print("Objective value:", solver.objective_value)
    print("Best objective bound:", solver.best_objective_bound)

    print("Total demand:", total_demand)
    print("Total served:", total_served)
    print("Total unmet:", total_unmet)
    print("Vehicles deployed:", total_deployed)

    print("Deployment cost:", total_deployment_cost)
    print("Weighted shortage cost:", total_shortage_cost)
    print("Total optimization objective:", total_optimization_cost)

    print("Overall fleet utilization:", round(overall_utilization, 2), "%")

    # Preserve the existing zone-based return structure.
    return results


if __name__ == "__main__":

    predicted_demand = {
        "Zone_A": 28,
        "Zone_B": 18,
        "Zone_C": 14,
        "Zone_D": 10
    }

    custom_priorities = {
        "Zone_A": 1.0,
        "Zone_B": 1.0,
        "Zone_C": 10.0,
        "Zone_D": 1.0
    }

    optimize_transportation(
        predicted_demand,
        total_fleet=2,
        vehicle_capacity=20,
        priority_weights=custom_priorities,
        deployment_cost=10,
        shortage_penalty=1000
    )
