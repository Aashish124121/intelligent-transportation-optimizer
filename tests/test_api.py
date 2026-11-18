
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "UP"


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_simulate_scenario_success():
    payload = {
        "total_fleet": 1,
        "vehicle_capacity": 20,
        "demand_multiplier": 1.0,
        "priority_weights": {
            "Zone_A": 1,
            "Zone_B": 10,
            "Zone_C": 1,
            "Zone_D": 1
        },
        "deployment_cost": 50,
        "shortage_penalty": 2000
    }

    response = client.post("/simulate-scenario", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["scenario_demand"] == 35
    assert data["vehicles_deployed"] == 1
    assert data["unmet_demand"] == 26

    assert data["optimization_summary"]["total_deployment_cost"] == 50
    assert data["optimization_summary"]["total_weighted_shortage_cost"] == 52000
    assert data["optimization_summary"]["total_optimization_objective"] == 52050

    assert data["optimization_results"]["Zone_B"]["vehicles_assigned"] == 1


def test_invalid_priority_weights():
    payload = {
        "total_fleet": 1,
        "vehicle_capacity": 20,
        "demand_multiplier": 1.0,
        "priority_weights": {
            "Zone_A": 1,
            "Zone_B": 15,
            "Zone_C": 1,
            "Zone_D": 1
        },
        "deployment_cost": 50,
        "shortage_penalty": 2000
    }

    response = client.post("/simulate-scenario", json=payload)

    assert response.status_code == 422


def test_invalid_vehicle_capacity():
    payload = {
        "total_fleet": 1,
        "vehicle_capacity": 0,
        "demand_multiplier": 1.0,
        "priority_weights": {
            "Zone_A": 1,
            "Zone_B": 2,
            "Zone_C": 1.5,
            "Zone_D": 1
        },
        "deployment_cost": 50,
        "shortage_penalty": 2000
    }

    response = client.post("/simulate-scenario", json=payload)

    assert response.status_code == 422


def test_invalid_deployment_cost():
    payload = {
        "total_fleet": 1,
        "vehicle_capacity": 20,
        "demand_multiplier": 1.0,
        "priority_weights": {
            "Zone_A": 1,
            "Zone_B": 2,
            "Zone_C": 1.5,
            "Zone_D": 1
        },
        "deployment_cost": 0,
        "shortage_penalty": 2000
    }

    response = client.post("/simulate-scenario", json=payload)

    assert response.status_code == 422
