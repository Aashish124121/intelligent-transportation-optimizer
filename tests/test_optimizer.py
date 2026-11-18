
import unittest

from optimization.transport_optimizer import optimize_transportation


class TestTransportationOptimizer(unittest.TestCase):

    def test_high_priority_zone_gets_vehicle(self):

        demand = {
            "Zone_A": 14,
            "Zone_B": 9,
            "Zone_C": 7,
            "Zone_D": 5
        }

        priorities = {
            "Zone_A": 1,
            "Zone_B": 10,
            "Zone_C": 1,
            "Zone_D": 1
        }

        results = optimize_transportation(
            demand,
            total_fleet=1,
            vehicle_capacity=20,
            priority_weights=priorities,
            deployment_cost=50,
            shortage_penalty=2000
        )

        # Verify high-priority zone receives the vehicle
        self.assertEqual(
            results["Zone_B"]["vehicles_assigned"], 1
        )

        # Verify fleet constraint
        total_vehicles = sum(
            result["vehicles_assigned"]
            for result in results.values()
        )

        self.assertLessEqual(total_vehicles, 1)

        # Verify shortage cost uses the configured penalty
        self.assertEqual(
            results["Zone_A"]["shortage_cost"], 28000
        )

    def test_negative_demand_rejected(self):

        with self.assertRaises(ValueError):
            optimize_transportation(
                {"Zone_A": -5},
                priority_weights={"Zone_A": 1}
            )

    def test_zero_vehicle_capacity_rejected(self):

        with self.assertRaises(ValueError):
            optimize_transportation(
                {"Zone_A": 10},
                vehicle_capacity=0,
                priority_weights={"Zone_A": 1}
            )

    def test_invalid_priority_rejected(self):

        with self.assertRaises(ValueError):
            optimize_transportation(
                {"Zone_A": 10},
                priority_weights={"Zone_A": 15}
            )


if __name__ == "__main__":
    unittest.main()

import unittest

from optimization.transport_optimizer import optimize_transportation


class TestTransportationOptimizer(unittest.TestCase):

    def test_high_priority_zone_gets_vehicle(self):

        demand = {
            "Zone_A": 14,
            "Zone_B": 9,
            "Zone_C": 7,
            "Zone_D": 5
        }

        priorities = {
            "Zone_A": 1,
            "Zone_B": 10,
            "Zone_C": 1,
            "Zone_D": 1
        }

        results = optimize_transportation(
            demand,
            total_fleet=1,
            vehicle_capacity=20,
            priority_weights=priorities,
            deployment_cost=50,
            shortage_penalty=2000
        )

        # Verify high-priority zone receives the vehicle
        self.assertEqual(
            results["Zone_B"]["vehicles_assigned"], 1
        )

        # Verify fleet constraint
        total_vehicles = sum(
            result["vehicles_assigned"]
            for result in results.values()
        )

        self.assertLessEqual(total_vehicles, 1)

        # Verify shortage cost uses the configured penalty
        self.assertEqual(
            results["Zone_A"]["shortage_cost"], 28000
        )

    def test_negative_demand_rejected(self):

        with self.assertRaises(ValueError):
            optimize_transportation(
                {"Zone_A": -5},
                priority_weights={"Zone_A": 1}
            )

    def test_zero_vehicle_capacity_rejected(self):

        with self.assertRaises(ValueError):
            optimize_transportation(
                {"Zone_A": 10},
                vehicle_capacity=0,
                priority_weights={"Zone_A": 1}
            )

    def test_invalid_priority_rejected(self):

        with self.assertRaises(ValueError):
            optimize_transportation(
                {"Zone_A": 10},
                priority_weights={"Zone_A": 15}
            )


if __name__ == "__main__":
    unittest.main()
