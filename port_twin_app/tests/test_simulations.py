from django.test import SimpleTestCase

from port_twin_app.services.simulation import (
    SimulationBerth,
    SimulationConfig,
    SimulationVessel,
    run_simulation,
)


class SimulationEngineTests(SimpleTestCase):
    def setUp(self):
        self.vessels = [
            SimulationVessel("V1", 200, 10),
            SimulationVessel("V2", 180, 9),
            SimulationVessel("V3", 150, 8),
        ]
        self.berths = [
            SimulationBerth("B1", 220, 12),
        ]
        self.config = SimulationConfig(
            duration_hours=24,
            expected_arrivals=3,
            arrival_interval_hours=2,
            service_time_hours=8,
        )

    def test_allocates_vessels_and_calculates_waiting_time(self):
        output = run_simulation(self.vessels, self.berths, self.config)

        self.assertEqual(len(output.vessel_results), 3)

        first, second, third = output.vessel_results

        self.assertEqual(first.status, "completed")
        self.assertEqual(first.waiting_hours, 0)

        self.assertEqual(second.status, "completed")
        self.assertEqual(second.waiting_hours, 6)

        self.assertEqual(third.status, "completed")
        self.assertEqual(third.waiting_hours, 12)

        self.assertEqual(output.summary.vessels_served, 3)
        self.assertEqual(output.summary.average_waiting_hours, 9)

    def test_rejects_more_arrivals_than_supplied_vessels(self):
        config = SimulationConfig(
            duration_hours=24,
            expected_arrivals=4,
            arrival_interval_hours=2,
            service_time_hours=8,
        )

        with self.assertRaises(ValueError):
            run_simulation(self.vessels, self.berths, config)

    def test_identifies_vessels_with_no_compatible_berth(self):
        small_berth = SimulationBerth("B2", 100, 5)

        output = run_simulation(
            [self.vessels[0]],
            [small_berth],
            SimulationConfig(24, 1, 2, 8),
        )

        self.assertEqual(
            output.vessel_results[0].status,
            "incompatible_berth",
        )

    def test_calculates_utilization(self):
        output = run_simulation(self.vessels, self.berths, self.config)

        # Three 8-hour services occupy the berth for all 24 hours.
        self.assertAlmostEqual(
            output.summary.berth_utilization_percent,
            100.0,
        )
