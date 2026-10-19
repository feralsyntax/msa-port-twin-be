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
        self.assertEqual(output.summary.average_waiting_hours, 6)

    def test_rejects_more_arrivals_than_supplied_vessels(self):
        config = SimulationConfig(
            duration_hours=24,
            expected_arrivals=4,
            arrival_interval_hours=2,
            service_time_hours=8,
        )

        with self.assertRaises(ValueError):
            run_simulation(self.vessels, self.berths, config)

    def test_rejects_negative_expected_arrivals(self):
        config = SimulationConfig(24, -1, 2, 8)

        with self.assertRaisesMessage(
            ValueError,
            "Expected arrivals cannot be negative.",
        ):
            run_simulation(self.vessels, self.berths, config)

    def test_rejects_non_integer_expected_arrivals(self):
        config = SimulationConfig(24, 1.5, 2, 8)

        with self.assertRaisesMessage(
            ValueError,
            "Expected arrivals must be an integer.",
        ):
            run_simulation(self.vessels, self.berths, config)

    def test_rejects_non_finite_simulation_duration(self):
        config = SimulationConfig(float("inf"), 3, 2, 8)

        with self.assertRaisesMessage(
            ValueError,
            "Simulation duration must be a finite number greater than zero.",
        ):
            run_simulation(self.vessels, self.berths, config)

    def test_rejects_duplicate_vessel_identifiers(self):
        duplicate_vessels = [
            SimulationVessel("V1", 200, 10),
            SimulationVessel("V1", 180, 9),
        ]
        config = SimulationConfig(24, 2, 2, 8)

        with self.assertRaisesMessage(
            ValueError,
            "Vessel identifiers must be unique.",
        ):
            run_simulation(duplicate_vessels, self.berths, config)

    def test_rejects_duplicate_berth_identifiers(self):
        duplicate_berths = [
            SimulationBerth("B1", 220, 12),
            SimulationBerth("B1", 250, 15),
        ]

        with self.assertRaisesMessage(
            ValueError,
            "Berth identifiers must be unique.",
        ):
            run_simulation(self.vessels, duplicate_berths, self.config)

    def test_rejects_non_positive_vessel_dimensions(self):
        invalid_vessels = [
            SimulationVessel("V1", 0, 10),
        ]

        with self.assertRaisesMessage(
            ValueError,
            "Vessel length and draft must be finite numbers greater than zero.",
        ):
            run_simulation(
                invalid_vessels,
                self.berths,
                SimulationConfig(24, 1, 2, 8),
            )

    def test_rejects_non_positive_berth_capacity(self):
        invalid_berths = [
            SimulationBerth("B1", 220, 0),
        ]

        with self.assertRaisesMessage(
            ValueError,
            "Berth capacity limits must be finite numbers greater than zero.",
        ):
            run_simulation(self.vessels, invalid_berths, self.config)

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

    def test_average_waiting_excludes_vessels_not_started_by_horizon(self):
        vessels = [
            SimulationVessel("V1", 200, 10),
            SimulationVessel("V2", 180, 9),
            SimulationVessel("V3", 150, 8),
        ]
        config = SimulationConfig(
            duration_hours=7,
            expected_arrivals=3,
            arrival_interval_hours=1,
            service_time_hours=4,
        )

        output = run_simulation(vessels, self.berths, config)

        # V1 waits 0 hours; V2 waits 3 hours.
        # V3 has not started service by the 7-hour horizon.
        self.assertAlmostEqual(output.summary.average_waiting_hours, 1.5)
        self.assertEqual(output.summary.vessels_waiting, 1)
        self.assertEqual(output.summary.vessels_in_service, 1)
