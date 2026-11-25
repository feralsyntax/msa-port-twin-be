from django.test import SimpleTestCase

from port_twin_app.services.simulation import (
    SimulationBerth,
    SimulationConfig,
    SimulationVessel,
    run_crane_aware_simulation,
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

        result = output.vessel_results[0]

        self.assertEqual(result.status, "incompatible_berth")
        self.assertIsNone(result.berth_id)
        self.assertEqual(
            result.reason,
            "Vessel length exceeds all berth limits. "
            "Vessel draft exceeds all berth limits.",
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
        self.assertEqual(output.vessel_results[2].waiting_hours, 5)

    def test_allocates_vessel_to_berth_available_soonest(self):
        berths = [
            SimulationBerth("B1", 220, 12),
            SimulationBerth("B2", 220, 12),
        ]
        vessels = [
            SimulationVessel("V1", 200, 10),
            SimulationVessel("V2", 180, 9),
            SimulationVessel("V3", 150, 8),
        ]
        config = SimulationConfig(
            duration_hours=24,
            expected_arrivals=3,
            arrival_interval_hours=1,
            service_time_hours=4,
        )

        output = run_simulation(vessels, berths, config)

        self.assertEqual(output.vessel_results[0].berth_id, "B1")
        self.assertEqual(output.vessel_results[1].berth_id, "B2")
        self.assertEqual(output.vessel_results[2].berth_id, "B1")
        self.assertEqual(output.vessel_results[2].waiting_hours, 2)

    def test_simulation_produces_reproducible_results(self):
        first_output = run_simulation(
            self.vessels,
            self.berths,
            self.config,
        )
        second_output = run_simulation(
            self.vessels,
            self.berths,
            self.config,
        )

        self.assertEqual(first_output, second_output)

    def test_caps_berth_utilization_at_simulation_horizon(self):
        config = SimulationConfig(
            duration_hours=6,
            expected_arrivals=1,
            arrival_interval_hours=2,
            service_time_hours=8,
        )

        output = run_simulation(
            [self.vessels[0]],
            self.berths,
            config,
        )

        self.assertEqual(output.vessel_results[0].status, "in_service")
        self.assertAlmostEqual(
            output.summary.berth_utilization_percent,
            100.0,
        )

    def test_handles_simulation_without_berths(self):
        output = run_simulation(
            self.vessels,
            [],
            self.config,
        )

        self.assertEqual(output.summary.total_arrivals, 3)
        self.assertEqual(output.summary.vessels_served, 0)
        self.assertEqual(output.summary.vessels_waiting, 0)
        self.assertEqual(output.summary.vessels_in_service, 0)
        self.assertEqual(output.summary.average_waiting_hours, 0.0)
        self.assertEqual(output.summary.average_turnaround_hours, 0.0)
        self.assertEqual(output.summary.berth_utilization_percent, 0.0)
        self.assertEqual(
            output.vessel_results[0].reason,
            "No berths are configured.",
        )

        self.assertTrue(
            all(
                result.status == "incompatible_berth"
                for result in output.vessel_results
            )
        )

    def test_vessel_starting_at_horizon_is_still_waiting(self):
        config = SimulationConfig(
            duration_hours=8,
            expected_arrivals=2,
            arrival_interval_hours=1,
            service_time_hours=8,
        )

        output = run_simulation(
            self.vessels[:2],
            self.berths,
            config,
        )

        second = output.vessel_results[1]

        self.assertEqual(second.service_start_hour, 8)
        self.assertEqual(second.status, "waiting")
        self.assertEqual(second.waiting_hours, 7)
        self.assertEqual(output.summary.vessels_waiting, 1)
        self.assertEqual(output.summary.vessels_in_service, 0)

    def test_supports_simultaneous_arrivals_in_input_order(self):
        vessels = [
            SimulationVessel("V1", 200, 10),
            SimulationVessel("V2", 180, 9),
            SimulationVessel("V3", 150, 8),
        ]
        berths = [
            SimulationBerth("B1", 220, 12),
            SimulationBerth("B2", 220, 12),
        ]
        config = SimulationConfig(
            duration_hours=12,
            expected_arrivals=3,
            arrival_interval_hours=0,
            service_time_hours=4,
        )

        output = run_simulation(vessels, berths, config)

        first, second, third = output.vessel_results

        self.assertEqual(
            [result.arrival_hour for result in output.vessel_results],
            [0, 0, 0],
        )
        self.assertEqual(first.berth_id, "B1")
        self.assertEqual(second.berth_id, "B2")
        self.assertEqual(third.berth_id, "B1")

        self.assertEqual(first.waiting_hours, 0)
        self.assertEqual(second.waiting_hours, 0)
        self.assertEqual(third.waiting_hours, 4)

        self.assertEqual(output.summary.total_arrivals, 3)
        self.assertEqual(output.summary.vessels_served, 3)

    def test_rejects_negative_arrival_interval(self):
        config = SimulationConfig(24, 3, -1, 8)

        with self.assertRaisesMessage(
            ValueError,
            "Arrival interval must be a finite number greater than or equal to zero.",
        ):
            run_simulation(self.vessels, self.berths, config)

    def test_rejects_invalid_numeric_configuration(self):
        invalid_configs = [
            SimulationConfig(None, 3, 2, 8),
            SimulationConfig(24, 3, None, 8),
            SimulationConfig(24, 3, 2, None),
        ]

        for config in invalid_configs:
            with self.subTest(config=config), self.assertRaises(ValueError):
                run_simulation(self.vessels, self.berths, config)

    def test_rejects_empty_vessel_identifiers(self):
        vessels = [
            SimulationVessel("", 200, 10),
        ]

        with self.assertRaisesMessage(
            ValueError,
            "Vessel identifiers must be non-empty strings.",
        ):
            run_simulation(
                vessels,
                self.berths,
                SimulationConfig(24, 1, 2, 8),
            )

    def test_rejects_empty_berth_identifiers(self):
        berths = [
            SimulationBerth("", 220, 12),
        ]

        with self.assertRaisesMessage(
            ValueError,
            "Berth identifiers must be non-empty strings.",
        ):
            run_simulation(self.vessels, berths, self.config)

    def test_excludes_arrivals_at_simulation_horizon(self):
        config = SimulationConfig(
            duration_hours=4,
            expected_arrivals=3,
            arrival_interval_hours=2,
            service_time_hours=2,
        )

        output = run_simulation(self.vessels, self.berths, config)

        self.assertEqual(output.summary.total_arrivals, 2)
        self.assertEqual(
            [result.arrival_hour for result in output.vessel_results],
            [0, 2],
        )

    def test_handles_zero_expected_arrivals(self):
        config = SimulationConfig(
            duration_hours=24,
            expected_arrivals=0,
            arrival_interval_hours=2,
            service_time_hours=8,
        )

        output = run_simulation([], self.berths, config)

        self.assertEqual(output.vessel_results, ())
        self.assertEqual(output.summary.total_arrivals, 0)
        self.assertEqual(output.summary.vessels_served, 0)
        self.assertEqual(output.summary.average_waiting_hours, 0)
        self.assertEqual(output.summary.average_turnaround_hours, 0)
        self.assertEqual(output.summary.berth_utilization_percent, 0)


class VesselPrioritySchedulingTests(SimpleTestCase):
    def setUp(self):
        self.berths = [
            SimulationBerth("B1", 300, 20),
        ]

    def test_higher_priority_waiting_vessel_is_served_first(self):
        vessels = [
            SimulationVessel("V1", 200, 10, 0, "container"),
            SimulationVessel("V2", 200, 10, 1, "container"),
            SimulationVessel("V3", 200, 10, 2, "tanker"),
        ]
        config = SimulationConfig(
            duration_hours=24,
            expected_arrivals=3,
            arrival_interval_hours=None,
            service_time_hours=4,
            vessel_type_priorities={
                "container": 1,
                "tanker": 0,
            },
        )

        output = run_simulation(vessels, self.berths, config)
        first, second, third = output.vessel_results

        # Results remain in arrival order.
        self.assertEqual(
            [result.vessel_id for result in output.vessel_results],
            ["V1", "V2", "V3"],
        )

        # But the higher-priority tanker is served before V2.
        self.assertEqual(first.service_start_hour, 0)
        self.assertEqual(third.service_start_hour, 4)
        self.assertEqual(second.service_start_hour, 8)

        self.assertEqual(first.service_order, 1)
        self.assertEqual(third.service_order, 2)
        self.assertEqual(second.service_order, 3)

    def test_future_arrival_cannot_jump_ahead_of_waiting_vessel(self):
        vessels = [
            SimulationVessel("V1", 200, 10, 0, "container"),
            SimulationVessel("V2", 200, 10, 1, "tanker"),
            SimulationVessel("V3", 200, 10, 5, "passenger"),
        ]
        config = SimulationConfig(
            duration_hours=24,
            expected_arrivals=3,
            arrival_interval_hours=None,
            service_time_hours=4,
            vessel_type_priorities={
                "container": 2,
                "tanker": 1,
                "passenger": 0,
            },
        )

        output = run_simulation(vessels, self.berths, config)
        first, second, third = output.vessel_results

        self.assertEqual(first.service_start_hour, 0)
        self.assertEqual(second.service_start_hour, 4)
        self.assertEqual(third.service_start_hour, 8)

    def test_equal_priorities_preserve_arrival_order(self):
        vessels = [
            SimulationVessel("V1", 200, 10, 0, "container"),
            SimulationVessel("V2", 200, 10, 1, "tanker"),
            SimulationVessel("V3", 200, 10, 2, "passenger"),
        ]
        config = SimulationConfig(
            duration_hours=24,
            expected_arrivals=3,
            arrival_interval_hours=None,
            service_time_hours=4,
            vessel_type_priorities={
                "container": 1,
                "tanker": 1,
                "passenger": 1,
            },
        )

        output = run_simulation(vessels, self.berths, config)

        self.assertEqual(
            [result.service_start_hour for result in output.vessel_results],
            [0, 4, 8],
        )
        self.assertEqual(
            [result.service_order for result in output.vessel_results],
            [1, 2, 3],
        )

    def test_unconfigured_vessel_types_receive_default_priority(self):
        vessels = [
            SimulationVessel("V1", 200, 10, 0, "container"),
            SimulationVessel("V2", 200, 10, 1, "unknown"),
        ]
        config = SimulationConfig(
            duration_hours=24,
            expected_arrivals=2,
            arrival_interval_hours=None,
            service_time_hours=4,
            vessel_type_priorities={"container": 0},
        )

        output = run_simulation(vessels, self.berths, config)

        # Unconfigured types receive priority 0, so arrival order breaks the tie.
        self.assertEqual(
            [result.service_start_hour for result in output.vessel_results],
            [0, 4],
        )

    def test_rejects_invalid_priority_values(self):
        invalid_priorities = [
            {"container": -1},
            {"container": 1.5},
            {"container": True},
            {"": 1},
        ]

        vessel = SimulationVessel("V1", 200, 10, 0, "container")

        for priorities in invalid_priorities:
            config = SimulationConfig(
                duration_hours=24,
                expected_arrivals=1,
                arrival_interval_hours=None,
                service_time_hours=4,
                vessel_type_priorities=priorities,
            )

            with self.subTest(priorities=priorities), self.assertRaises(ValueError):
                run_simulation([vessel], self.berths, config)


class CraneAwareSimulationTests(SimpleTestCase):
    def setUp(self):
        self.vessels = [
            SimulationVessel("V1", 200, 10),
            SimulationVessel("V2", 180, 9),
        ]
        self.berths = [
            SimulationBerth("B1", 250, 15),
        ]
        self.config = SimulationConfig(
            duration_hours=24,
            expected_arrivals=2,
            arrival_interval_hours=0,
            service_time_hours=8,
            available_cranes=2,
            max_cranes_per_vessel=2,
        )

    def test_baseline_assigns_one_crane_per_vessel(self):
        output = run_crane_aware_simulation(
            self.vessels,
            self.berths,
            self.config,
            policy="baseline",
        )

        self.assertEqual(
            [result.allocated_cranes for result in output.vessel_results],
            [1, 1],
        )
        self.assertEqual(
            [result.service_time_hours for result in output.vessel_results],
            [8, 8],
        )

    def test_optimised_policy_can_allocate_multiple_cranes(self):
        output = run_crane_aware_simulation(
            self.vessels,
            self.berths,
            self.config,
            policy="optimised",
        )

        first, second = output.vessel_results

        self.assertEqual(first.allocated_cranes, 2)
        self.assertEqual(first.service_time_hours, 4)
        self.assertEqual(first.departure_hour, 4)

        self.assertEqual(second.service_start_hour, 4)
        self.assertEqual(second.departure_hour, 8)

    def test_rejects_invalid_policy(self):
        with self.assertRaisesMessage(
            ValueError,
            "Policy must be 'baseline' or 'optimised'.",
        ):
            run_crane_aware_simulation(
                self.vessels,
                self.berths,
                self.config,
                policy="unknown",
            )

    def test_rejects_zero_available_cranes(self):
        config = SimulationConfig(
            24,
            2,
            0,
            8,
            available_cranes=0,
        )

        with self.assertRaisesMessage(
            ValueError,
            "Available cranes must be a positive integer.",
        ):
            run_crane_aware_simulation(
                self.vessels,
                self.berths,
                config,
            )

    def test_rejects_maximum_cranes_above_capacity(self):
        config = SimulationConfig(
            24,
            2,
            0,
            8,
            available_cranes=2,
            max_cranes_per_vessel=3,
        )

        with self.assertRaisesMessage(
            ValueError,
            "Maximum cranes per vessel cannot exceed available cranes.",
        ):
            run_crane_aware_simulation(
                self.vessels,
                self.berths,
                config,
            )

    def test_honours_vessel_type_priorities(self):
        vessels = [
            SimulationVessel(
                "V1",
                200,
                10,
                vessel_type="standard",
            ),
            SimulationVessel(
                "V2",
                180,
                9,
                vessel_type="priority",
            ),
        ]
        config = SimulationConfig(
            duration_hours=24,
            expected_arrivals=2,
            arrival_interval_hours=0,
            service_time_hours=8,
            vessel_type_priorities={
                "standard": 5,
                "priority": 1,
            },
            available_cranes=1,
            max_cranes_per_vessel=1,
        )

        output = run_crane_aware_simulation(
            vessels,
            self.berths,
            config,
            policy="baseline",
        )

        first_served = min(
            output.vessel_results,
            key=lambda result: result.service_order,
        )

        self.assertEqual(first_served.vessel_id, "V2")

    def test_cranes_cannot_be_shared_by_simultaneous_vessels(self):
        vessels = [
            SimulationVessel("V1", 200, 10),
            SimulationVessel("V2", 180, 9),
        ]
        berths = [
            SimulationBerth("B1", 250, 15),
            SimulationBerth("B2", 250, 15),
        ]
        config = SimulationConfig(
            duration_hours=24,
            expected_arrivals=2,
            arrival_interval_hours=0,
            service_time_hours=8,
            available_cranes=1,
            max_cranes_per_vessel=1,
        )

        output = run_crane_aware_simulation(vessels, berths, config, policy="baseline")

        first, second = output.vessel_results

        self.assertEqual(first.service_start_hour, 0)
        self.assertEqual(first.departure_hour, 8)
        self.assertEqual(second.service_start_hour, 8)

    def test_berth_and_crane_availability_both_affect_start_time(self):
        vessels = [
            SimulationVessel("V1", 200, 10),
            SimulationVessel("V2", 180, 9),
        ]
        berths = [
            SimulationBerth("B1", 250, 15),
            SimulationBerth("B2", 250, 15),
        ]
        config = SimulationConfig(
            duration_hours=24,
            expected_arrivals=2,
            arrival_interval_hours=0,
            service_time_hours=8,
            available_cranes=2,
            max_cranes_per_vessel=1,
        )

        output = run_crane_aware_simulation(vessels, berths, config, policy="baseline")

        first, second = output.vessel_results

        self.assertEqual(first.service_start_hour, 0)
        self.assertEqual(second.service_start_hour, 0)
        self.assertNotEqual(first.berth_id, second.berth_id)
