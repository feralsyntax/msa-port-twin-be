from datetime import timedelta

from django.contrib.gis.geos import Point
from django.test import SimpleTestCase, TestCase
from django.utils import timezone

from port_twin_app.models import Vessel, VesselPosition
from port_twin_app.services.ais.eta import estimate_vessel_eta
from port_twin_app.services.ais.tracking import (
    VesselZone,
    classify_position,
)


class VesselTrackingTests(SimpleTestCase):
    def test_classifies_port_operations_position(self):
        self.assertEqual(
            classify_position(39.65, -4.07),
            VesselZone.PORT_OPERATIONS,
        )

    def test_classifies_approaches_position(self):
        self.assertEqual(
            classify_position(39.75, -4.15),
            VesselZone.APPROACHES,
        )

    def test_classifies_position_outside_approaches(self):
        self.assertEqual(
            classify_position(40.0, -4.05),
            VesselZone.OUTSIDE_APPROACHES,
        )

    def test_returns_unknown_for_missing_coordinates(self):
        self.assertEqual(
            classify_position(None, None),
            VesselZone.UNKNOWN,
        )

    def test_returns_unknown_for_invalid_coordinates(self):
        self.assertEqual(
            classify_position(181, -4.05),
            VesselZone.UNKNOWN,
        )


class VesselETATests(TestCase):
    def setUp(self):
        self.vessel = Vessel.objects.create(name="Test Vessel")
        self.now = timezone.now()

    def create_position(
        self,
        longitude=39.58,
        latitude=-4.07,
        recorded_at=None,
        speed=10,
        course=90,
        source=VesselPosition.Source.AIS,
    ):
        return VesselPosition.objects.create(
            vessel=self.vessel,
            source=source,
            position=Point(longitude, latitude, srid=4326),
            speed_over_ground=speed,
            course_over_ground=course,
            recorded_at=recorded_at or self.now,
        )

    def create_reliable_track(self, **latest_position_fields):
        latest_recorded_at = latest_position_fields.pop(
            "recorded_at",
            self.now - timedelta(minutes=5),
        )

        self.create_position(
            recorded_at=latest_recorded_at - timedelta(minutes=5),
        )
        self.create_position(
            recorded_at=latest_recorded_at,
            **latest_position_fields,
        )

    def test_estimates_eta_for_eastbound_vessel(self):
        self.create_reliable_track()

        eta = estimate_vessel_eta(self.vessel, now=self.now)

        self.assertIsNotNone(eta)
        self.assertEqual(eta.vessel_id, str(self.vessel.pk))
        self.assertGreater(eta.eta_hours, 0)
        self.assertAlmostEqual(
            eta.eta_at.timestamp(),
            (self.now + timedelta(hours=eta.eta_hours)).timestamp(),
            places=3,
        )

    def test_returns_no_eta_for_vessel_heading_away(self):
        self.create_reliable_track(course=270)

        eta = estimate_vessel_eta(self.vessel, now=self.now)

        self.assertIsNone(eta)

    def test_returns_no_eta_for_vessel_already_in_port_operations(self):
        self.create_reliable_track(
            longitude=39.65,
            latitude=-4.07,
        )

        eta = estimate_vessel_eta(self.vessel, now=self.now)

        self.assertIsNone(eta)

    def test_returns_no_eta_when_speed_is_missing(self):
        self.create_reliable_track(speed=None)

        eta = estimate_vessel_eta(self.vessel, now=self.now)

        self.assertIsNone(eta)

    def test_returns_no_eta_when_course_is_missing(self):
        self.create_reliable_track(course=None)

        eta = estimate_vessel_eta(self.vessel, now=self.now)

        self.assertIsNone(eta)

    def test_returns_no_eta_for_zero_speed(self):
        self.create_reliable_track(speed=0)

        eta = estimate_vessel_eta(self.vessel, now=self.now)

        self.assertIsNone(eta)

    def test_returns_no_eta_for_invalid_course(self):
        self.create_reliable_track(course=360)

        eta = estimate_vessel_eta(self.vessel, now=self.now)

        self.assertIsNone(eta)

    def test_returns_no_eta_for_stale_track(self):
        self.create_reliable_track(
            recorded_at=self.now - timedelta(minutes=20),
        )

        eta = estimate_vessel_eta(self.vessel, now=self.now)

        self.assertIsNone(eta)

    def test_returns_no_eta_with_insufficient_observations(self):
        self.create_position(recorded_at=self.now - timedelta(minutes=5))

        eta = estimate_vessel_eta(self.vessel, now=self.now)

        self.assertIsNone(eta)

    def test_ignores_manual_positions(self):
        self.create_position(
            recorded_at=self.now - timedelta(minutes=10),
            source=VesselPosition.Source.MANUAL,
        )
        self.create_position(
            recorded_at=self.now - timedelta(minutes=5),
            source=VesselPosition.Source.MANUAL,
        )

        eta = estimate_vessel_eta(self.vessel, now=self.now)

        self.assertIsNone(eta)
