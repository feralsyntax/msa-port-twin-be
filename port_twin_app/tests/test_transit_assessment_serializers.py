from django.test import SimpleTestCase

from port_twin_app.models import TransitAssessment
from port_twin_app.serializers import (
    TransitAssessmentCreateRequestSerializer,
)


class TransitAssessmentRequestSerializerTests(SimpleTestCase):
    def test_defaults_are_applied(self):
        serializer = TransitAssessmentCreateRequestSerializer(
            data={"scenario_name": "calm_tidal_baseline"}
        )

        self.assertTrue(serializer.is_valid(), serializer.errors)

        data = serializer.validated_data

        self.assertEqual(
            data["domain_name"],
            TransitAssessment.Domain.APPROACHES,
        )
        self.assertEqual(
            data["configuration"]["maximum_current_speed_ms"],
            0.75,
        )
        self.assertEqual(
            data["configuration"]["maximum_wind_speed_ms"],
            10.0,
        )
        self.assertEqual(
            len(data["configuration"]["draft_classes"]),
            3,
        )

    def test_custom_configuration_is_accepted(self):
        serializer = TransitAssessmentCreateRequestSerializer(
            data={
                "scenario_name": "calm_tidal_baseline",
                "domain_name": "port_operations",
                "draft_classes": [
                    {
                        "name": "custom",
                        "representative_draft_m": 8.0,
                        "minimum_ukc_m": 1.2,
                    }
                ],
                "maximum_current_speed_ms": 0.6,
                "maximum_wind_speed_ms": 8.0,
            }
        )

        self.assertTrue(serializer.is_valid(), serializer.errors)

        data = serializer.validated_data

        self.assertEqual(data["domain_name"], "port_operations")
        self.assertEqual(
            data["configuration"]["maximum_current_speed_ms"],
            0.6,
        )
        self.assertEqual(
            data["configuration"]["maximum_wind_speed_ms"],
            8.0,
        )
        self.assertEqual(
            data["configuration"]["draft_classes"][0]["name"],
            "custom",
        )

    def test_unknown_scenario_is_rejected(self):
        serializer = TransitAssessmentCreateRequestSerializer(
            data={"scenario_name": "unknown_scenario"}
        )

        self.assertFalse(serializer.is_valid())
        self.assertIn("scenario_name", serializer.errors)

    def test_invalid_domain_is_rejected(self):
        serializer = TransitAssessmentCreateRequestSerializer(
            data={
                "scenario_name": "calm_tidal_baseline",
                "domain_name": "unknown_domain",
            }
        )

        self.assertFalse(serializer.is_valid())
        self.assertIn("domain_name", serializer.errors)

    def test_negative_current_limit_is_rejected(self):
        serializer = TransitAssessmentCreateRequestSerializer(
            data={
                "scenario_name": "calm_tidal_baseline",
                "maximum_current_speed_ms": -1,
            }
        )

        self.assertFalse(serializer.is_valid())
        self.assertIn("maximum_current_speed_ms", serializer.errors)

    def test_duplicate_draft_class_names_are_rejected(self):
        serializer = TransitAssessmentCreateRequestSerializer(
            data={
                "scenario_name": "calm_tidal_baseline",
                "draft_classes": [
                    {
                        "name": "custom",
                        "representative_draft_m": 5,
                        "minimum_ukc_m": 1,
                    },
                    {
                        "name": "custom",
                        "representative_draft_m": 10,
                        "minimum_ukc_m": 1.5,
                    },
                ],
            }
        )

        self.assertFalse(serializer.is_valid())
        self.assertIn("non_field_errors", serializer.errors)
