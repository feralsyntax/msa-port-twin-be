from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from port_twin_app.models import MarineCondition, MarinePrediction


class DataStatusAPITests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.url = "/api/v1/data-status/"

    def test_empty_data_returns_null_timestamps_and_zero_counts(self):
        response = self.client.get(self.url)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["marine_conditions"]["total_records"], 0)
        self.assertEqual(response.data["marine_conditions"]["source_counts"], [])
        self.assertIsNone(response.data["marine_conditions"]["latest_recorded_at"])
        self.assertEqual(response.data["vessel_positions"]["total_records"], 0)
        self.assertIsNone(response.data["vessel_positions"]["latest_recorded_at"])
        self.assertIsNone(response.data["ai"]["latest_prediction_at"])

    def test_marine_condition_provenance_is_reported(self):
        MarineCondition.objects.create(
            location="SRID=4326;POINT (39.6682 -4.0435)",
            recorded_at="2026-10-17T08:00:00Z",
            source="demo",
        )
        MarineCondition.objects.create(
            location="SRID=4326;POINT (39.6682 -4.0435)",
            recorded_at="2026-10-18T08:00:00Z",
            source="observed",
        )
        response = self.client.get(self.url)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["marine_conditions"]["total_records"], 2)
        self.assertEqual(
            response.data["marine_conditions"]["source_counts"],
            [
                {"source": "demo", "count": 1},
                {"source": "observed", "count": 1},
            ],
        )
        self.assertIsNotNone(response.data["marine_conditions"]["latest_recorded_at"])
        self.assertIsNotNone(response.data["marine_conditions"]["latest_ingested_at"])

    @override_settings(
        MARINE_AI_BACKEND="local",
        MARINE_AI_CHECKPOINT="/tmp/msa-port-twin-missing-checkpoint.pt",
    )
    def test_missing_local_checkpoint_is_reported_as_unavailable(self):
        response = self.client.get(self.url)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["ai"]["backend"], "local")
        self.assertFalse(response.data["ai"]["configured"])
        self.assertEqual(response.data["ai"]["readiness"], "unavailable")

    @override_settings(
        MARINE_AI_BACKEND="modelarts",
        MARINE_AI_MODELARTS_ENDPOINT="https://example.com/inference",
        MARINE_AI_MODELARTS_VERSION="modelarts-test",
    )
    def test_modelarts_configuration_does_not_claim_verified_readiness(self):
        response = self.client.get(self.url)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data["ai"]["configured"])
        self.assertEqual(response.data["ai"]["readiness"], "configured")
        self.assertEqual(
            response.data["ai"]["configured_model_version"],
            "modelarts-test",
        )

    def test_latest_prediction_metadata_is_exposed(self):
        prediction = MarinePrediction.objects.create(
            grid={"west": 39.5, "east": 39.8},
            forecast={"steps": 3},
            input_data={"source": "test"},
            result_data={"status": "completed"},
            model_version="fno-test-1",
            contract_version="1.0.0",
            status=MarinePrediction.Status.COMPLETED,
        )

        response = self.client.get(self.url)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.data["ai"]["latest_prediction_status"],
            prediction.status,
        )
        self.assertEqual(
            response.data["ai"]["latest_prediction_model_version"],
            prediction.model_version,
        )
        self.assertEqual(
            response.data["ai"]["latest_prediction_contract_version"],
            prediction.contract_version,
        )
        self.assertIsNotNone(response.data["ai"]["latest_prediction_at"])
