from unittest.mock import patch

from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from port_twin_app.models import TransitAssessment


class TransitAssessmentAPITests(APITestCase):
    @patch("port_twin_app.views.execute_transit_assessment_task.delay")
    def test_create_assessment_returns_202(self, mock_delay):
        response = self.client.post(
            reverse("port_twin_app:transit-assessment-create"),
            {"scenario_name": "calm_tidal_baseline"},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)

        assessment = TransitAssessment.objects.get(pk=response.data["id"])

        self.assertEqual(
            assessment.status,
            TransitAssessment.Status.PENDING,
        )
        self.assertEqual(
            assessment.domain_name,
            TransitAssessment.Domain.APPROACHES,
        )
        mock_delay.assert_called_once_with(str(assessment.id))

    @patch("port_twin_app.views.execute_transit_assessment_task.delay")
    def test_create_assessment_accepts_custom_configuration(self, mock_delay):
        response = self.client.post(
            reverse("port_twin_app:transit-assessment-create"),
            {
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
            },
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)

        assessment = TransitAssessment.objects.get(pk=response.data["id"])

        self.assertEqual(assessment.domain_name, "port_operations")
        self.assertEqual(
            assessment.configuration["maximum_current_speed_ms"],
            0.6,
        )
        self.assertEqual(
            assessment.configuration["draft_classes"][0]["name"],
            "custom",
        )

    @patch("port_twin_app.views.execute_transit_assessment_task.delay")
    def test_invalid_request_is_rejected(self, mock_delay):
        response = self.client.post(
            reverse("port_twin_app:transit-assessment-create"),
            {"scenario_name": "unknown_scenario"},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(TransitAssessment.objects.count(), 0)
        mock_delay.assert_not_called()

    @patch(
        "port_twin_app.views.execute_transit_assessment_task.delay",
        side_effect=RuntimeError("Broker unavailable"),
    )
    def test_queue_failure_marks_assessment_failed(self, mock_delay):
        response = self.client.post(
            reverse("port_twin_app:transit-assessment-create"),
            {"scenario_name": "calm_tidal_baseline"},
            format="json",
        )

        self.assertEqual(
            response.status_code,
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )

        assessment = TransitAssessment.objects.get()

        self.assertEqual(
            assessment.status,
            TransitAssessment.Status.FAILED,
        )
        self.assertTrue(assessment.completed_at)
        self.assertTrue(assessment.error_message)

    @patch("port_twin_app.views.execute_transit_assessment_task.delay")
    def test_retrieve_assessment(self, mock_delay):
        assessment = TransitAssessment.objects.create(
            scenario_name="calm_tidal_baseline",
            domain_name=TransitAssessment.Domain.APPROACHES,
            configuration={},
        )

        response = self.client.get(
            reverse(
                "port_twin_app:transit-assessment-detail",
                kwargs={"pk": assessment.id},
            )
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["id"], str(assessment.id))
        self.assertEqual(response.data["status"], "pending")

    def test_retrieve_unknown_assessment_returns_404(self):
        import uuid

        response = self.client.get(
            reverse(
                "port_twin_app:transit-assessment-detail",
                kwargs={"pk": uuid.uuid4()},
            )
        )

        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
