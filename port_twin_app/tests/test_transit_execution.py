from unittest.mock import patch

from django.test import TestCase

from port_twin_app.models import TransitAssessment
from port_twin_app.services.marine_data.transit.execution import (
    SAFE_ERROR_MESSAGE,
    execute_transit_assessment,
)


class TransitAssessmentExecutionTests(TestCase):
    def setUp(self):
        self.assessment = TransitAssessment.objects.create(
            scenario_name="calm_tidal_baseline",
            domain_name="approaches",
            configuration={
                "draft_classes": [
                    {
                        "name": "small",
                        "representative_draft_m": 5.0,
                        "minimum_ukc_m": 1.0,
                    }
                ],
                "maximum_current_speed_ms": 0.75,
                "maximum_wind_speed_ms": 10.0,
            },
        )

    @patch(
        "port_twin_app.services.marine_data.transit.execution.run_transit_assessment"
    )
    def test_successful_execution_persists_results(self, mock_assessment):
        expected_result = {
            "assessment_type": "experimental",
            "advisory_only": True,
            "draft_classes": [],
        }
        mock_assessment.return_value = expected_result

        result = execute_transit_assessment(self.assessment.id)

        self.assessment.refresh_from_db()

        self.assertEqual(self.assessment.status, TransitAssessment.Status.COMPLETED)
        self.assertEqual(self.assessment.result_data, expected_result)
        self.assertEqual(self.assessment.error_message, "")
        self.assertIsNotNone(self.assessment.started_at)
        self.assertIsNotNone(self.assessment.completed_at)
        self.assertEqual(result.pk, self.assessment.pk)

        mock_assessment.assert_called_once()
        self.assertEqual(
            mock_assessment.call_args.kwargs["scenario_name"],
            "calm_tidal_baseline",
        )
        self.assertEqual(
            mock_assessment.call_args.kwargs["domain_name"],
            "approaches",
        )

        config = mock_assessment.call_args.kwargs["assessment_config"]
        self.assertEqual(len(config.draft_classes), 1)
        self.assertEqual(config.draft_classes[0].name, "small")
        self.assertEqual(config.draft_classes[0].representative_draft_m, 5.0)

    @patch(
        "port_twin_app.services.marine_data.transit.execution.run_transit_assessment",
        side_effect=ValueError("Internal test failure"),
    )
    def test_failed_execution_records_safe_error(self, mock_assessment):
        with self.assertRaisesMessage(ValueError, "Internal test failure"):
            execute_transit_assessment(self.assessment.id)

        self.assessment.refresh_from_db()

        self.assertEqual(self.assessment.status, TransitAssessment.Status.FAILED)
        self.assertEqual(self.assessment.error_message, SAFE_ERROR_MESSAGE)
        self.assertIsNotNone(self.assessment.started_at)
        self.assertIsNotNone(self.assessment.completed_at)
        self.assertIsNone(self.assessment.result_data)

        mock_assessment.assert_called_once()

    @patch(
        "port_twin_app.services.marine_data.transit.execution.run_transit_assessment"
    )
    def test_running_assessment_is_not_executed_again(self, mock_assessment):
        self.assessment.status = TransitAssessment.Status.RUNNING
        self.assessment.save(update_fields=["status"])

        result = execute_transit_assessment(self.assessment.id)

        self.assertEqual(result.status, TransitAssessment.Status.RUNNING)
        mock_assessment.assert_not_called()

    @patch(
        "port_twin_app.services.marine_data.transit.execution.run_transit_assessment"
    )
    def test_completed_assessment_is_not_executed_again(self, mock_assessment):
        self.assessment.status = TransitAssessment.Status.COMPLETED
        self.assessment.save(update_fields=["status"])

        result = execute_transit_assessment(self.assessment.id)

        self.assertEqual(result.status, TransitAssessment.Status.COMPLETED)
        mock_assessment.assert_not_called()

    @patch(
        "port_twin_app.services.marine_data.transit.execution.run_transit_assessment"
    )
    def test_failed_assessment_is_not_executed_again(self, mock_assessment):
        self.assessment.status = TransitAssessment.Status.FAILED
        self.assessment.save(update_fields=["status"])

        result = execute_transit_assessment(self.assessment.id)

        self.assertEqual(result.status, TransitAssessment.Status.FAILED)
        mock_assessment.assert_not_called()

    @patch(
        "port_twin_app.services.marine_data.transit.execution.run_transit_assessment"
    )
    def test_missing_assessment_raises_does_not_exist(self, mock_assessment):
        with self.assertRaises(TransitAssessment.DoesNotExist):
            execute_transit_assessment("00000000-0000-0000-0000-000000000000")

        mock_assessment.assert_not_called()
