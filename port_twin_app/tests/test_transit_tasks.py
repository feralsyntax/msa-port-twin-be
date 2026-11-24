from unittest.mock import patch

from django.test import SimpleTestCase

from port_twin_app.tasks import execute_transit_assessment_task


class TransitAssessmentTaskTests(SimpleTestCase):
    @patch("port_twin_app.tasks.execute_transit_assessment")
    def test_task_executes_assessment(self, mock_execute):
        assessment_id = "test-assessment-id"

        execute_transit_assessment_task.run(assessment_id)

        mock_execute.assert_called_once_with(assessment_id)

    @patch(
        "port_twin_app.tasks.execute_transit_assessment",
        side_effect=ValueError("Internal test failure"),
    )
    def test_task_reraises_execution_failure(self, mock_execute):
        assessment_id = "test-assessment-id"

        with self.assertRaises(ValueError):
            execute_transit_assessment_task.run(assessment_id)

        mock_execute.assert_called_once_with(assessment_id)
