from unittest.mock import patch

from django.test import SimpleTestCase
from rest_framework import serializers, status
from rest_framework.exceptions import APIException
from rest_framework.test import APIRequestFactory

from port_twin_app.exceptions import custom_exception_handler


class CustomExceptionHandlerTests(SimpleTestCase):
    def setUp(self):
        self.factory = APIRequestFactory()
        self.request = self.factory.get("/api/v1/test/")
        self.context = {"request": self.request}

    def test_validation_errors_keep_existing_response_shape(self):
        exc = serializers.ValidationError({"name": ["This field is required."]})

        response = custom_exception_handler(exc, self.context)

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("name", response.data)

    def test_api_exceptions_use_standardized_error_envelope(self):
        class ExampleAPIException(APIException):
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            default_detail = "Service unavailable."
            default_code = "service_unavailable"

        response = custom_exception_handler(
            ExampleAPIException(),
            self.context,
        )

        self.assertEqual(
            response.status_code,
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )
        self.assertEqual(
            response.data["error"]["code"],
            "service_unavailable",
        )
        self.assertEqual(
            response.data["error"]["message"],
            "Service unavailable.",
        )

    def test_unexpected_errors_return_safe_generic_response(self):
        sensitive_message = "secret-token-value"
        exc = RuntimeError(sensitive_message)

        with patch("port_twin_app.exceptions.logger.error") as log_error:
            response = custom_exception_handler(exc, self.context)

        self.assertEqual(
            response.status_code,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
        self.assertEqual(
            response.data["error"]["message"],
            "An unexpected error occurred.",
        )
        self.assertNotIn(sensitive_message, str(response.data))
        self.assertNotIn(sensitive_message, str(log_error.call_args))
