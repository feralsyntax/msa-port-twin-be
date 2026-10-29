import logging

from rest_framework import serializers, status
from rest_framework.exceptions import APIException
from rest_framework.response import Response
from rest_framework.views import exception_handler as drf_exception_handler

logger = logging.getLogger(__name__)


def custom_exception_handler(exc, context):
    """Standardize API errors without exposing unexpected exception details."""

    response = drf_exception_handler(exc, context)
    request = context.get("request")

    if response is not None:
        if isinstance(exc, serializers.ValidationError):
            return response

        if isinstance(exc, APIException):
            message = str(exc.detail)
            code = exc.default_code
        else:
            message = "The request could not be completed."
            code = "request_error"

        response.data = {
            "error": {
                "code": str(code),
                "message": message,
                "details": response.data,
            }
        }
        return response

    logger.error(
        "Unhandled API exception: method=%s path=%s exception_type=%s",
        getattr(request, "method", "unknown"),
        getattr(request, "path", "unknown"),
        type(exc).__name__,
    )

    return Response(
        {
            "error": {
                "code": "internal_server_error",
                "message": "An unexpected error occurred.",
                "details": None,
            }
        },
        status=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )
