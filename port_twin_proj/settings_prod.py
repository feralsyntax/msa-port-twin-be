"""Production settings for MSA Port Twin."""

import os

if not os.environ.get("SECRET_KEY"):
    raise RuntimeError("SECRET_KEY must be set in production.")

os.environ["MODE"] = "prod"

from .settings import *

DEBUG = False

if not ALLOWED_HOSTS or "*" in ALLOWED_HOSTS:
    raise RuntimeError("Set explicit production ALLOWED_HOSTS.")

if not CORS_ALLOWED_ORIGINS:
    raise RuntimeError("Set production CORS_ALLOWED_ORIGINS.")

if any(not origin.startswith("https://") for origin in CORS_ALLOWED_ORIGINS):
    raise RuntimeError("Production CORS origins must use HTTPS.")

if any(not origin.startswith("https://") for origin in CSRF_TRUSTED_ORIGINS):
    raise RuntimeError("Production CSRF trusted origins must use HTTPS.")

SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_REFERRER_POLICY = "same-origin"
X_FRAME_OPTIONS = "DENY"

SECURE_SSL_REDIRECT = True
SECURE_HSTS_SECONDS = int(os.environ.get("SECURE_HSTS_SECONDS", "0"))
SECURE_HSTS_INCLUDE_SUBDOMAINS = os.environ.get(
    "SECURE_HSTS_INCLUDE_SUBDOMAINS", "False"
).lower() in ("true", "1", "yes")
SECURE_HSTS_PRELOAD = os.environ.get("SECURE_HSTS_PRELOAD", "False").lower() in (
    "true",
    "1",
    "yes",
)

if not DATABASES["default"].get("HOST") and not os.environ.get("DATABASE_URL"):
    raise RuntimeError("Configure DATABASE_URL for production.")
