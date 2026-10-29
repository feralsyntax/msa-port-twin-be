"""Production settings for MSA Port Twin."""

import os

from decouple import config

# Set the mode before loading shared settings so production cache
# and API throttling configuration are selected.
os.environ["MODE"] = "prod"

from .settings import *

# Explicit production defaults
MODE = "prod"
IS_PRODUCTION = True
DEBUG = False

SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_REFERRER_POLICY = "same-origin"
X_FRAME_OPTIONS = "DENY"

# Enable HTTPS redirects in production. If TLS terminates at a
# reverse proxy, configure that proxy correctly to avoid redirect loops.
SECURE_SSL_REDIRECT = config(
    "SECURE_SSL_REDIRECT",
    default=True,
    cast=bool,
)

SECURE_HSTS_SECONDS = config(
    "SECURE_HSTS_SECONDS",
    default=0,
    cast=int,
)
SECURE_HSTS_INCLUDE_SUBDOMAINS = config(
    "SECURE_HSTS_INCLUDE_SUBDOMAINS",
    default=False,
    cast=bool,
)
SECURE_HSTS_PRELOAD = config(
    "SECURE_HSTS_PRELOAD",
    default=False,
    cast=bool,
)


# Production validation
production_errors = []

if not os.environ.get("SECRET_KEY", "").strip():
    production_errors.append("SECRET_KEY must be explicitly configured.")

if len(SECRET_KEY) < 50:
    production_errors.append(
        "SECRET_KEY must contain at least 50 characters in production."
    )

if not os.environ.get("DATABASE_URL", "").strip():
    production_errors.append(
        "DATABASE_URL must be explicitly configured for production."
    )

if not ALLOWED_HOSTS or "*" in ALLOWED_HOSTS:
    production_errors.append(
        "ALLOWED_HOSTS must contain explicit production hostnames."
    )

if CORS_ALLOW_ALL_ORIGINS:
    production_errors.append("CORS_ALLOW_ALL_ORIGINS must be False in production.")

if not CORS_ALLOWED_ORIGINS:
    production_errors.append("CORS_ALLOWED_ORIGINS must be configured in production.")

if any(not origin.startswith("https://") for origin in CORS_ALLOWED_ORIGINS):
    production_errors.append("Production CORS origins must use HTTPS.")

if any(not origin.startswith("https://") for origin in CSRF_TRUSTED_ORIGINS):
    production_errors.append("Production CSRF trusted origins must use HTTPS.")

if not REDIS_URL:
    production_errors.append("REDIS_URL must be explicitly configured in production.")

if MARINE_AI_BACKEND not in {"local", "modelarts"}:
    production_errors.append("MARINE_AI_BACKEND must be either 'local' or 'modelarts'.")

if MARINE_AI_BACKEND == "modelarts":
    if not MARINE_AI_MODELARTS_ENDPOINT:
        production_errors.append(
            "MARINE_AI_MODELARTS_ENDPOINT is required for ModelArts."
        )
    if not MARINE_AI_MODELARTS_TOKEN:
        production_errors.append("MARINE_AI_MODELARTS_TOKEN is required for ModelArts.")

if production_errors:
    raise RuntimeError(
        "Invalid production configuration:\n- " + "\n- ".join(production_errors)
    )
