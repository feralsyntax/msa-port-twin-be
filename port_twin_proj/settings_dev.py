"""Development settings for MSA Port Twin."""

from .settings import *

MODE = "dev"
IS_PRODUCTION = False
DEBUG = config("DEBUG", default=True, cast=bool)

# Development can use local HTTP origins and local-memory caching.
SESSION_COOKIE_SECURE = False
CSRF_COOKIE_SECURE = False
SECURE_SSL_REDIRECT = config(
    "SECURE_SSL_REDIRECT",
    default=False,
    cast=bool,
)
