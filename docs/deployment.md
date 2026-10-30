# MSA Port Twin — Deployment Guide

This guide describes the production configuration and deployment requirements for the MSA Port Twin Django API.

## 1. Deployment Architecture

The production backend uses:

- **Django:** API application.
- **Gunicorn:** WSGI application server.
- **PostgreSQL + PostGIS:** Persistent database and geospatial data storage.
- **Redis:** Production cache and API throttling support.
- **WhiteNoise:** Static file serving.
- **Huawei ModelArts (optional):** Remote marine AI inference.

The application supports either local FNO inference or a remotely hosted ModelArts endpoint.

## 2. System Requirements

The deployment environment must provide:

- Python compatible with the pinned project dependencies.
- PostgreSQL with PostGIS enabled.
- GDAL, GEOS, and PROJ system libraries required by GeoDjango.
- A reachable Redis instance.
- HTTPS termination through a reverse proxy or hosting platform.

A live cloud deployment has not yet been performed.

## 3. Python Dependencies

Install production dependencies:

```bash
pip install -r requirements.txt
```

Development and local AI dependencies:

```bash
pip install -r requirements-dev.txt
```

The production requirements intentionally exclude heavyweight local AI dependencies.

## 4. Environment Configuration

Production settings use:

```bash
DJANGO_SETTINGS_MODULE=port_twin_proj.settings_prod
```

Configure these environment variables:

| Variable | Required | Description |
|---|---|---|
| MODE | Yes | `prod` |
| DEBUG | Yes | `False` |
| SECRET_KEY | Yes | Unique secret of at least 50 characters |
| DATABASE_URL | Yes | PostgreSQL/PostGIS connection URL |
| ALLOWED_HOSTS | Yes | Comma-separated production hostnames |
| CORS_ALLOWED_ORIGINS | Yes | Comma-separated HTTPS frontend origins |
| CSRF_TRUSTED_ORIGINS | Yes | Comma-separated HTTPS trusted origins |
| REDIS_URL | Yes | Redis connection URL |
| MARINE_AI_BACKEND | Yes | `local` or `modelarts` |
| MARINE_AI_MODELARTS_ENDPOINT | ModelArts only | Remote inference endpoint |
| MARINE_AI_MODELARTS_TOKEN | ModelArts only | Authentication token |
| MARINE_AI_MODELARTS_TIMEOUT | No | Inference timeout; defaults to 30 seconds |
| MARINE_AI_MODELARTS_VERSION | No | Model version identifier |
| MARINE_AI_CHECKPOINT | Local AI only | Local model checkpoint path |
| SECURE_SSL_REDIRECT | No | Defaults to `True` in production |
| SECURE_HSTS_SECONDS | No | HSTS duration; defaults to 0 |
| SECURE_HSTS_INCLUDE_SUBDOMAINS | No | Defaults to `False` |
| SECURE_HSTS_PRELOAD | No | Defaults to `False` |

Example configuration (replace placeholders with actual deployment values):

```dotenv
MODE=prod
DEBUG=False
SECRET_KEY=replace-with-a-unique-secret-of-at-least-50-characters
DATABASE_URL=postgis://username:password@database-host:5432/database-name
ALLOWED_HOSTS=api.example.com
CORS_ALLOWED_ORIGINS=https://app.example.com
CSRF_TRUSTED_ORIGINS=https://app.example.com
REDIS_URL=redis://redis-host:6379/1

MARINE_AI_BACKEND=modelarts
MARINE_AI_MODELARTS_ENDPOINT=https://modelarts-endpoint.example.com
MARINE_AI_MODELARTS_TOKEN=replace-with-modelarts-token
MARINE_AI_MODELARTS_TIMEOUT=30
MARINE_AI_MODELARTS_VERSION=modelarts-unknown
```

Never commit `.env` files, credentials, tokens, or production secrets.

## 5. Database Setup

Create a PostgreSQL database and enable PostGIS:

```sql
CREATE EXTENSION IF NOT EXISTS postgis;
```

Apply migrations:

```bash
python manage.py migrate --settings=port_twin_proj.settings_prod
```

## 6. Static Files

Collect static assets before starting the application:

```bash
python manage.py collectstatic --noinput --settings=port_twin_proj.settings_prod
```

WhiteNoise serves collected assets from `staticfiles/`.

## 7. Production Checks

Run:

```bash
python manage.py check --deploy --settings=port_twin_proj.settings_prod
```

Review security warnings and configure HTTPS settings according to the hosting environment.

## 8. Starting Gunicorn

Set:

```bash
DJANGO_SETTINGS_MODULE=port_twin_proj.settings_prod
```

Start Gunicorn:

```bash
gunicorn port_twin_proj.wsgi:application \
  --bind 0.0.0.0:8000 \
  --workers 3 \
  --timeout 120
```

Adjust worker count and timeout according to available compute resources.

## 9. AI Backend Configuration

### ModelArts

Configure:

```dotenv
MARINE_AI_BACKEND=modelarts
MARINE_AI_MODELARTS_ENDPOINT=https://your-modelarts-endpoint
MARINE_AI_MODELARTS_TOKEN=your-token
```

The production dependency set excludes the local PyTorch stack. Selecting ModelArts avoids importing PyTorch during application startup.

### Local FNO

Configure:

```dotenv
MARINE_AI_BACKEND=local
MARINE_AI_CHECKPOINT=/path/to/marine_fno.pt
```

The local backend requires the AI dependencies and a valid trained checkpoint.

## 10. Current Cloud Deployment Status

Production configuration and local deployment-readiness checks have been completed.

Actual Huawei Cloud provisioning and deployment remain pending the availability of a genuinely free compute and managed-service option. No paid cloud resources should be provisioned without explicit budget approval.

Potential Huawei Cloud services include:

- ECS for hosting Django.
- RDS for PostgreSQL/PostGIS, subject to extension compatibility.
- DCS for Redis.
- ModelArts for remote AI inference.

Final service selection depends on availability, regional support, PostGIS compatibility, and confirmed cost.