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

The application can use either the local FNO inference backend or a remotely hosted ModelArts endpoint.

## 2. System Requirements

The deployment environment must provide:

- Python compatible with the pinned project dependencies.
- PostgreSQL with PostGIS enabled.
- GDAL, GEOS, and PROJ system libraries required by GeoDjango.
- A reachable Redis instance.
- HTTPS termination through a reverse proxy or hosting platform.

The application has been validated locally with Django's production settings and Gunicorn. A live cloud deployment has not yet been performed.

## 3. Python Dependencies

Install production dependencies:

```bash
pip install -r requirements.txt