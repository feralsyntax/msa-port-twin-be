import os

from celery import Celery

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "port_twin_proj.settings_dev",
)

app = Celery("port_twin_proj")

app.config_from_object("django.conf:settings", namespace="CELERY")

app.autodiscover_tasks()
