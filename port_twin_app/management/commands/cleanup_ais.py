"""Run AIS data retention cleanup."""

from django.core.management.base import BaseCommand, CommandError

from port_twin_app.services.ais.retention import AISRetentionService


class Command(BaseCommand):
    help = "Delete expired AIS positions and message receipts."

    def handle(self, *args, **options):
        try:
            result = AISRetentionService().cleanup()
        except ValueError as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(
            self.style.SUCCESS(
                "AIS retention cleanup complete: "
                f"positions_deleted={result['positions_deleted']}, "
                f"receipts_deleted={result['receipts_deleted']}, "
                f"cutoff={result['cutoff'].isoformat()}"
            )
        )
