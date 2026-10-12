from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Populate the database with demo data for MSA Port Twin"

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("Demo data command is ready."))
