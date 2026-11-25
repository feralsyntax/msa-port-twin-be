"""Generate clearly labelled synthetic terminal EDI messages."""

import uuid

from django.core.management.base import BaseCommand, CommandError

from port_twin_app.services.terminal_edi.ingestion import (
    TerminalEDIIngestionService,
)


class Command(BaseCommand):
    help = "Generate synthetic COPARN, CODECO and COARRI messages."

    MESSAGE_TYPES = {
        "COPARN": "11",
        "CODECO": "34",
        "COARRI": "98",
    }

    def add_arguments(self, parser):
        parser.add_argument(
            "--count",
            type=int,
            default=1,
            help="Number of messages to generate per selected message type.",
        )
        parser.add_argument(
            "--message-types",
            nargs="+",
            choices=list(self.MESSAGE_TYPES),
            default=list(self.MESSAGE_TYPES),
            help="EDI message types to generate. Defaults to all three.",
        )
        parser.add_argument(
            "--source",
            default="synthetic-terminal",
            help="Source identifier for the generated messages.",
        )

    def handle(self, *args, **options):
        count = options["count"]

        if count < 1:
            raise CommandError("--count must be at least 1.")

        try:
            ingestion = TerminalEDIIngestionService(
                source=options["source"],
            )
        except ValueError as exc:
            raise CommandError(str(exc)) from exc

        totals = {
            "generated": 0,
            "processed": 0,
            "duplicates": 0,
            "failed": 0,
        }

        for message_type in options["message_types"]:
            for _ in range(count):
                raw_message = self._build_message(message_type)
                totals["generated"] += 1

                try:
                    result = ingestion.ingest(
                        raw_message,
                        is_synthetic=True,
                    )
                except Exception as exc:
                    totals["failed"] += 1
                    self.stderr.write(
                        self.style.ERROR(f"{message_type} ingestion failed: {exc}")
                    )
                    continue

                if result["duplicate"]:
                    totals["duplicates"] += 1
                elif result["status"] == "processed":
                    totals["processed"] += 1

        self.stdout.write(
            self.style.SUCCESS(
                "Synthetic terminal EDI generation complete: "
                f"generated={totals['generated']}, "
                f"processed={totals['processed']}, "
                f"duplicates={totals['duplicates']}, "
                f"failed={totals['failed']}"
            )
        )

    @classmethod
    def _build_message(cls, message_type):
        control_reference = uuid.uuid4().hex[:8].upper()
        message_reference = f"SYN-{control_reference}"
        container_reference = f"SYNTH{control_reference}"

        document_code = cls.MESSAGE_TYPES[message_type]

        return (
            f"UNH+{control_reference}+{message_type}:D:95B:UN'"
            f"BGM+{document_code}+{message_reference}+9'"
            f"EQD+CN+{container_reference}'"
            f"UNT+4+{control_reference}'"
        )
