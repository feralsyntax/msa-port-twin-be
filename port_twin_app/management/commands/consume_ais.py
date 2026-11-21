"""Run the AIS decoding and ingestion pipeline."""

import logging
import time

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from port_twin_app.services.ais.adapters.replay import ReplayAISAdapter
from port_twin_app.services.ais.decoder import AISDecoder
from port_twin_app.services.ais.ingestion import AISIngestionService

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Consume AIS messages from the configured source."

    def add_arguments(self, parser):
        parser.add_argument(
            "--file",
            required=True,
            help="Path to a newline-delimited AIS NMEA replay file.",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=100,
            help="Number of NMEA sentences to process per batch.",
        )
        parser.add_argument(
            "--loop",
            action="store_true",
            help="Replay the file continuously.",
        )
        parser.add_argument(
            "--interval",
            type=float,
            default=5,
            help="Seconds to wait between replay cycles when looping.",
        )

    def handle(self, *args, **options):
        source = settings.AIS_SOURCE

        if source != "replay":
            raise CommandError(
                f"Unsupported AIS source '{source}'. "
                "No adapter has been registered for this source."
            )

        if options["interval"] < 0:
            raise CommandError("--interval cannot be negative.")

        try:
            adapter = ReplayAISAdapter(
                file_path=options["file"],
                batch_size=options["batch_size"],
            )
        except (ValueError, FileNotFoundError) as exc:
            raise CommandError(str(exc)) from exc

        decoder = AISDecoder()
        ingestion = AISIngestionService(source=source)

        self.stdout.write(
            f"Starting AIS consumer: source={source}, file={adapter.file_path}"
        )

        try:
            while True:
                batch_count = 0
                totals = {
                    "received": 0,
                    "duplicates": 0,
                    "vessels_created": 0,
                    "positions_created": 0,
                }

                for sentences in adapter.iter_batches():
                    batch_count += 1
                    messages = decoder.decode_sentences(sentences)

                    if not messages:
                        continue

                    result = ingestion.ingest(messages)

                    for key in totals:
                        totals[key] += result[key]

                self.stdout.write(
                    self.style.SUCCESS(
                        f"Replay complete: batches={batch_count}, "
                        f"received={totals['received']}, "
                        f"duplicates={totals['duplicates']}, "
                        f"vessels_created={totals['vessels_created']}, "
                        f"positions_created={totals['positions_created']}"
                    )
                )

                if not options["loop"]:
                    break

                time.sleep(options["interval"])

        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING("AIS consumer stopped."))
