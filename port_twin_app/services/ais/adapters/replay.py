"""Read AIS NMEA sentences from a local replay file."""

from pathlib import Path

from port_twin_app.services.ais.adapters.base import AISSourceAdapter


class ReplayAISAdapter(AISSourceAdapter):
    """Read newline-delimited NMEA sentences in batches."""

    def __init__(self, file_path, batch_size=100):
        self.file_path = Path(file_path)
        self.batch_size = batch_size

        if batch_size <= 0:
            raise ValueError("AIS replay batch size must be positive.")

        if not self.file_path.is_file():
            raise FileNotFoundError(f"AIS replay file does not exist: {self.file_path}")

    def iter_batches(self):
        batch = []

        with self.file_path.open("rb") as replay_file:
            for line in replay_file:
                sentence = line.strip()

                if not sentence:
                    continue

                batch.append(sentence)

                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []

        if batch:
            yield batch
