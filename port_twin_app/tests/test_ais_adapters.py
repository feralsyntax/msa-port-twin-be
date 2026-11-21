from pathlib import Path
from tempfile import TemporaryDirectory

from django.test import SimpleTestCase

from port_twin_app.services.ais.adapters.replay import ReplayAISAdapter


class ReplayAISAdapterTests(SimpleTestCase):
    def test_reads_sentences_in_batches(self):
        with TemporaryDirectory() as directory:
            replay_file = Path(directory) / "ais.txt"
            replay_file.write_bytes(b"!AIVDM,first\n!AIVDM,second\n\n!AIVDM,third\n")

            adapter = ReplayAISAdapter(replay_file, batch_size=2)
            batches = list(adapter.iter_batches())

        self.assertEqual(
            batches,
            [
                [b"!AIVDM,first", b"!AIVDM,second"],
                [b"!AIVDM,third"],
            ],
        )

    def test_rejects_missing_file(self):
        with TemporaryDirectory() as directory:
            missing_file = Path(directory) / "missing.txt"

            with self.assertRaises(FileNotFoundError):
                ReplayAISAdapter(missing_file)

    def test_rejects_invalid_batch_size(self):
        with TemporaryDirectory() as directory:
            replay_file = Path(directory) / "ais.txt"
            replay_file.write_text("!AIVDM,test\n")

            with self.assertRaises(ValueError):
                ReplayAISAdapter(replay_file, batch_size=0)

    def test_empty_file_yields_no_batches(self):
        with TemporaryDirectory() as directory:
            replay_file = Path(directory) / "ais.txt"
            replay_file.touch()

            adapter = ReplayAISAdapter(replay_file)

            self.assertEqual(list(adapter.iter_batches()), [])
