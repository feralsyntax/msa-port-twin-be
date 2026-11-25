from io import StringIO

from django.core.management import call_command
from django.test import TestCase

from port_twin_app.models import TerminalEDIMessage, TerminalOperationalEvent


class TerminalEDIGeneratorTests(TestCase):
    def test_generates_all_three_message_types_by_default(self):
        call_command(
            "generate_terminal_edi",
            stdout=StringIO(),
        )

        self.assertEqual(TerminalEDIMessage.objects.count(), 3)
        self.assertEqual(TerminalOperationalEvent.objects.count(), 3)

        self.assertSetEqual(
            set(
                TerminalEDIMessage.objects.values_list(
                    "message_type",
                    flat=True,
                )
            ),
            {"COPARN", "CODECO", "COARRI"},
        )

    def test_generated_messages_are_marked_synthetic(self):
        call_command(
            "generate_terminal_edi",
            stdout=StringIO(),
        )

        self.assertEqual(
            TerminalEDIMessage.objects.filter(is_synthetic=True).count(),
            3,
        )

    def test_generates_requested_number_per_message_type(self):
        call_command(
            "generate_terminal_edi",
            "--count",
            "2",
            stdout=StringIO(),
        )

        self.assertEqual(TerminalEDIMessage.objects.count(), 6)
        self.assertEqual(TerminalOperationalEvent.objects.count(), 6)

    def test_can_generate_only_selected_message_types(self):
        call_command(
            "generate_terminal_edi",
            "--message-types",
            "CODECO",
            stdout=StringIO(),
        )

        self.assertEqual(TerminalEDIMessage.objects.count(), 1)
        self.assertEqual(
            TerminalEDIMessage.objects.get().message_type,
            "CODECO",
        )

    def test_uses_custom_source(self):
        call_command(
            "generate_terminal_edi",
            "--source",
            "demo-terminal-02",
            stdout=StringIO(),
        )

        self.assertEqual(
            TerminalEDIMessage.objects.get().source,
            "demo-terminal-02",
        )
