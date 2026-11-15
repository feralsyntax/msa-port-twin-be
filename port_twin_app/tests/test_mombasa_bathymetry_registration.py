from django.test import TestCase

from port_twin_app.models import BathymetryDataset
from port_twin_app.services.marine_data.registration import (
    register_all_mombasa_bathymetry,
    register_mombasa_bathymetry,
)


class MombasaBathymetryRegistrationTests(TestCase):
    def test_registers_both_prepared_zones(self):
        results = register_all_mombasa_bathymetry()

        self.assertEqual(len(results), 2)
        self.assertTrue(all(created for _, created in results))
        self.assertEqual(BathymetryDataset.objects.count(), 2)

        zones = set(BathymetryDataset.objects.values_list("zone", flat=True))
        self.assertEqual(zones, {"approaches", "port_operations"})

    def test_registration_is_idempotent(self):
        first, created = register_mombasa_bathymetry("port_operations")
        second, created_again = register_mombasa_bathymetry("port_operations")

        self.assertTrue(created)
        self.assertFalse(created_again)
        self.assertEqual(first.pk, second.pk)
        self.assertEqual(BathymetryDataset.objects.count(), 1)

    def test_registers_provisional_dataset_metadata(self):
        record, _ = register_mombasa_bathymetry("port_operations")

        self.assertEqual(
            record.quality_status,
            BathymetryDataset.QualityStatus.PROVISIONAL,
        )
        self.assertEqual(record.geographic_crs, "EPSG:4326")
        self.assertEqual(record.projected_crs, "EPSG:32737")
        self.assertEqual(record.target_resolution_m, 100)

        # The source grid is approximately 15 arc-seconds, not 100 metres.
        self.assertAlmostEqual(
            float(record.source_longitude_spacing_arc_seconds),
            15.0,
            places=3,
        )
        self.assertEqual(record.grid_width, 29)
        self.assertEqual(record.grid_height, 17)
        self.assertEqual(
            set(record.variables),
            {"elevation_m", "tid", "bathymetry_m"},
        )

    def test_footprint_uses_actual_dataset_bounds(self):
        record, _ = register_mombasa_bathymetry("port_operations")

        west, south, east, north = record.footprint.extent

        self.assertAlmostEqual(west, 39.5979166667, places=6)
        self.assertAlmostEqual(south, -4.1020833333, places=6)
        self.assertAlmostEqual(east, 39.7145833333, places=6)
        self.assertAlmostEqual(north, -4.0354166667, places=6)

    def test_paths_are_repository_relative(self):
        record, _ = register_mombasa_bathymetry("approaches")

        self.assertFalse(record.data_file_path.startswith("/"))
        self.assertFalse(record.provenance_file_path.startswith("/"))
        self.assertTrue(record.data_file_path.endswith(".nc"))
        self.assertTrue(record.provenance_file_path.endswith(".provenance.json"))
