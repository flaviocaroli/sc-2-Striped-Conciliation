from __future__ import annotations

import json
import unittest
from pathlib import Path


class CountThinningProtocolContractTests(unittest.TestCase):
    def test_validation_panel_count_key_matches_materializer(self) -> None:
        repo = Path(__file__).resolve().parents[1]

        protocol_path = (
            repo
            / "configs"
            / "extension_2026"
            / "count_thinning_protocol_v1.json"
        )

        materializer_path = (
            repo
            / "scripts"
            / "data"
            / "materialize_count_thinning_panels.py"
        )

        protocol = json.loads(
            protocol_path.read_text(
                encoding="utf-8"
            )
        )

        thinning = protocol["thinning"]

        self.assertEqual(
            thinning["expected_validation_panel_count"],
            15,
        )

        self.assertNotIn(
            "expected_validation_panels",
            thinning,
        )

        source = materializer_path.read_text(
            encoding="utf-8"
        )

        good = (
            'protocol["thinning"]'
            '["expected_validation_panel_count"]'
        )

        bad = (
            'protocol["thinning"]'
            '["expected_validation_panels"]'
        )

        self.assertEqual(
            source.count(good),
            1,
        )

        self.assertEqual(
            source.count(bad),
            0,
        )


if __name__ == "__main__":
    unittest.main()
