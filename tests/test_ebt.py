import csv
import tempfile
import unittest
from pathlib import Path

from app.ebt_dataset import EBT_DATASET_ID, EBT_FILES, EBTRepository


class EBTRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for name in EBT_FILES:
            (self.root / name).write_text("", encoding="utf-8")
        self._write_csv(
            "artifacts.csv",
            ["id", "content", "layer", "summary"],
            [
                ["100", "A subscriber shall register.", "Requirement", ""],
                ["141", "Register a valid subscriber.", "Test", ""],
                ["142", "Reject a duplicate subscriber.", "Test", ""],
                ["900", "implementation", "Code", ""],
            ],
        )
        self._write_csv(
            "traces.csv",
            ["s_id", "t_id", "label"],
            [["141", "100", "1"], ["142", "100", "1"], ["900", "141", "1"]],
        )
        self.repository = EBTRepository(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def _write_csv(self, name, header, rows):
        with (self.root / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)

    def test_builds_requirement_to_test_retrieval_benchmark(self):
        dataset, documents = self.repository.build_retrieval_dataset()

        self.assertEqual(dataset.id, EBT_DATASET_ID)
        self.assertEqual(len(dataset.cases), 1)
        self.assertEqual(
            dataset.cases[0].expected_document_ids,
            ["EBT-TEST-141", "EBT-TEST-142"],
        )
        self.assertEqual({item.id for item in documents}, {"EBT-TEST-141", "EBT-TEST-142"})
        self.assertTrue(all(item.metadata["ticket_type"] == "COMMON" for item in documents))

    def test_status_reports_only_requirement_test_traces(self):
        status = self.repository.status()

        self.assertTrue(status["ready"])
        self.assertEqual(status["artifact_count"], 4)
        self.assertEqual(status["positive_trace_count"], 2)
        self.assertEqual(status["retrieval_case_count"], 1)


if __name__ == "__main__":
    unittest.main()
