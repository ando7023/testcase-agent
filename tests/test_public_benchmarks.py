import csv
import tempfile
import unittest
from pathlib import Path

from app.ebt_dataset import EBT_FILES, EBTRepository
from app.orchestrator import TestCaseOrchestrator
from app.public_benchmarks import (
    PublicBenchmarkService,
    SRSRepository,
    STORYSEEK_FILES,
    StorySeekRepository,
)
from app.store import JsonStore


class PublicBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _write_csv(path, header, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)

    def _ebt(self):
        root = self.root / "external" / "ebt"
        root.mkdir(parents=True, exist_ok=True)
        for name in EBT_FILES:
            (root / name).write_text("", encoding="utf-8")
        self._write_csv(
            root / "artifacts.csv",
            ["id", "content", "layer", "summary"],
            [
                ["100", "A registered user shall submit a request and receive a visible result.", "Requirement", ""],
                ["141", "Test case: submit request. Preconditions user registered. Steps submit. Postconditions result is visible.", "Test", ""],
            ],
        )
        self._write_csv(
            root / "traces.csv",
            ["s_id", "t_id", "label"],
            [["141", "100", "1"]],
        )
        return EBTRepository(root)

    def test_storyseek_uses_project_level_splits(self):
        repository = StorySeekRepository(self.root / "storyseek")
        header = [
            "storyid", "background", "problems", "solutions", "goal",
            "actor", "impact", "deliverable", "us_actor", "us_action",
            "us_expected_outcome",
        ]
        for index, name in enumerate(STORYSEEK_FILES, 1):
            self._write_csv(
                repository.root / name,
                header,
                [[index, "Background", "Problem", "Solution", "Goal", "User", "Impact", "Feature", "User", "submit ticket", "ticket is created"]],
            )

        self.assertTrue(repository.ready)
        self.assertEqual(len(repository.load("development")), 6)
        self.assertEqual(len(repository.load("validation")), 2)
        self.assertEqual(len(repository.load("test")), 2)
        self.assertEqual(repository.status()["license"], "MIT")

    def test_srs_discovers_original_and_normalized_pairs(self):
        repository = SRSRepository(self.root / "srs")
        folder = repository.files_root / "project-a"
        folder.mkdir(parents=True)
        (folder / "spec.md").write_text("# Requirement\nUser submits a request.", encoding="utf-8")
        (folder / "spec_Raw.txt").write_text("Requirement User submits a request.", encoding="utf-8")
        (folder / "FunctionalRequirements.txt").write_text("User submits a request.", encoding="utf-8")

        pairs = repository.pairs()

        self.assertEqual(len(pairs), 1)
        self.assertTrue(pairs[0]["source_path"].endswith("spec.md"))
        self.assertTrue(pairs[0]["relevant_path"].endswith("FunctionalRequirements.txt"))

    def test_critic_and_ebt_generation_suites_persist_reports(self):
        ebt = self._ebt()
        service = PublicBenchmarkService(self.root, ebt)
        factory = lambda root: TestCaseOrchestrator(JsonStore(root))

        critic = service.run("critic_mutation", 4, "test", "offline", factory)
        generation = service.run("ebt_generation", 1, "test", "offline", factory)

        self.assertEqual(critic["metrics"]["defect_detection_recall"], 1.0)
        self.assertEqual(critic["metrics"]["clean_false_positive_count"], 0.0)
        self.assertEqual(generation["sample_count"], 1)
        self.assertEqual(generation["metrics"]["task_success_rate"], 1.0)
        self.assertEqual(len(service.list_reports()), 2)


if __name__ == "__main__":
    unittest.main()
