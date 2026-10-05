import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.skill_runtime import SkillRegistry, SkillError


class SkillPackageTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / "custom-check"
        (self.folder / "references").mkdir(parents=True)
        (self.folder / "SKILL.md").write_text(
            '---\nname: custom-check\ndescription: Audit generated cases\nmetadata:\n  case_types: [functional]\n---\n'
            'PRIVATE_BODY_ONLY_WHEN_SELECTED\nRead references/details.md when needed.', encoding="utf-8")
        (self.folder / "references/details.md").write_text("PRIVATE_REFERENCE", encoding="utf-8")

    def test_discovery_does_not_load_body_or_resources(self):
        with patch.object(SkillRegistry, "_text", side_effect=AssertionError("eager resource read")):
            registry = SkillRegistry(self.root)
            entry = next(e for e in registry.catalog() if e["id"] == "custom-check")
        self.assertNotIn("instruction", entry)
        loaded = registry.load("custom-check")
        self.assertIn("PRIVATE_BODY", loaded["instruction"])
        self.assertNotIn("PRIVATE_REFERENCE", str(loaded))
        self.assertEqual(loaded["resources"], ["references/details.md"])
        self.assertEqual(registry.read_resource("custom-check", "references/details.md")["content"], "PRIVATE_REFERENCE")

    def test_unknown_traversal_and_scripts_cannot_be_read_as_resources(self):
        registry = SkillRegistry(self.root)
        for path in ("../other", "references/../../secret", "C:/secret", "references\\secret", "scripts/check.py"):
            with self.assertRaises(SkillError):
                registry.read_resource("custom-check", path)
        with self.assertRaises(SkillError):
            registry.load("invented")

    def test_invalid_package_is_reported_and_legacy_remains_available(self):
        bad = self.root / "bad"
        bad.mkdir()
        (bad / "SKILL.md").write_text("---\nname: bad\n---\nbody", encoding="utf-8")
        registry = SkillRegistry(self.root)
        self.assertEqual(registry.errors, [{"folder": "bad", "code": "invalid_skill_package"}])
        self.assertTrue(registry.load("permission")["instruction"])

    def test_version_changes_and_large_resources_are_rejected(self):
        registry = SkillRegistry(self.root)
        old = registry.load("custom-check")["version"]
        path = self.folder / "SKILL.md"
        path.write_text(path.read_text(encoding="utf-8") + "\nupdated", encoding="utf-8")
        self.assertNotEqual(old, registry.load("custom-check")["version"])
        (self.folder / "references/details.md").write_text("x" * 24001, encoding="utf-8")
        with self.assertRaises(SkillError):
            registry.read_resource("custom-check", "references/details.md")


if __name__ == "__main__":
    unittest.main()
