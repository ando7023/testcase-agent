import tempfile
import unittest
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

from app.skill_runtime import SkillRegistry, SkillError
from app.skill_scripts import SkillScriptRunner


def write_script(folder, code, timeout=3):
    (folder / "scripts").mkdir(exist_ok=True)
    path = folder / "scripts/check.py"
    path.write_text(code, encoding="utf-8")
    (folder / "runtime.json").write_text(json.dumps({"scripts": {"check": {
        "path": "scripts/check.py", "description": "Inspect the supplied data", "input_source": "cases",
        "parameters": {"type": "object", "properties": {"note": {"type": "string"}}, "additionalProperties": False},
        "output_schema": {"type": "object"}, "timeout_seconds": timeout}}}), encoding="utf-8")
    return path


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

    def runner(self, code, timeout=3, trusted=True):
        path = write_script(self.folder, code, timeout)
        trust = self.root / "runtime-trust.json"
        trust.write_text(json.dumps({"custom-check:scripts/check.py": hashlib.sha256(path.read_bytes()).hexdigest()} if trusted else {}), encoding="utf-8")
        return SkillScriptRunner(SkillRegistry(self.root), self.root / "runs")

    def test_script_executes_as_separate_process_without_secret_environment(self):
        runner = self.runner('import json,sys,os\nx=json.load(sys.stdin)\nprint(json.dumps({"arguments":x["arguments"],"data":x["data"],"secret_present":"LLM_API_KEY" in os.environ,"cwd":os.path.basename(os.getcwd())}))')
        with patch.dict(os.environ, {"LLM_API_KEY": "private-never-forwarded"}):
            result = runner.execute("custom-check", "check", {"note": "$(not-a-shell-command);& 中文"}, {"cases": []})
        self.assertFalse(result["output"]["secret_present"])
        self.assertEqual(result["output"]["arguments"]["note"], "$(not-a-shell-command);& 中文")
        self.assertTrue(result["output"]["cwd"].startswith("skill-"))
        self.assertEqual(list((self.root / "runs").iterdir()), [])

    def test_untrusted_changed_scripts_and_unknown_arguments_are_rejected(self):
        runner = self.runner('print("{}")', trusted=False)
        with self.assertRaisesRegex(SkillError, "not_trusted"):
            runner.execute("custom-check", "check", {}, {})
        runner = self.runner('print("{}")')
        (self.folder / "scripts/check.py").write_text('print("CHANGED")', encoding="utf-8")
        with self.assertRaisesRegex(SkillError, "not_trusted"):
            runner.execute("custom-check", "check", {}, {})
        with self.assertRaisesRegex(SkillError, "invalid_script_arguments"):
            runner.execute("custom-check", "check", {"shell": "anything"}, {})

    def test_relative_work_directory_and_script_version_changes(self):
        runner = self.runner('print("{}")')
        original = Path.cwd()
        try:
            os.chdir(self.root)
            runner = SkillScriptRunner(runner.registry, "./relative-skill-test")
        finally:
            os.chdir(original)
        self.assertTrue(runner.work_root.is_absolute())
        self.assertEqual(runner.execute("custom-check", "check", {}, {})["output"], {})
        before = runner.registry.load("custom-check")["version"]
        (self.folder / "scripts/check.py").write_text('print("{}")\n# changed', encoding="utf-8")
        self.assertNotEqual(before, runner.registry.load("custom-check")["version"])

    def test_timeout_output_limit_and_invalid_output_have_distinct_codes(self):
        examples = [('import time\ntime.sleep(5)', "skill_script_timeout", 1),
                    ('print("x"*100000)', "skill_script_output_limit", 3),
                    ('print("not-json")', "skill_script_invalid_json", 3),
                    ('print("["*2000+"0"+"]"*2000)', "skill_script_invalid_json", 3),
                    ('print("[]")', "skill_script_invalid_output", 3),
                    ('import sys\nprint("PRIVATE_STDERR",file=sys.stderr)\nsys.exit(2)', "skill_script_failed", 3)]
        for code, error, timeout in examples:
            with self.subTest(error=error):
                runner = self.runner(code, timeout)
                with self.assertRaises(SkillError) as caught:
                    runner.execute("custom-check", "check", {}, {})
                self.assertEqual(caught.exception.code, error)
                self.assertNotIn("PRIVATE_STDERR", str(caught.exception))

    def test_schema_cannot_fetch_remote_references_and_traversal_is_rejected(self):
        self.runner('print("{}")')
        manifest = self.folder / "runtime.json"
        data = json.loads(manifest.read_text())
        data["scripts"]["check"]["parameters"]["properties"]["note"] = {"$ref": "https://example.invalid/schema"}
        manifest.write_text(json.dumps(data))
        with self.assertRaisesRegex(SkillError, "references_not_supported"):
            SkillRegistry(self.root).load("custom-check")
        data["scripts"]["check"]["parameters"]["properties"] = {}
        data["scripts"]["check"]["path"] = "scripts/../../other.py"
        manifest.write_text(json.dumps(data))
        with self.assertRaises(SkillError):
            SkillRegistry(self.root).load("custom-check")

    def test_yaml_description_delimiter_does_not_split_body(self):
        path = self.folder / "SKILL.md"
        path.write_text('---\nname: custom-check\ndescription: "Check --- cases"\n---\nCOMPLETE_BODY', encoding="utf-8")
        self.assertEqual(SkillRegistry(self.root).load("custom-check")["instruction"], "COMPLETE_BODY")

    def test_hyphenated_package_preserves_existing_skill_id(self):
        folder = self.root / "state-transition"
        folder.mkdir()
        (folder / "SKILL.md").write_text('---\nname: state-transition\ndescription: Test state changes\nmetadata:\n  legacy_id: state_transition\n  case_types: [state_transition]\n---\nNATIVE_STRATEGY', encoding="utf-8")
        registry = SkillRegistry(self.root)
        self.assertEqual(registry.load("state_transition")["instruction"], "NATIVE_STRATEGY")
        self.assertEqual(registry.resolve(["state_transition"])[0].name, "state_transition")


if __name__ == "__main__":
    unittest.main()
