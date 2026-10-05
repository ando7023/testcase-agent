"""Progressively disclosed local skill packages; executable resources opt in separately."""
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath

import yaml

from .skills import SKILLS, TestSkill


class SkillError(ValueError):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


class SkillRegistry:
    MAX_HEADER = 8192
    MAX_TEXT = 24000

    def __init__(self, root=None):
        configured = root or os.getenv("CASEFORGE_SKILLS_DIR") or Path(__file__).resolve().parents[1] / "skills" / "testcase"
        self.root = Path(os.path.abspath(str(configured))).resolve()
        self.entries = {}
        self.errors = []
        for name, skill in SKILLS.items():
            self.entries[name] = {"id": name, "name": name, "description": skill.instruction,
                                  "path": "legacy:" + name, "source": "legacy",
                                  "case_types": skill.case_types, "triggers": skill.triggers}
        if self.root.is_dir():
            for folder in sorted(self.root.iterdir()):
                if not folder.is_dir():
                    continue
                try:
                    entry = self._metadata(folder.name)
                    if entry["id"] in self.entries and self.entries[entry["id"]]["source"] != "legacy":
                        raise SkillError("duplicate_skill_id")
                    self.entries[entry["id"]] = entry
                except (OSError, ValueError, yaml.YAMLError):
                    self.errors.append({"folder": folder.name, "code": "invalid_skill_package"})

    def _path(self, folder, relative):
        # Reject traversal, Windows drive/UNC paths, alternate streams and symlinks.
        if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
            raise SkillError("invalid_resource_path")
        parts = PurePosixPath(relative).parts
        if PurePosixPath(relative).is_absolute() or any(p in {"..", "."} for p in relative.split("/")):
            raise SkillError("invalid_resource_path")
        base = self.root / folder
        target = base.joinpath(*parts)
        current = self.root
        for part in (folder,) + parts:
            current = current / part
            if current.is_symlink():
                raise SkillError("resource_symlink_rejected")
        try:
            target.resolve().relative_to(base.resolve())
            base.resolve().relative_to(self.root)
        except ValueError:
            raise SkillError("resource_outside_skill")
        return target

    def _metadata(self, folder):
        path = self._path(folder, "SKILL.md")
        lines, size = [], 0
        with path.open(encoding="utf-8") as handle:
            if handle.readline(self.MAX_HEADER + 1).strip() != "---":
                raise SkillError("skill_frontmatter_required")
            while True:
                line = handle.readline(self.MAX_HEADER + 1)
                if not line:
                    raise SkillError("skill_frontmatter_required")
                size += len(line.encode("utf-8"))
                if size > self.MAX_HEADER:
                    raise SkillError("skill_header_too_large")
                if line.strip() == "---":
                    break
                lines.append(line)
        data = yaml.safe_load("".join(lines))
        if not isinstance(data, dict):
            raise SkillError("invalid_skill_metadata")
        name, description = data.get("name"), data.get("description")
        if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", name):
            raise SkillError("invalid_skill_name")
        if not isinstance(description, str) or not description.strip() or len(description) > 1200:
            raise SkillError("invalid_skill_description")
        meta = data.get("metadata") or {}
        if not isinstance(meta, dict):
            raise SkillError("invalid_skill_metadata")
        legacy_id = meta.get("legacy_id")
        if legacy_id is not None and (not isinstance(legacy_id, str) or legacy_id not in SKILLS or name != legacy_id.replace("_", "-")):
            raise SkillError("invalid_skill_legacy_alias")
        for key in ("case_types", "triggers"):
            value = meta.get(key, [])
            if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
                raise SkillError("invalid_skill_metadata")
        return {"id": legacy_id or name, "name": name, "description": description.strip(),
                "path": folder + "/SKILL.md", "folder": folder, "source": "package",
                "case_types": meta.get("case_types", []), "triggers": meta.get("triggers", [])}

    def catalog(self):
        return [{k: v for k, v in e.items() if k != "folder"} for e in self.entries.values()]

    def require(self, names):
        if any(name not in self.entries for name in names):
            raise SkillError("unknown_skill")
        return [self.entries[name] for name in dict.fromkeys(names)]

    def select(self, text):
        selected = [e["id"] for e in self.entries.values() if any(t in text for t in e["triggers"])]
        return selected or ["happy_path", "boundary", "exception_recovery"]

    @classmethod
    def _text(cls, path):
        with path.open("rb") as handle:
            raw = handle.read(cls.MAX_TEXT + 1)
        if len(raw) > cls.MAX_TEXT:
            raise SkillError("skill_resource_too_large")
        try:
            return raw.decode("utf-8"), hashlib.sha256(raw).hexdigest()
        except UnicodeError:
            raise SkillError("skill_resource_not_utf8")

    def load(self, name):
        entry = self.require([name])[0]
        if entry["source"] == "legacy":
            body = SKILLS[name].instruction
            return dict(entry, instruction=body, version=hashlib.sha256(body.encode()).hexdigest(),
                        resources=[], scripts={})
        if self._metadata(entry["folder"])["id"] != name:
            raise SkillError("skill_metadata_changed_restart_required")
        text, digest = self._text(self._path(entry["folder"], "SKILL.md"))
        lines = text.splitlines(keepends=True)
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
        body = "".join(lines[end + 1:]).strip()
        if not body:
            raise SkillError("empty_skill_body")
        resources = []
        for category in ("references", "assets"):
            directory = self.root / entry["folder"] / category
            if directory.is_dir():
                for path in sorted(directory.rglob("*")):
                    if path.is_file():
                        relative = path.relative_to(self.root / entry["folder"]).as_posix()
                        self._path(entry["folder"], relative)
                        resources.append(relative)
        scripts = {}
        manifest = self._path(entry["folder"], "runtime.json")
        if manifest.exists():
            from .skill_scripts import check_schema
            manifest_text, manifest_digest = self._text(manifest)
            try:
                definitions = json.loads(manifest_text)
            except ValueError:
                raise SkillError("invalid_skill_manifest")
            if not isinstance(definitions, dict) or set(definitions) != {"scripts"} or not isinstance(definitions["scripts"], dict):
                raise SkillError("invalid_skill_manifest")
            for key, spec in definitions["scripts"].items():
                if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", key) or not isinstance(spec, dict):
                    raise SkillError("invalid_skill_manifest")
                if set(spec) != {"path", "description", "input_source", "parameters", "output_schema", "timeout_seconds"}:
                    raise SkillError("invalid_skill_manifest")
                if not isinstance(spec["path"], str) or not spec["path"].startswith("scripts/") or not spec["path"].endswith(".py"):
                    raise SkillError("invalid_skill_script_path")
                self._path(entry["folder"], spec["path"])
                if not isinstance(spec["input_source"], str) or spec["input_source"] not in {"analysis", "modules", "cases"}:
                    raise SkillError("invalid_skill_input_source")
                if not isinstance(spec["description"], str) or not 1 <= len(spec["description"]) <= 1000:
                    raise SkillError("invalid_skill_script_description")
                if type(spec["timeout_seconds"]) is not int or not 1 <= spec["timeout_seconds"] <= 30:
                    raise SkillError("invalid_skill_script_timeout")
                for field in ("parameters", "output_schema"):
                    check_schema(spec[field])
                    if spec[field].get("type") != "object":
                        raise SkillError("script_schema_must_be_object")
                if spec["parameters"].get("additionalProperties") is not False:
                    raise SkillError("script_parameters_must_reject_unknown_fields")
                script_path = self._path(entry["folder"], spec["path"])
                with script_path.open("rb") as handle:
                    script_bytes = handle.read(self.MAX_TEXT + 1)
                if len(script_bytes) > self.MAX_TEXT:
                    raise SkillError("skill_script_too_large")
                script_version = hashlib.sha256(script_bytes).hexdigest()
                scripts[key] = dict(spec, version=script_version)
                digest = hashlib.sha256((digest + script_version).encode()).hexdigest()
            digest = hashlib.sha256((digest + manifest_digest).encode()).hexdigest()
        return dict(entry, instruction=body, version=digest, resources=resources[:64], scripts=scripts)

    def read_resource(self, name, relative):
        entry = self.require([name])[0]
        if entry["source"] != "package" or relative.split("/", 1)[0] not in {"references", "assets"}:
            raise SkillError("resource_not_readable")
        text, digest = self._text(self._path(entry["folder"], relative))
        return {"skill_id": name, "path": relative, "content": text, "version": digest}

    def resolve(self, names, loaded=None):
        result = []
        for entry in self.require(names):
            data = (loaded or {}).get(entry["id"]) or self.load(entry["id"])
            result.append(TestSkill(entry["id"], entry["triggers"], data["instruction"], entry["case_types"]))
        return result
