"""Inspect skill packages or run a trusted helper on saved project artifacts, without an LLM."""
import argparse
import json
import os
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.skill_runtime import SkillRegistry, SkillError
from app.skill_scripts import SkillScriptRunner
from app.models import ProjectState


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--skill")
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--resource")
    operation.add_argument("--script")
    parser.add_argument("--project-id")
    parser.add_argument("--data-root", type=Path, default=Path(os.getenv("CASEFORGE_DATA_DIR", str(ROOT / "data"))))
    parser.add_argument("--arguments-json", default="{}")
    args = parser.parse_args()
    registry = SkillRegistry()
    if args.list:
        if args.skill or args.resource or args.script:
            parser.error("--list cannot be combined with a skill operation")
        return {"skills": registry.catalog(), "errors": registry.errors}
    if not args.skill:
        parser.error("Specify --list or --skill")
    loaded = registry.load(args.skill)
    if args.resource:
        return registry.read_resource(args.skill, args.resource)
    if not args.script:
        return loaded
    if not args.project_id:
        parser.error("--script requires --project-id")
    spec = loaded["scripts"].get(args.script)
    if not spec:
        raise SkillError("unknown_skill_script")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", args.project_id):
        raise SkillError("invalid_project_id")
    project_path = args.data_root / "projects" / (args.project_id + ".json")
    if not project_path.is_file():
        raise SkillError("skill_project_not_found")
    project = ProjectState.model_validate_json(project_path.read_text(encoding="utf-8"))
    runner = SkillScriptRunner(registry, args.data_root / "skill_runs")
    try:
        arguments = json.loads(args.arguments_json)
    except ValueError:
        raise SkillError("invalid_script_arguments")
    return runner.execute(args.skill, args.script, arguments, runner.project_input(spec, project))


if __name__ == "__main__":
    try:
        print(json.dumps(main(), ensure_ascii=True, indent=2))
    except (SkillError, OSError) as exc:
        print(json.dumps({"error_code": exc.code if isinstance(exc, SkillError) else "skill_resource_unavailable"}))
        raise SystemExit(1)
