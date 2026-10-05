"""Run explicitly trusted Python skill helpers; this is not an OS security sandbox."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time

from jsonschema import Draft202012Validator, SchemaError, ValidationError

from .skill_runtime import SkillError


def check_schema(schema):
    if not isinstance(schema, dict):
        raise SkillError("invalid_script_schema")
    def walk(value):
        if isinstance(value, dict):
            if "$ref" in value or "$dynamicRef" in value:
                raise SkillError("script_schema_references_not_supported")
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(schema)
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError:
        raise SkillError("invalid_script_schema")


def validate_json(value, schema, code):
    try:
        Draft202012Validator(schema).validate(value)
    except ValidationError:
        raise SkillError(code)


class SkillScriptRunner:
    MAX_INPUT = 2 * 1024 * 1024
    MAX_OUTPUT = 65536

    def __init__(self, registry, work_root, trust_file=None):
        self.registry = registry
        self.work_root = Path(os.path.abspath(str(work_root))).resolve()
        self.trust_file = Path(trust_file or os.getenv("CASEFORGE_SKILL_TRUST_FILE") or
                               registry.root / "runtime-trust.json")

    def _trusted(self, name, relative, digest):
        try:
            with self.trust_file.open("rb") as handle:
                raw = handle.read(32769)
            if len(raw) > 32768:
                return False
            trusted = json.loads(raw)
            return isinstance(trusted, dict) and trusted.get(name + ":" + relative) == digest
        except (OSError, ValueError):
            return False

    @staticmethod
    def project_input(spec, project):
        source = spec["input_source"]
        if source == "cases":
            if not project.cases:
                raise SkillError("skill_script_requires_cases")
            return {"cases": [case.model_dump() for case in project.cases],
                    "analysis": project.analysis.model_dump() if project.analysis else None,
                    "module_tree": project.module_tree.model_dump() if project.module_tree else None}
        value = project.analysis if source == "analysis" else project.module_tree
        if not value:
            raise SkillError("skill_script_requires_" + source)
        return value.model_dump()

    @staticmethod
    def _kill(process):
        if os.name == "nt":
            taskkill = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/taskkill.exe"
            try:
                subprocess.run([str(taskkill), "/PID", str(process.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=subprocess.CREATE_NO_WINDOW, timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                pass
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            process.kill()

    def execute(self, name, script_id, arguments, data):
        loaded = self.registry.load(name)
        spec = loaded["scripts"].get(script_id)
        if not spec:
            raise SkillError("unknown_skill_script")
        if not isinstance(arguments, dict) or len(json.dumps(arguments).encode()) > 4096:
            raise SkillError("invalid_script_arguments")
        validate_json(arguments, spec["parameters"], "invalid_script_arguments")
        entry = self.registry.require([name])[0]
        path = self.registry._path(entry["folder"], spec["path"])
        with path.open("rb") as handle:
            script = handle.read(self.registry.MAX_TEXT + 1)
        if len(script) > self.registry.MAX_TEXT:
            raise SkillError("skill_script_too_large")
        digest = hashlib.sha256(script).hexdigest()
        if not self._trusted(name, spec["path"], digest):
            raise SkillError("skill_script_not_trusted")
        raw_input = json.dumps({"arguments": arguments, "data": data}, ensure_ascii=False).encode("utf-8")
        if len(raw_input) > self.MAX_INPUT:
            raise SkillError("skill_script_input_too_large")
        self.work_root.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="skill-", dir=str(self.work_root)) as directory:
            # Execute exactly the verified bytes, not a mutable original path.
            executable = Path(directory) / "helper.py"
            executable.write_bytes(script)
            env = {key: os.environ[key] for key in ("SystemRoot", "WINDIR") if key in os.environ}
            env.update(TMP=directory, TEMP=directory, TMPDIR=directory)
            options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
            try:
                process = subprocess.Popen([sys.executable, "-I", "-S", "-X", "utf8", str(executable)],
                                           cwd=directory, env=env, stdin=subprocess.PIPE,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, **options)
            except OSError:
                raise SkillError("skill_script_start_failed")
            overflow = threading.Event()
            captures = [bytearray(), bytearray()]
            def read_stream(stream, target):
                try:
                    while True:
                        chunk = stream.read(4096)
                        if not chunk:
                            break
                        target.extend(chunk[:self.MAX_OUTPUT + 1 - len(target)])
                        if len(target) > self.MAX_OUTPUT:
                            overflow.set()
                            self._kill(process)
                            break
                finally:
                    stream.close()
            def write_input():
                try:
                    process.stdin.write(raw_input)
                    process.stdin.flush()
                except (OSError, ValueError):
                    pass
                finally:
                    process.stdin.close()
            readers = [threading.Thread(target=read_stream, args=(stream, captures[i]), daemon=True)
                       for i, stream in enumerate((process.stdout, process.stderr))]
            for thread in readers:
                thread.start()
            writer = threading.Thread(target=write_input, daemon=True)
            writer.start()
            timed_out = False
            try:
                process.wait(timeout=spec["timeout_seconds"])
            except subprocess.TimeoutExpired:
                timed_out = True
                self._kill(process)
                process.wait(timeout=5)
            finally:
                for thread in readers + [writer]:
                    thread.join(timeout=5)
            if timed_out:
                raise SkillError("skill_script_timeout")
            if overflow.is_set() or any(t.is_alive() for t in readers):
                self._kill(process)
                raise SkillError("skill_script_output_limit")
            if process.returncode:
                raise SkillError("skill_script_failed")  # Never echo raw stderr or environment.
            try:
                output = json.loads(captures[0].decode("utf-8"))
            except (ValueError, UnicodeError, RecursionError):
                raise SkillError("skill_script_invalid_json")
            validate_json(output, spec["output_schema"], "skill_script_invalid_output")
            return {"skill_id": name, "script_id": script_id, "script_version": digest,
                    "skill_version": loaded["version"], "input_source": spec["input_source"],
                    "input_version": hashlib.sha256(raw_input).hexdigest(), "output": output,
                    "duration_ms": round((time.monotonic() - started) * 1000)}
