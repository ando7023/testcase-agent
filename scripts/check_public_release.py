"""Inspect a publishable Git ref without printing matched secrets.

Heuristic guard, not a substitute for reviewing business content.
Checks all commits reachable from the selected ref, not only its latest tree.
"""
import argparse
import os
from pathlib import Path, PurePosixPath
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(
        ["git", "-c", "safe.directory=" + ROOT.as_posix(), *args], cwd=ROOT)


def private_path(name):
    p = PurePosixPath(name)
    return (p.parts[0] in {"data", "tmp", ".local-private", ".idea", ".claude", ".codex", ".agents", ".venv"}
            or p.parts[0].startswith(".tmp-")
            or p.name == ".env" or (p.name.startswith(".env.") and p.name != ".env.example")
            or p.suffix.lower() in {".pdf", ".sqlite3", ".db", ".pem", ".key", ".bundle"}
            or "__pycache__" in p.parts)


def check(ref):
    local_keys = []
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip().endswith(("_API_KEY", "_TOKEN")):
                value = value.strip().strip("\"'")
                if len(value) >= 12:
                    local_keys.append(value.encode())
    issues, seen, count = set(), set(), 0
    for commit in git("rev-list", ref).decode().splitlines():
        count += 1
        for row in git("ls-tree", "-r", "-z", commit).split(b"\0"):
            if not row:
                continue
            meta, raw_name = row.split(b"\t", 1)
            mode, kind, oid = meta.decode().split()
            name = raw_name.decode("utf-8")
            if private_path(name):
                issues.add((name, "private/generated path in reachable history"))
            if mode in {"120000", "160000"}:
                issues.add((name, "review symlink/submodule before publishing"))
            if kind != "blob" or oid in seen:
                continue
            seen.add(oid)
            content = git("cat-file", "blob", oid)
            if any(key in content for key in local_keys):
                issues.add((name, "matches a local credential; value suppressed"))
            if re.search(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", content):
                issues.add((name, "private key material"))
            if re.search(rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{24,})", content):
                issues.add((name, "possible access token"))
            if re.search(rb"[Cc]:[\\/]+Users[\\/]+[A-Za-z0-9_.-]+[\\/]", content):
                issues.add((name, "personal absolute path"))
    for name, reason in sorted(issues):
        print(f"BLOCKED {name}: {reason}")
    print(f"Checked {count} reachable commits and {len(seen)} unique blobs; findings={len(issues)}")
    return not issues


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="HEAD")
    args = parser.parse_args()
    raise SystemExit(0 if check(args.ref) else 1)
