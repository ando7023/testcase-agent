"""Unified diffs and project-scoped, recoverable Memory checkpoints. No LLM calls."""
import hashlib
import json
import re
import uuid

from .feedback import reconcile_feedback
from .models import AdaptiveMemoryRecord, CaseSetVersion, KnowledgeDocument, ModuleTreeVersion, ProjectState, utc_now_iso


def stable(value):
    """Ignore record telemetry, never similarly named fields in business payloads."""
    if isinstance(value, list):
        return [stable(v) for v in value]
    if isinstance(value, dict):
        if {"project", "facts", "knowledge", "badcases"} <= value.keys():
            return {**value, "project": stable(value["project"]), "facts": stable(value["facts"])}
        if {"id", "memory_type", "fact_key", "content"} <= value.keys():
            return {k: v for k, v in value.items() if k not in {"access_count", "last_accessed_at", "updated_at"}}
        if {"id", "requirement", "cases", "module_tree"} <= value.keys():
            return {k: v for k, v in value.items() if k not in {"updated_at", "human_acceptance"}}
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def version_diff(before, after):
    """Stable item IDs make reordered case/fact arrays comparable by identity."""
    changes = []
    def walk(left, right, path):
        if left == right:
            return
        if isinstance(left, list) and isinstance(right, list):
            if all(isinstance(v, dict) and v.get("id") for v in left + right):
                if len({v["id"] for v in left}) == len(left) and len({v["id"] for v in right}) == len(right):
                    return walk({v["id"]: v for v in left}, {v["id"]: v for v in right}, path)
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                child = path + "/" + str(key).replace("~", "~0").replace("/", "~1")
                if key not in left:
                    changes.append({"path": child, "operation": "added", "before": None, "after": right[key]})
                elif key not in right:
                    changes.append({"path": child, "operation": "removed", "before": left[key], "after": None})
                else:
                    walk(left[key], right[key], child)
        else:
            changes.append({"path": path or "/", "operation": "modified", "before": left, "after": right})
    walk(stable(before), stable(after), "")
    return {"changes": changes, "summary": {kind: sum(c["operation"] == kind for c in changes)
                                            for kind in ("added", "removed", "modified")}}


class MemoryVersions:
    def __init__(self, store):
        self.store = store

    def _project(self, project_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", project_id):
            raise ValueError("Invalid project ID")
        project = self.store.get_project(project_id)
        if project is None:
            raise ValueError("Project not found")
        return project

    @staticmethod
    def _owns(document, project_id):
        return (document.metadata.get("project_id") == project_id or
                document.id.startswith(("EX-" + project_id + "-", "BC-" + project_id + "-")))

    def _capture(self, project_id):
        project = self._project(project_id)
        return {"project": project.model_dump(exclude={"traces", "trace_run_ids", "human_acceptance"}),
                "facts": [m.model_dump() for m in self.store.list_memory_records(True) if m.project_id == project_id],
                "knowledge": [d.model_dump() for d in self.store.list_knowledge() if self._owns(d, project_id)],
                "badcases": self.store.list_badcases(project_id)}

    def _path(self, project_id, snapshot_id):
        if not re.fullmatch(r"MS-[a-f0-9]{32}", snapshot_id):
            raise ValueError("Invalid snapshot ID")
        self._project(project_id)
        return self.store.root / "memory_snapshots" / project_id / (snapshot_id + ".json")

    def _new_snapshot(self, project_id, label, bundle):
        return {"schema_version": 1, "id": "MS-" + uuid.uuid4().hex, "project_id": project_id,
                "label": label, "created_at": utc_now_iso(), "checksum": digest(bundle), "bundle": bundle}

    @staticmethod
    def _summary(snapshot):
        return {k: v for k, v in snapshot.items() if k != "bundle"}

    def _load(self, project_id, snapshot_id):
        snapshot = self.store._read_json(self._path(project_id, snapshot_id), None)
        if not snapshot or snapshot.get("project_id") != project_id or snapshot.get("id") != snapshot_id:
            raise ValueError("Snapshot not found in this project")
        if snapshot.get("schema_version") != 1 or digest(snapshot.get("bundle")) != snapshot.get("checksum"):
            raise ValueError("Snapshot integrity check failed")
        bundle = snapshot["bundle"]
        project = ProjectState.model_validate(bundle["project"])
        facts = [AdaptiveMemoryRecord.model_validate(v) for v in bundle["facts"]]
        knowledge = [KnowledgeDocument.model_validate(v) for v in bundle["knowledge"]]
        if (project.id != project_id or any(m.project_id != project_id for m in facts) or
                any(not self._owns(d, project_id) for d in knowledge) or
                any(b.get("project_id") != project_id for b in bundle["badcases"])):
            raise ValueError("Snapshot contains data outside this project")
        for items in (facts, knowledge, project.cases):
            if len({v.id for v in items}) != len(items):
                raise ValueError("Snapshot contains duplicate IDs")
        def module_ids(modules):
            return [m.id for m in modules] + [v for m in modules for v in module_ids(m.children)]
        ids = module_ids(project.module_tree.modules) if project.module_tree else []
        if len(ids) != len(set(ids)) or any(c.module_id not in ids for c in project.cases):
            raise ValueError("Snapshot case/module references are inconsistent")
        return snapshot

    def create(self, project_id, label="手动快照"):
        with self.store.memory_transaction():
            snapshot = self._new_snapshot(project_id, label, self._capture(project_id))
            self.store._write_json(self._path(project_id, snapshot["id"]), snapshot)
            return self._summary(snapshot)

    def catalog(self, project_id):
        with self.store.memory_transaction():
            project = self._project(project_id)
            folder = self.store.root / "memory_snapshots" / project_id
            snapshots = [self._summary(self._load(project_id, p.stem)) for p in folder.glob("MS-*.json")]
            return {"snapshots": sorted(snapshots, key=lambda s: (s["created_at"], s["id"]), reverse=True),
                    "modules": [{"id": v.id, "created_at": v.created_at, "label": v.instruction} for v in project.module_versions],
                    "cases": [{"id": v.id, "created_at": v.created_at, "label": v.instruction} for v in project.case_versions]}

    def compare(self, project_id, kind, left_id, right_id="current"):
        with self.store.memory_transaction():
            project = self._project(project_id)
            current = self._capture(project_id)
            def value(version_id):
                if kind == "snapshot":
                    return current if version_id == "current" else self._load(project_id, version_id)["bundle"]
                versions = project.case_versions if kind == "cases" else project.module_versions
                if version_id == "current":
                    return ({"cases": [c.model_dump() for c in project.cases], "memory_state": project.case_decision_state.model_dump()}
                            if kind == "cases" else {"module_tree": project.module_tree.model_dump() if project.module_tree else None,
                                                     "memory_state": project.module_decision_state.model_dump()})
                version = next((v for v in versions if v.id == version_id), None)
                if not version:
                    raise ValueError("Version not found in this project")
                return ({"cases": [c.model_dump() for c in version.cases], "memory_state": version.memory_state.model_dump() if version.memory_state else None}
                        if kind == "cases" else {"module_tree": version.module_tree.model_dump(),
                                                "memory_state": version.memory_state.model_dump() if version.memory_state else None})
            if kind not in {"snapshot", "cases", "modules"}:
                raise ValueError("Unknown version kind")
            return dict(version_diff(value(left_id), value(right_id)), kind=kind, left_id=left_id, right_id=right_id,
                        current_fingerprint=digest(stable(current)))

    def restore(self, project_id, snapshot_id, expected_fingerprint):
        with self.store.memory_transaction():
            snapshot = self._load(project_id, snapshot_id)
            current = self._capture(project_id)
            if not expected_fingerprint or expected_fingerprint != digest(stable(current)):
                raise ValueError("项目已变化，请重新查看差异后恢复")
            if any(r.status == "running" for r in self.store.list_agent_runs(project_id)):
                raise ValueError("项目仍有运行中的 Agent，请先处理运行状态")
            backup = self._new_snapshot(project_id, "恢复前自动备份", current)
            bundle = snapshot["bundle"]
            project = ProjectState.model_validate(bundle["project"])
            latest = self._project(project_id)
            # Preserve later human decisions and audit trails; never silently undo a rejection.
            project.feedback = latest.feedback
            project.traces, project.trace_run_ids = latest.traces, latest.trace_run_ids
            project.review = project.evaluation = None
            project.module_review = None
            project.memory_epoch = uuid.uuid4().hex
            if project.module_tree:
                project.module_tree.confirmed = False
            for case in project.cases:
                case.review_status = "pending"
            project.phase = "modules_planned" if project.module_tree else "analyzed" if project.analysis else "draft"
            project.updated_at = utc_now_iso()
            for field in ("case_versions", "module_versions"):
                versions = {v.id: v for v in getattr(project, field)}
                versions.update({v.id: v for v in getattr(latest, field)})
                setattr(project, field, list(versions.values()))
            if project.module_tree:
                project.module_versions.append(ModuleTreeVersion(id="MV-" + uuid.uuid4().hex[:10], mode="restore",
                    instruction="恢复 Memory 快照 " + snapshot_id, module_tree=project.module_tree.model_copy(deep=True),
                    memory_state=project.module_decision_state.model_copy(deep=True)))
            if project.cases:
                project.case_versions.append(CaseSetVersion(id="CV-" + uuid.uuid4().hex[:10], mode="restore",
                    instruction="恢复 Memory 快照 " + snapshot_id, cases=[c.model_copy(deep=True) for c in project.cases],
                    memory_state=project.case_decision_state.model_copy(deep=True)))
            reconcile_feedback(project)
            records = {v["id"]: AdaptiveMemoryRecord.model_validate(v) for v in bundle["facts"]}
            all_records = self.store.list_memory_records(True)
            if any(m.id in records and m.project_id != project_id for m in all_records):
                raise ValueError("Memory ID collision outside project")
            for item in all_records:
                if item.project_id == project_id and item.id not in records:
                    item.status = "inactive"
                    item.invalidation_reason = "project_snapshot_restored"
                    records[item.id] = item
            # Human confirmation must be renewed even when its old fact was captured.
            for item in records.values():
                if item.source == "human_gate" and item.status == "active":
                    item.status = "inactive"
                    item.invalidation_reason = "project_snapshot_requires_confirmation"
            facts = [m.model_dump() for m in all_records if m.project_id != project_id] + [m.model_dump() for m in records.values()]
            docs = {v["id"]: KnowledgeDocument.model_validate(v) for v in bundle["knowledge"]}
            other_docs = []
            for doc in self.store.list_knowledge():
                if not self._owns(doc, project_id):
                    if doc.id in docs:
                        raise ValueError("Knowledge ID collision outside project")
                    other_docs.append(doc.model_dump())
                elif doc.id not in docs:
                    doc.status = "inactive"
                    docs[doc.id] = doc
            mutations, _ = self.store.case_example_mutations(project, list(docs.values()))
            docs.update({v["id"]: KnowledgeDocument.model_validate(v) for v in mutations})
            badcases = [v for v in self.store.list_badcases() if v.get("project_id") != project_id] + bundle["badcases"]
            receipt = {"id": "MR-" + uuid.uuid4().hex, "project_id": project_id, "snapshot_id": snapshot_id,
                       "backup_id": backup["id"], "created_at": utc_now_iso()}
            self.store.atomic_write({
                "projects/" + project_id + ".json": project.model_dump(),
                "adaptive_memory.json": facts, "knowledge.json": other_docs + [d.model_dump() for d in docs.values()],
                "badcases.json": badcases,
                "memory_snapshots/{}/{}.json".format(project_id, backup["id"]): backup,
                "memory_restores/{}/{}.json".format(project_id, receipt["id"]): receipt,
            })
            return dict(receipt, project=project.model_dump())
