import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.models import ReviewReport
from app.orchestrator import TestCaseOrchestrator
from app.skill_runtime import SkillRegistry
from app.store import JsonStore
from app.supervisor import AgenticSupervisor
from test_skill_runtime import write_script


def tool(name, **fields):
    return dict(action="invoke_tool", capability=name, reason="Use the selected skill", **fields)


def ask():
    return dict(action="request_input", reason="Pause for the test", question="Continue?")


class Planner:
    enabled = True
    def __init__(self, decisions):
        self.decisions = decisions
        self.inputs = []
    def generate_json(self, system, user, schema):
        self.inputs.append(json.loads(user))
        return self.decisions.pop(0)


class SkillSupervisorTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        skill_root = root / "skills"
        folder = skill_root / "audit"
        (folder / "references").mkdir(parents=True)
        (folder / "SKILL.md").write_text('---\nname: audit-cases\ndescription: Audit cases\nmetadata:\n  case_types: [functional]\n---\nBODY_AFTER_LOADING. Read references/check.md and use the check script when needed.', encoding="utf-8")
        (folder / "references/check.md").write_text("REFERENCE_AFTER_READING", encoding="utf-8")
        script = write_script(folder, 'import json,sys\nx=json.load(sys.stdin)\nprint(json.dumps({"actual_case_count":len(x["data"]["cases"])}))')
        (skill_root / "runtime-trust.json").write_text(json.dumps({"audit-cases:scripts/check.py": hashlib.sha256(script.read_bytes()).hexdigest()}), encoding="utf-8")
        self.worker = TestCaseOrchestrator(JsonStore(root / "data"))
        self.worker.llm.api_key = ""
        self.worker.skills = SkillRegistry(skill_root)
        project = self.worker.store.create_project("Synthetic", "登录用户创建申请，数量范围1到3。")
        project = self.worker.analyze(project)
        project = self.worker.plan_modules(project)
        project = self.worker.generate_cases(project)
        self.project_id = project.id

    def test_load_read_execute_resume_and_feed_result_to_generation(self):
        planner = Planner([tool("load_skill", skill_id="audit-cases"),
                           tool("read_skill_resource", skill_id="audit-cases", resource_path="references/check.md"), ask()])
        run = AgenticSupervisor(self.worker, planner).start(self.project_id)
        self.assertNotIn("instruction", next(s for s in planner.inputs[0]["skills"] if s["id"] == "audit-cases"))
        self.assertNotIn("BODY_AFTER_LOADING", json.dumps(planner.inputs[0]))
        self.assertIn("BODY_AFTER_LOADING", json.dumps(planner.inputs[1]))
        self.assertNotIn("REFERENCE_AFTER_READING", json.dumps(planner.inputs[1]))
        self.assertIn("REFERENCE_AFTER_READING", json.dumps(planner.inputs[2]))
        planner = Planner([tool("run_skill_script", skill_id="audit-cases", script_id="check"),
                           dict(action="invoke_agent", capability="case_generation", skills=["audit-cases"], mode="chat", reason="Use audit findings", instruction="保持现有用例"), ask()])
        with patch.object(self.worker.case_agent, "run", wraps=self.worker.case_agent.run) as worker_run:
            result = AgenticSupervisor(self.worker, planner).resume(self.project_id, run.id, "继续")
        self.assertEqual(result.status, "waiting_input")
        self.assertEqual(result.steps[3].status, "success")
        self.assertTrue(result.skill_script_results[0]["output"]["actual_case_count"])
        context = worker_run.call_args[0][1]
        self.assertIn("BODY_AFTER_LOADING", context["knowledge"])
        self.assertIn("REFERENCE_AFTER_READING", context["knowledge"])
        self.assertIn("actual_case_count", context["knowledge"])
        self.assertEqual(context["loaded_test_skills"][0].name, "audit-cases")
        self.assertEqual(self.worker.store.get_agent_run(self.project_id, run.id).loaded_skills, result.loaded_skills)

    def test_script_cannot_replace_semantic_review_or_run_before_loading(self):
        planner = Planner([tool("run_skill_script", skill_id="audit-cases", script_id="check"),
                           tool("load_skill", skill_id="audit-cases"),
                           tool("run_skill_script", skill_id="audit-cases", script_id="check"),
                           dict(action="finish", reason="Script passed"), ask()])
        run = AgenticSupervisor(self.worker, planner).start(self.project_id)
        self.assertEqual(run.steps[0].status, "rejected")
        self.assertEqual(run.steps[2].status, "success")
        self.assertEqual(run.steps[3].status, "rejected")
        self.assertIsNone(self.worker.store.get_project(self.project_id).review)

    def test_stale_script_results_are_not_forwarded_to_worker(self):
        planner = Planner([tool("load_skill", skill_id="audit-cases"),
                           tool("run_skill_script", skill_id="audit-cases", script_id="check"), ask()])
        run = AgenticSupervisor(self.worker, planner).start(self.project_id)
        project = self.worker.store.get_project(self.project_id)
        project.cases[0].title += " changed"
        self.worker.store.save_project(project)
        planner = Planner([dict(action="invoke_agent", capability="case_generation", skills=["audit-cases"], mode="chat", reason="Use selected skill"), ask()])
        with patch.object(self.worker.case_agent, "run", wraps=self.worker.case_agent.run) as worker_run:
            AgenticSupervisor(self.worker, planner).resume(self.project_id, run.id, "继续")
        self.assertTrue(planner.inputs[0]["skill_script_results"][0]["stale"])
        self.assertNotIn("actual_case_count", worker_run.call_args[0][1]["knowledge"])

    def test_selected_package_is_loaded_without_running_any_script(self):
        planner = Planner([dict(action="invoke_agent", capability="case_generation", skills=["audit-cases"], mode="chat", reason="Use selected skill"), ask()])
        with patch.object(self.worker.case_agent, "run", wraps=self.worker.case_agent.run) as worker_run:
            run = AgenticSupervisor(self.worker, planner).start(self.project_id)
        self.assertEqual(run.steps[0].status, "success")
        self.assertIn("audit-cases", run.loaded_skills)
        self.assertFalse(run.skill_script_results)
        self.assertFalse(run.skill_resources)
        self.assertNotIn("REFERENCE_AFTER_READING", worker_run.call_args[0][1]["knowledge"])

    def test_changed_resources_can_be_reread_without_waiving_duplicate_guard(self):
        planner = Planner([tool("load_skill", skill_id="audit-cases"),
                           tool("read_skill_resource", skill_id="audit-cases", resource_path="references/check.md"), ask()])
        supervisor = AgenticSupervisor(self.worker, planner)
        run = supervisor.start(self.project_id)
        path = self.worker.skills.root / "audit/references/check.md"
        path.write_text("UPDATED_REFERENCE", encoding="utf-8")
        planner.decisions = [tool("read_skill_resource", skill_id="audit-cases", resource_path="references/check.md"),
                             tool("read_skill_resource", skill_id="audit-cases", resource_path="references/check.md"), ask()]
        result = supervisor.resume(self.project_id, run.id, "资料已更新")
        self.assertEqual(result.steps[3].status, "success")
        self.assertEqual(result.steps[4].status, "rejected")
        self.assertEqual(result.skill_resources["audit-cases:references/check.md"]["content"], "UPDATED_REFERENCE")

    def test_changed_skill_is_reported_and_can_be_reloaded(self):
        planner = Planner([tool("load_skill", skill_id="audit-cases"), ask()])
        supervisor = AgenticSupervisor(self.worker, planner)
        run = supervisor.start(self.project_id)
        path = self.worker.skills.root / "audit/SKILL.md"
        path.write_text(path.read_text(encoding="utf-8") + "\nNEW_INSTRUCTION", encoding="utf-8")
        planner.decisions = [tool("load_skill", skill_id="audit-cases"), ask()]
        result = supervisor.resume(self.project_id, run.id, "技能内容已更新")
        self.assertIn("audit-cases", planner.inputs[2]["changed_skills"])
        self.assertEqual(result.steps[2].status, "success")
        self.assertIn("NEW_INSTRUCTION", result.loaded_skills["audit-cases"]["instruction"])


if __name__ == "__main__":
    unittest.main()
