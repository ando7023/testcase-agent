import asyncio
import functools
import csv
import io
import json
import os
import re
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional

from typing_extensions import Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .evaluation import list_datasets
from .orchestrator import PipelineError, TestCaseOrchestrator
from .store import JsonStore
from .supervisor import AgenticSupervisor, CAPABILITIES
from .memory_versions import MemoryVersions, version_diff
from dataclasses import asdict


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "app" / "static"
DATA_DIR = Path(os.getenv("CASEFORGE_DATA_DIR", str(BASE_DIR / "data")))

store = JsonStore(DATA_DIR)
orchestrator = TestCaseOrchestrator(store)
app = FastAPI(title="CaseForge", version="0.1.0")


@app.middleware("http")
async def trace_http_request(request: Request, call_next):
    should_trace = request.url.path.startswith("/api/") and request.method in {
        "POST", "PUT", "PATCH", "DELETE"
    }
    if not should_trace:
        return await call_next(request)
    project_match = re.match(r"^/api/projects/([^/]+)", request.url.path)
    project_id = project_match.group(1) if project_match else ""
    operation = "{} {}".format(request.method, request.url.path)
    trace_run = None
    response = None
    with orchestrator.tracer.run(
        operation,
        project_id=project_id,
        transport="http",
        attributes={
            "method": request.method,
            "path": request.url.path,
            "content_type": request.headers.get("content-type", ""),
        },
    ) as trace_run:
        response = await call_next(request)
        trace_run.attributes["http_status"] = response.status_code
        root = trace_run.spans[0]
        root.attributes["http_status"] = response.status_code
        if response.status_code >= 500:
            root.status = "error"
        elif response.status_code >= 400:
            root.status = "client_error"
    response.headers["X-Trace-Id"] = trace_run.trace_id
    if project_id:
        store.attach_trace_run(project_id, trace_run.trace_id)
    return response


@app.middleware("http")
async def guard_project_writes(request: Request, call_next):
    match = re.match(r"^/api/projects/([^/]+)(/.*)?$", request.url.path)
    if not match or request.method not in {"POST", "PUT", "PATCH", "DELETE"} or (match.group(2) or "").startswith("/agent-runs"):
        return await call_next(request)
    lease = store.project_lease(match.group(1))
    try:
        lease.__enter__()
    except ValueError as exc:
        return JSONResponse(status_code=409, content={"detail": str(exc)})
    try:
        return await call_next(request)
    finally:
        lease.__exit__(None, None, None)


class AgentRunCreate(BaseModel):
    goal: str = Field(default="生成满足需求的测试用例，并根据评审反馈修复", min_length=1, max_length=4000)
    max_steps: int = Field(default=12, ge=1, le=20)


class AgentRunContinue(BaseModel):
    answer: str = Field(default="", max_length=4000)
    additional_steps: int = Field(default=0, ge=0, le=20, strict=True)


@app.get("/api/agent-capabilities")
def agent_capabilities():
    return [asdict(item) for item in CAPABILITIES.values()]


@app.post("/api/projects/{project_id}/agent-runs")
def start_agent_run(project_id: str, payload: AgentRunCreate):
    require_project(project_id)
    try:
        return AgenticSupervisor(orchestrator).start(project_id, payload.goal, payload.max_steps)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.get("/api/projects/{project_id}/agent-runs")
def list_agent_runs(project_id: str):
    require_project(project_id)
    return store.list_agent_runs(project_id)


@app.get("/api/projects/{project_id}/agent-runs/{run_id}")
def get_agent_run(project_id: str, run_id: str):
    require_project(project_id)
    run = store.get_agent_run(project_id, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Agent run not found")
    return run


@app.post("/api/projects/{project_id}/agent-runs/{run_id}/continue")
def continue_agent_run(project_id: str, run_id: str, payload: AgentRunContinue):
    require_project(project_id)
    try:
        return AgenticSupervisor(orchestrator).resume(project_id, run_id, payload.answer, payload.additional_steps)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


class ProjectCreate(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    requirement: str = Field(min_length=10)
    context: str = ""
    source_documents: List[dict] = Field(default_factory=list)


class ModuleConfirm(BaseModel):
    modules: List[dict]


class ModuleOperationCreate(BaseModel):
    mode: Literal["full", "regenerate", "continue", "targeted", "chat"] = "full"
    instruction: str = ""
    target_module_id: str = ""


class ModuleVersionRestore(BaseModel):
    version_id: str = Field(min_length=3)


class CaseOperationCreate(BaseModel):
    mode: Literal["full", "regenerate", "continue", "targeted", "chat"] = "full"
    instruction: str = ""
    target_module_id: str = ""


class CaseVersionRestore(BaseModel):
    version_id: str = Field(min_length=3)


TicketType = Literal["COMMON", "EQUIPMENT_BORROW", "P2P", "RISK_CONTROL", "GENERAL_TICKET"]


class KnowledgeCreate(BaseModel):
    title: str
    content: str = Field(min_length=5)
    doc_type: str = "business_rule"
    tags: List[str] = Field(default_factory=list)
    ticket_type: TicketType = "COMMON"
    source: str = ""
    section: str = ""
    page: Optional[int] = None
    version: str = "1.0"
    effective_at: str = ""
    expires_at: str = ""


class KnowledgeSplitCreate(BaseModel):
    parts: List[str] = Field(min_length=2)


class KnowledgeMergeCreate(BaseModel):
    document_ids: List[str] = Field(min_length=2)
    title: str = ""

class RAGSearchCreate(BaseModel):
    query: str = Field(min_length=3)


class RAGEvaluationCreate(BaseModel):
    dataset_id: str = "EQUIPMENT-RAG-V1"
    k: int = Field(default=5, ge=1, le=20)


class BenchmarkRunCreate(BaseModel):
    suite: str = Field(pattern="^(ebt_generation|storyseek_pipeline|srs_document|critic_mutation)$")
    limit: int = Field(default=3, ge=1, le=20)
    split: str = Field(default="test", pattern="^(development|validation|test|all)$")
    mode: str = Field(default="offline", pattern="^(offline|live)$")
    execution: Literal["workflow", "agentic"] = "workflow"
    human_policy: Literal["pause", "simulate_confirm"] = "pause"
    max_steps: int = Field(default=12, ge=1, le=20)
    stream: bool = True
    reasoning_effort: Literal["low", "high", "max"] = "low"
    timeout_seconds: int = Field(default=180, ge=1, le=900)


class MemoryCreate(BaseModel):
    rule: str = Field(min_length=3)
    ticket_type: TicketType = "COMMON"
    user_id: str = ""
    project_id: str = ""
    agent_id: str = ""
    fact_key: str = ""
    valid_from: str = ""
    valid_to: str = ""


class MemoryRevisionCreate(BaseModel):
    content: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class MemoryInvalidationCreate(BaseModel):
    reason: str = Field(min_length=1)


class MemoryExtractCreate(BaseModel):
    text: str = Field(min_length=3)
    ticket_type: TicketType = "COMMON"
    user_id: str = ""
    project_id: str = ""
    agent_id: str = ""
    run_id: str = ""
    source_run_id: str = ""
    source_version_id: str = ""
    source: str = "manual_conversation"
    infer: bool = True


class MemorySearchCreate(BaseModel):
    query: str = Field(min_length=2)
    ticket_type: TicketType = "COMMON"
    user_id: str = ""
    project_id: str = ""
    agent_id: str = ""
    run_id: str = ""
    top_k: int = Field(default=5, ge=1, le=20)
    token_budget: int = Field(default=1000, ge=100, le=8000)


class FeedbackCreate(BaseModel):
    action: str = Field(pattern="^(adopted|edited|rejected)$")
    reason: str = ""
    edited_case: Optional[dict] = None


class EvaluationCreate(BaseModel):
    dataset_id: str = ""


class TemplateLearnCreate(BaseModel):
    ticket_type: TicketType = "COMMON"
    content: str = Field(min_length=10)
    source: str = "test_plan"


def require_project(project_id: str):
    project = store.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


def pipeline_call(operation, project_id: str):
    project = require_project(project_id)
    try:
        return operation(project)
    except (PipelineError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "llm_enabled": orchestrator.llm.enabled,
        "model": orchestrator.llm.model,
        "knowledge_chunks": len(store.list_knowledge()),
        "memory_records": len(store.list_memory_records()),
        "domain": "ticket",
        "ocr_backend": orchestrator.document_agent.ocr_backend.name,
    }


@app.post("/api/documents/parse")
async def parse_document(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Document exceeds 15 MB")
    try:
        return orchestrator.parse_document(file.filename or "document.txt", data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/api/projects")
def list_projects():
    return store.list_projects()


@app.post("/api/projects")
def create_project(payload: ProjectCreate):
    return store.create_project(
        payload.title, payload.requirement, payload.context, payload.source_documents
    )


@app.get("/api/projects/{project_id}")
def get_project(project_id: str):
    return require_project(project_id)


@app.post("/api/projects/{project_id}/analyze")
def analyze(project_id: str):
    return pipeline_call(orchestrator.analyze, project_id)


@app.post("/api/projects/{project_id}/modules")
def plan_modules(project_id: str):
    return pipeline_call(orchestrator.plan_modules, project_id)


@app.post("/api/projects/{project_id}/modules/generate")
def operate_modules(project_id: str, payload: ModuleOperationCreate):
    project = require_project(project_id)
    try:
        return orchestrator.operate_modules(
            project,
            mode=payload.mode,
            instruction=payload.instruction,
            target_module_id=payload.target_module_id,
        )
    except (PipelineError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))


def run_project_stream(project_id, operation):
    with store.project_lease(project_id):
        return operation()


@app.post("/api/projects/{project_id}/modules/chat")
def chat_modules(project_id: str, payload: ModuleOperationCreate):
    project = require_project(project_id)
    try:
        return orchestrator.operate_modules(
            project,
            mode="chat",
            instruction=payload.instruction,
            target_module_id=payload.target_module_id,
        )
    except (PipelineError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.post("/api/projects/{project_id}/modules/restore")
def restore_modules(project_id: str, payload: ModuleVersionRestore):
    project = require_project(project_id)
    try:
        return orchestrator.restore_module_version(project, payload.version_id)
    except (PipelineError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.websocket("/ws/projects/{project_id}/modules")
async def module_workspace_socket(websocket: WebSocket, project_id: str):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_json()
            payload = ModuleOperationCreate.model_validate(raw)
            project = store.get_project(project_id)
            if not project:
                await websocket.send_json({"type": "error", "message": "Project not found"})
                continue
            loop = asyncio.get_running_loop()

            def publish(event):
                orchestrator.tracer.event(
                    "stream." + str(event.get("type", "event")), event
                )
                event = dict(event)
                event["trace_id"] = orchestrator.tracer.current_trace_id
                future = asyncio.run_coroutine_threadsafe(
                    websocket.send_json(event), loop
                )
                future.result(timeout=10)

            try:
                def operation():
                    project = require_project(project_id)
                    with orchestrator.tracer.run(
                        "WS modules.{}".format(payload.mode),
                        project_id=project_id,
                        transport="websocket",
                        attributes={
                            "mode": payload.mode,
                            "target_module_id": payload.target_module_id,
                        },
                    ) as trace_run:
                        result = orchestrator.operate_modules(
                            project,
                            payload.mode,
                            payload.instruction,
                            payload.target_module_id,
                            publish,
                        )
                    store.attach_trace_run(project_id, trace_run.trace_id)
                    return result, trace_run.trace_id

                result, trace_id = await loop.run_in_executor(None, functools.partial(run_project_stream, project_id, operation))
                await websocket.send_json({
                    "type": "complete",
                    "project": result.model_dump(),
                    "trace_id": trace_id,
                })
            except (PipelineError, ValueError) as exc:
                await websocket.send_json({"type": "error", "message": str(exc)})
    except WebSocketDisconnect:
        return

@app.put("/api/projects/{project_id}/modules/confirm")
def confirm_modules(project_id: str, payload: ModuleConfirm):
    project = require_project(project_id)
    try:
        return orchestrator.confirm_modules(project, payload.modules)
    except (PipelineError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.post("/api/projects/{project_id}/cases")
def generate_cases(project_id: str):
    return pipeline_call(orchestrator.generate_cases, project_id)


@app.post("/api/projects/{project_id}/cases/generate")
def operate_cases(project_id: str, payload: CaseOperationCreate):
    project = require_project(project_id)
    try:
        return orchestrator.operate_cases(
            project,
            mode=payload.mode,
            instruction=payload.instruction,
            target_module_id=payload.target_module_id,
        )
    except (PipelineError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.post("/api/projects/{project_id}/cases/chat")
def chat_cases(project_id: str, payload: CaseOperationCreate):
    project = require_project(project_id)
    try:
        return orchestrator.operate_cases(
            project,
            mode="chat",
            instruction=payload.instruction,
            target_module_id=payload.target_module_id,
        )
    except (PipelineError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.post("/api/projects/{project_id}/cases/restore")
def restore_cases(project_id: str, payload: CaseVersionRestore):
    project = require_project(project_id)
    try:
        return orchestrator.restore_case_version(project, payload.version_id)
    except (PipelineError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.websocket("/ws/projects/{project_id}/cases")
async def case_workspace_socket(websocket: WebSocket, project_id: str):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_json()
            payload = CaseOperationCreate.model_validate(raw)
            project = store.get_project(project_id)
            if not project:
                await websocket.send_json({"type": "error", "message": "Project not found"})
                continue
            loop = asyncio.get_running_loop()

            def publish(event):
                orchestrator.tracer.event(
                    "stream." + str(event.get("type", "event")), event
                )
                event = dict(event)
                event["trace_id"] = orchestrator.tracer.current_trace_id
                future = asyncio.run_coroutine_threadsafe(
                    websocket.send_json(event), loop
                )
                future.result(timeout=10)

            try:
                def operation():
                    project = require_project(project_id)
                    with orchestrator.tracer.run(
                        "WS cases.{}".format(payload.mode),
                        project_id=project_id,
                        transport="websocket",
                        attributes={
                            "mode": payload.mode,
                            "target_module_id": payload.target_module_id,
                        },
                    ) as trace_run:
                        result = orchestrator.operate_cases(
                            project,
                            payload.mode,
                            payload.instruction,
                            payload.target_module_id,
                            publish,
                        )
                    store.attach_trace_run(project_id, trace_run.trace_id)
                    return result, trace_run.trace_id

                result, trace_id = await loop.run_in_executor(None, functools.partial(run_project_stream, project_id, operation))
                await websocket.send_json({
                    "type": "complete",
                    "project": result.model_dump(),
                    "trace_id": trace_id,
                })
            except (PipelineError, ValueError) as exc:
                await websocket.send_json({"type": "error", "message": str(exc)})
    except WebSocketDisconnect:
        return


@app.post("/api/projects/{project_id}/review")
def review(project_id: str):
    return pipeline_call(orchestrator.review, project_id)


@app.post("/api/projects/{project_id}/revise")
def revise(project_id: str):
    return pipeline_call(orchestrator.revise, project_id)


@app.get("/api/evaluation/datasets")
def evaluation_datasets():
    return list_datasets()


@app.post("/api/projects/{project_id}/evaluate")
def evaluate_project(project_id: str, payload: EvaluationCreate = EvaluationCreate()):
    project = require_project(project_id)
    try:
        return orchestrator.evaluate(project, payload.dataset_id)
    except (PipelineError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.post("/api/projects/{project_id}/cases/{case_id}/feedback")
def case_feedback(project_id: str, case_id: str, payload: FeedbackCreate):
    project = require_project(project_id)
    try:
        return orchestrator.record_feedback(project, case_id, payload.action, payload.reason, payload.edited_case)
    except (PipelineError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.get("/api/projects/{project_id}/metrics")
def project_metrics(project_id: str):
    return orchestrator.metrics(require_project(project_id))


@app.get("/api/projects/{project_id}/traces")
def project_traces(project_id: str, limit: int = 30):
    require_project(project_id)
    runs = orchestrator.tracer.list_runs(project_id, limit)
    return [trace_summary(item) for item in runs]


@app.get("/api/traces/{trace_id}")
def trace_detail(trace_id: str):
    run = orchestrator.tracer.read(trace_id)
    if not run:
        raise HTTPException(status_code=404, detail="Trace not found")
    return run


def trace_summary(run):
    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    for span in run.spans:
        for key in usage:
            usage[key] += int(span.usage.get(key, 0) or 0)
    return {
        "trace_id": run.trace_id,
        "operation": run.operation,
        "project_id": run.project_id,
        "transport": run.transport,
        "status": run.status,
        "started_at": run.started_at,
        "duration_ms": run.duration_ms,
        "span_count": len(run.spans),
        "error_count": len([span for span in run.spans if span.status == "error"]),
        "usage": usage,
    }


@app.get("/api/badcases")
def list_badcases(project_id: Optional[str] = None):
    return store.list_badcases(project_id)


@app.get("/api/projects/{project_id}/xmind/{view}")
def project_xmind(project_id: str, view: str):
    project = require_project(project_id)
    try:
        content = orchestrator.export_xmind(project, view)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return Response(
        content=content,
        media_type=orchestrator.xmind_exporter.media_type,
        headers={
            "Content-Disposition": 'attachment; filename="{}-{}.xmind"'.format(
                project.id, view
            )
        },
    )


@app.get("/api/projects/{project_id}/mindmap/{view}")
def project_mindmap(project_id: str, view: str):
    try:
        return orchestrator.mindmap(require_project(project_id), view)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.get("/api/projects/{project_id}/export/{format_name}")
def export(project_id: str, format_name: str):
    project = require_project(project_id)
    if format_name == "json":
        return Response(
            content=project.model_dump_json(indent=2),
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="{}-cases.json"'.format(project.id)},
        )
    if format_name == "xmind":
        try:
            content = orchestrator.export_xmind(project, "cases")
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return Response(
            content=content,
            media_type=orchestrator.xmind_exporter.media_type,
            headers={"Content-Disposition": 'attachment; filename="{}-cases.xmind"'.format(project.id)},
        )
    if format_name == "mindmap":
        try:
            document = orchestrator.mindmap(project, "cases")
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return Response(
            content=document.model_dump_json(indent=2),
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="{}-mindmap.json"'.format(project.id)},
        )
    if format_name == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["ID", "模块", "标题", "优先级", "类型", "前置条件", "步骤与预期", "需求ID", "证据", "自动化可行性"])
        for case in project.cases:
            writer.writerow([
                case.id,
                case.module_id,
                case.title,
                case.priority,
                case.case_type,
                "\n".join(case.preconditions),
                "\n".join("{}. {} => {}".format(i, step.action, step.expected) for i, step in enumerate(case.steps, 1)),
                ",".join(case.requirement_ids),
                "\n".join(case.source_evidence),
                case.automation_feasibility,
            ])
        return Response(
            content="\ufeff" + output.getvalue(),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="{}-cases.csv"'.format(project.id)},
        )
    raise HTTPException(status_code=400, detail="Supported formats: json, mindmap, xmind, csv")


@app.get("/api/scenario-rules")
def list_scenario_rules():
    return store.list_scenario_rules()


@app.get("/api/scenario-templates")
def list_scenario_templates():
    return store.list_scenario_templates()


@app.post("/api/scenario-templates/learn")
def learn_scenario_templates(payload: TemplateLearnCreate):
    return orchestrator.learn_test_plan(
        payload.ticket_type, payload.content, payload.source
    )


@app.get("/api/knowledge")
def list_knowledge():
    return store.list_knowledge()


@app.post("/api/knowledge")
def add_knowledge(payload: KnowledgeCreate):
    return store.add_knowledge(
        payload.title,
        payload.content,
        payload.doc_type,
        payload.tags,
        payload.ticket_type,
        payload.source,
        payload.section,
        payload.page,
        payload.version,
        payload.effective_at,
        payload.expires_at,
    )

@app.post("/api/knowledge/ingest")
async def ingest_knowledge_document(
    file: UploadFile = File(...),
    ticket_type: TicketType = Form("COMMON"),
    doc_type: str = Form(""),
    version: str = Form("1.0"),
    effective_at: str = Form(""),
    expires_at: str = Form(""),
    chunking_version: str = Form("section-child-v1"),
):
    data = await file.read()
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Document exceeds 15 MB")
    try:
        return orchestrator.ingest_document(
            file.filename or "document.txt",
            data,
            ticket_type,
            doc_type,
            version,
            effective_at,
            expires_at,
            chunking_version,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/api/knowledge/{document_id}/convert")
def convert_knowledge_chunk(document_id: str):
    try:
        return orchestrator.convert_knowledge_chunk(document_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

@app.post("/api/knowledge/{document_id}/split/suggest")
def suggest_knowledge_split(document_id: str):
    try:
        return orchestrator.suggest_knowledge_split(document_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

@app.post("/api/knowledge/{document_id}/split")
def split_knowledge_chunk(document_id: str, payload: KnowledgeSplitCreate):
    try:
        return orchestrator.split_knowledge_chunk(document_id, payload.parts)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.post("/api/knowledge/merge")
def merge_knowledge_chunks(payload: KnowledgeMergeCreate):
    try:
        return orchestrator.merge_knowledge_chunks(
            payload.document_ids, payload.title
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

@app.get("/api/ebt/status")
def ebt_status():
    return orchestrator.ebt_status()


@app.post("/api/ebt/import")
def import_ebt():
    try:
        return orchestrator.import_ebt()
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=502, detail="EBT import failed: {}".format(exc))


@app.get("/api/benchmarks")
def benchmark_catalog():
    return orchestrator.benchmark_catalog()


@app.post("/api/benchmarks/{dataset_id}/import")
def import_public_benchmark(dataset_id: str):
    try:
        return orchestrator.import_public_benchmark(dataset_id)
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Public benchmark import failed: {}".format(exc),
        )


@app.post("/api/benchmarks/run")
def run_public_benchmark(payload: BenchmarkRunCreate):
    try:
        return orchestrator.run_public_benchmark(
            payload.suite, payload.limit, payload.split, payload.mode,
            execution=payload.execution, human_policy=payload.human_policy, max_steps=payload.max_steps,
            llm_options={"stream": payload.stream, "reasoning_effort": payload.reasoning_effort,
                         "timeout_seconds": payload.timeout_seconds},
        )
    except (OSError, ValueError, PipelineError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/api/benchmarks/reports/{report_id}")
def benchmark_report(report_id: str):
    try:
        return orchestrator.benchmark_report(report_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/api/benchmarks/stream")
def stream_public_benchmark(payload: BenchmarkRunCreate):
    from .benchmark_stream import benchmark_stream
    return benchmark_stream(lambda emit: orchestrator.run_public_benchmark(
        payload.suite, payload.limit, payload.split, payload.mode,
        execution=payload.execution, human_policy=payload.human_policy, max_steps=payload.max_steps,
        llm_options={"stream": payload.stream, "reasoning_effort": payload.reasoning_effort,
                     "timeout_seconds": payload.timeout_seconds}, on_event=emit))


@app.get("/api/tools")
def tool_catalog():
    return orchestrator.tool_catalog()

@app.post("/api/rag/search")
def rag_search(payload: RAGSearchCreate):
    return orchestrator.search_knowledge(payload.query)


@app.get("/api/rag/datasets")
def rag_datasets():
    return orchestrator.retrieval_datasets()


@app.post("/api/rag/evaluate")
def rag_evaluate(payload: RAGEvaluationCreate):
    try:
        return orchestrator.evaluate_retrieval(payload.dataset_id, payload.k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

@app.get("/api/memory")
def get_memory(include_inactive: bool = False):
    return orchestrator.memory_catalog(include_inactive=include_inactive)


class MemorySnapshotCreate(BaseModel):
    label: str = Field(default="手动快照", min_length=1, max_length=160)


class MemoryVersionCompare(BaseModel):
    kind: Literal["snapshot", "cases", "modules"]
    left_id: str = Field(min_length=1, max_length=120)
    right_id: str = Field(default="current", max_length=120)


class ProjectMemoryRestore(BaseModel):
    snapshot_id: str = Field(pattern=r"^MS-[a-f0-9]{32}$")
    expected_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")


class FactRollback(BaseModel):
    target_id: str = Field(min_length=1, max_length=120)
    expected_current_id: str = Field(min_length=1, max_length=120)
    reason: str = Field(default="人工回滚", min_length=1, max_length=1000)


@app.get("/api/projects/{project_id}/memory-versions")
def memory_versions(project_id: str):
    require_project(project_id)
    try:
        return MemoryVersions(store).catalog(project_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.post("/api/projects/{project_id}/memory-versions")
def create_memory_snapshot(project_id: str, payload: MemorySnapshotCreate):
    require_project(project_id)
    return MemoryVersions(store).create(project_id, payload.label)


@app.post("/api/projects/{project_id}/memory-versions/diff")
def compare_memory_versions(project_id: str, payload: MemoryVersionCompare):
    require_project(project_id)
    try:
        return MemoryVersions(store).compare(project_id, payload.kind, payload.left_id, payload.right_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.post("/api/projects/{project_id}/memory-versions/restore")
def restore_project_memory(project_id: str, payload: ProjectMemoryRestore):
    require_project(project_id)
    try:
        return MemoryVersions(store).restore(project_id, payload.snapshot_id, payload.expected_fingerprint)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.get("/api/memory/{memory_id}/history")
def memory_history(memory_id: str):
    try:
        return orchestrator.adaptive_memory.history(memory_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/api/memory/{memory_id}/diff")
def compare_fact_versions(memory_id: str, target_id: str):
    try:
        history = orchestrator.adaptive_memory.history(memory_id)
        versions = {v["id"]: v for v in history["versions"]}
        if target_id not in versions:
            raise ValueError("Target is not in this fact's version history")
        return dict(version_diff(versions[target_id], versions[memory_id]), kind="fact",
                    left_id=target_id, right_id=memory_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.post("/api/memory/{memory_id}/rollback")
def rollback_fact(memory_id: str, payload: FactRollback):
    try:
        record = next((m for m in store.list_memory_records(True) if m.id == memory_id), None)
        with store.project_lease(record.project_id) if record and record.project_id else nullcontext():
            return orchestrator.adaptive_memory.rollback(memory_id, payload.target_id, payload.expected_current_id, payload.reason)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.post("/api/memory/rules")
def add_memory_rule(payload: MemoryCreate):
    try:
        return orchestrator.add_memory_rule(**payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/api/memory/{memory_id}/revise")
def revise_memory(memory_id: str, payload: MemoryRevisionCreate):
    try:
        return orchestrator.adaptive_memory.revise(memory_id, payload.content, payload.reason)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.post("/api/memory/{memory_id}/invalidate")
def invalidate_memory(memory_id: str, payload: MemoryInvalidationCreate):
    try:
        return orchestrator.adaptive_memory.invalidate(memory_id, payload.reason)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.post("/api/memory/extract")
def extract_memory(payload: MemoryExtractCreate):
    return orchestrator.remember_text(
        payload.text,
        user_id=payload.user_id,
        project_id=payload.project_id,
        agent_id=payload.agent_id,
        run_id=payload.run_id,
        source_run_id=payload.source_run_id,
        source_version_id=payload.source_version_id,
        ticket_type=payload.ticket_type,
        source=payload.source,
        infer=payload.infer,
    )


@app.post("/api/memory/search")
def search_memory(payload: MemorySearchCreate):
    return orchestrator.search_memory(
        payload.query,
        user_id=payload.user_id,
        project_id=payload.project_id,
        agent_id=payload.agent_id,
        run_id=payload.run_id,
        ticket_type=payload.ticket_type,
        top_k=payload.top_k,
        token_budget=payload.token_budget,
    )


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
