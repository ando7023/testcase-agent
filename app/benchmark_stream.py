"""Request-scoped SSE bridge. Disconnect detaches the viewer, not the model job."""
import asyncio
import json
import queue
import threading
import time

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

_slots = threading.BoundedSemaphore(2)


class EventBridge:
    def __init__(self):
        self.queue = queue.Queue(maxsize=256)
        self.detached = threading.Event()
        self.done = threading.Event()

    def emit(self, event):
        # Bound memory for slow readers; disconnect releases any producer wait.
        while not self.detached.is_set():
            try:
                self.queue.put(event, timeout=0.1)
                return
            except queue.Full:
                pass

    async def events(self):
        last_heartbeat = time.monotonic()
        try:
            while True:
                try:
                    event = self.queue.get_nowait()
                except queue.Empty:
                    if self.done.is_set():
                        break
                    if time.monotonic() - last_heartbeat >= 2:
                        yield 'data: {"event":"heartbeat"}\n\n'
                        last_heartbeat = time.monotonic()
                    await asyncio.sleep(0.05)
                    continue
                yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
        finally:
            self.detached.set()


def benchmark_stream(operation):
    if not _slots.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="已有两次评测在运行，请等待完成后再试。")
    bridge = EventBridge()

    def run():
        try:
            bridge.emit({"event": "started"})
            report = operation(bridge.emit)
            bridge.emit({"event": "finished", "report": report})
        except Exception as exc:
            bridge.emit({"event": "error", "code": type(exc).__name__,
                         "message": "评测未能完成，请检查数据集导入、模型配置及本地诊断。"})
        finally:
            bridge.done.set()
            _slots.release()

    # Thread has no request Trace ContextVars: sample traces remain isolated.
    thread = threading.Thread(target=run, daemon=True, name="benchmark-stream")
    try:
        thread.start()
    except Exception:
        _slots.release()
        raise
    return StreamingResponse(bridge.events(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"})
