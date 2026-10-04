/* Model content is untrusted text. Never insert it as HTML or expose reasoning. */
(function (root) {
  class SSEDecoder {
    constructor(onEvent) { this.buffer = ''; this.onEvent = onEvent; }
    feed(text, final = false) {
      this.buffer += text;
      let boundary;
      while ((boundary = /\r?\n\r?\n/.exec(this.buffer))) {
        const frame = this.buffer.slice(0, boundary.index);
        this.buffer = this.buffer.slice(boundary.index + boundary[0].length);
        this.frame(frame);
      }
      if (final && this.buffer.trim()) this.frame(this.buffer);
      if (final) this.buffer = '';
    }
    frame(frame) {
      const data = frame.split(/\r?\n/).filter(line => line.startsWith('data:'))
        .map(line => line.slice(5).replace(/^ /, '')).join('\n');
      if (data) this.onEvent(JSON.parse(data));
    }
  }

  const labels = {summary: '摘要', question: '待确认', message: '说明', title: '标题',
    statement: '需求', description: '描述', objective: '目标', action: '操作',
    expected_result: '预期结果', expected: '预期', instruction: '任务', evidence: '依据'};
  function preview(text) {
    const lines = [];
    const pattern = /"(summary|question|message|title|statement|description|objective|action|expected_result|expected|instruction|evidence)"\s*:\s*"((?:\\.|[^"\\])*)(?:"|$)/g;
    for (const match of text.matchAll(pattern)) {
      // An incomplete escape at the tail is left for the next content chunk.
      let value = match[2].replace(/\\u[0-9a-f]{0,3}$/i, '').replace(/\\$/, '');
      try { value = JSON.parse('"' + value + '"'); } catch (_) { continue; }
      if (value) lines.push(labels[match[1]] + '：' + value);
    }
    return lines.length ? lines.join('\n\n') : text;
  }

  const agents = {supervisor: '编排 Agent', requirement_understanding: '需求理解 Agent',
    module_planning: '模块规划 Agent', module_critic: '模块评审 Agent', module_revision: '模块修复 Agent',
    case_generation: '用例生成 Agent', case_review: '用例评审 Agent', requirement_analysis: '需求理解 Agent',
    requirement_analyst: '需求理解 Agent', module_planner: '模块规划 Agent',
    case_generator: '用例生成 Agent', quality_critic: '质量评审 Agent',
    case_revision: '用例修复', chat: '反馈修复', finish: '收尾', request_input: '请求确认',
    knowledge_retrieval: '知识检索', worker: '执行 Agent'};
  const agentName = name => agents[name] || name || 'Agent';
  const statuses = {passed: '流程与质量门禁通过', completed: '流程已完成',
    quality_failed: '质量门禁未通过', waiting_input: '等待补充信息',
    waiting_confirmation: '等待模块确认', technical_failed: '技术失败',
    degraded: '存在降级，需要复核', needs_attention: '需要复核', incomplete: '尚未完成'};

  class BenchmarkChat {
    constructor(container) {
      this.root = container;
      this.log = container.querySelector('.benchmark-chat-log');
      this.status = container.querySelector('[data-chat-status]');
      this.clock = container.querySelector('[data-chat-clock]');
      this.follow = container.querySelector('[data-chat-follow]');
      this.following = true;
      this.log.addEventListener('scroll', () => {
        this.following = this.log.scrollHeight - this.log.scrollTop - this.log.clientHeight < 60;
        this.follow.hidden = this.following;
      });
      this.follow.addEventListener('click', () => {
        this.following = true; this.scroll(); this.follow.hidden = true;
      });
    }
    scroll() { if (this.following) this.log.scrollTop = this.log.scrollHeight; }
    message(label, text, kind = 'assistant') {
      const node = document.createElement('article');
      node.className = 'benchmark-chat-message ' + kind;
      const heading = document.createElement('h4'); heading.textContent = label;
      const body = document.createElement('div'); body.className = 'benchmark-chat-body';
      body.textContent = text; node.append(heading, body); this.log.append(node);
      // The complete artifacts/report remain on disk; bound this live viewport.
      if (this.log.children.length > 100) {
        const old = this.log.firstElementChild;
        for (const [id, call] of this.calls) if (call.node === old) this.calls.delete(id);
        old.remove();
        this.root.querySelector('[data-chat-retention]').hidden = false;
      }
      this.scroll(); return {node, body};
    }
    render(call) {
      call.body.textContent = preview(call.text) || '等待模型返回正文…';
      call.raw.textContent = call.text;
      this.scroll();
    }
    handle(event) {
      if (this.terminal) return;
      this.received = Date.now();
      const call = this.calls.get(event.call_id);
      switch (event.event) {
        case 'started': this.status.textContent = '评测已启动'; break;
        case 'report_start': this.reportId = event.report_id; break;
        case 'sample_start':
          this.status.textContent = `正在处理样本 ${event.sample_id}`;
          this.message(`样本 ${event.sample_id}`, `开始第 ${event.sample_index} 个样本。`, 'stage'); break;
        case 'planning':
          this.status.textContent = `编排决策 · 第 ${event.index} / ${event.max_steps} 步`; break;
        case 'decision':
          this.message('编排决策', `第 ${event.index} 步 · ${agentName(event.capability)}`, 'stage'); break;
        case 'agent_start': this.status.textContent = `${agentName(event.agent)} 正在执行`; break;
        case 'agent_end':
          this.status.textContent = `${agentName(event.agent)} ${event.status === 'success' ? '执行完成' : '执行结束，请查看报告'}`; break;
        case 'llm_start': {
          const item = this.message(`${agentName(event.agent)} · 调用 ${event.call}`, '等待模型返回正文…');
          const note = document.createElement('p'); note.className = 'benchmark-chat-note';
          note.textContent = event.stream ? '正在连接模型 · 正文将实时显示' : '非流式调用 · 完成后一次性显示正文';
          const details = document.createElement('details');
          const summary = document.createElement('summary'); summary.textContent = '查看原始响应正文';
          const raw = document.createElement('pre'); details.append(summary, raw);
          item.node.append(note, details); item.node.classList.add('streaming');
          this.calls.set(event.call_id, {...item, note, raw, text: '', truncated: false, pending: false}); break;
        }
        case 'llm_progress':
          if (call) {
            call.note.textContent = event.phase === 'connected' ? '已连接模型，等待正文…' :
              event.content_chars ? '正在接收正文 · 预览尚未完成校验' : '模型正在处理，尚未返回正文…';
          }
          break;
        case 'delta':
          if (call) {
            call.note.textContent = '正在接收正文 · 预览尚未完成校验';
            const available = 60000 - call.text.length;
            call.text += event.text.slice(0, available);
            call.truncated ||= event.text.length > available;
            if (call.truncated) call.note.textContent += ' · 预览已截断，完整产物请查看报告与本地 Trace';
            if (!call.pending) {
              call.pending = true;
              requestAnimationFrame(() => { call.pending = false; this.render(call); });
            }
          }
          break;
        case 'llm_end':
          if (call) {
            this.render(call); call.node.classList.remove('streaming');
            call.note.textContent = event.status === 'success' ? '本次响应已接收 · 质量结论以最终报告为准' : '本次调用失败 · 部分正文不能作为有效产物';
            if (call.truncated) call.note.textContent += ' · 预览已截断';
          }
          break;
        case 'sample_end':
          this.message(`样本 ${event.sample_id} · ${statuses[event.status] || event.status}`,
            event.question || '该样本执行已结束，详细结论将写入报告。', event.technical_failure ? 'error' : 'stage'); break;
        case 'finished':
          this.terminal = true; this.report = event.report;
          this.status.textContent = '评测结束';
          this.message('评测报告已保存', '请查看下方报告中的流程、质量门禁与技术失败结果。'); break;
        case 'error':
          this.terminal = true; throw new Error(event.message || '评测执行失败。');
      }
      this.scroll();
    }
    async run(payload, label) {
      this.calls = new Map(); this.terminal = false; this.report = null; this.reportId = '';
      this.log.replaceChildren(); this.following = true; this.follow.hidden = true;
      this.root.hidden = false; this.root.querySelector('[data-chat-retention]').hidden = true;
      this.started = this.received = Date.now();
      this.status.textContent = '正在启动';
      this.message('你', `${label}\n${payload.mode === 'live' ? '真实模型' : '离线模式'} · ${payload.execution === 'agentic' ? 'Agentic 动态编排' : 'Workflow 固定流程'}${payload.mode === 'live' ? (payload.stream ? ' · 流式正文' : ' · 非流式正文') : ''}`, 'user');
      this.message('评测范围', `${payload.clarification_policy === 'evidence_only' ? '按原文生成行为级用例，执行细节单独记录；关键歧义仍会暂停。' : '严格澄清。'} 模块规划后自主推进，不补造业务答案。`, 'stage');
      const tick = () => {
        const elapsed = Math.floor((Date.now() - this.started) / 1000);
        const quiet = Math.floor((Date.now() - this.received) / 1000);
        this.clock.textContent = `${Math.floor(elapsed / 60)}:${String(elapsed % 60).padStart(2, '0')}${quiet > 10 && !this.terminal ? ` · 已 ${quiet} 秒未收到服务端事件` : ''}`;
      };
      tick(); const timer = setInterval(tick, 1000);
      this.root.scrollIntoView({block: 'nearest'});
      let reader;
      try {
        const response = await fetch('/api/benchmarks/stream', {
          method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
        if (!response.ok) {
          const body = await response.json().catch(() => ({}));
          this.terminal = true;
          throw new Error(typeof body.detail === 'string' ? body.detail : `评测请求失败 (${response.status})`);
        }
        if (!response.body) throw new Error('浏览器未提供流式响应。');
        reader = response.body.getReader();
        const decoder = new TextDecoder(); const frames = new SSEDecoder(event => this.handle(event));
        while (true) {
          const {value, done} = await reader.read();
          frames.feed(decoder.decode(value, {stream: !done}), done);
          if (done || this.terminal) break;
        }
        if (!this.report) throw new Error('进度连接已中断。');
        return this.report;
      } catch (error) {
        const suffix = this.terminal ? '' : ` 后台评测可能仍在运行；请稍后重新打开公开评测查看历史报告${this.reportId ? '（' + this.reportId + '）' : ''}，不要立即重复提交。`;
        this.status.textContent = this.terminal ? '评测未完成' : '进度连接中断';
        for (const call of this.calls.values()) {
          if (call.node.classList.contains('streaming')) call.note.textContent = '响应未确认完成 · 部分正文不能作为有效产物';
        }
        this.message('运行提示', error.message + suffix, 'error');
        throw error;
      } finally {
        this.terminal = true; clearInterval(timer); tick();
        for (const call of this.calls.values()) call.node.classList.remove('streaming');
        if (reader) { await reader.cancel().catch(() => {}); reader.releaseLock(); }
      }
    }
  }
  if (typeof module !== 'undefined') module.exports = {SSEDecoder, preview};
  root.BenchmarkChat = BenchmarkChat;
})(typeof window === 'undefined' ? globalThis : window);
