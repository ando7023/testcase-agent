function caseModuleOptions(selected = '') {
  return flattenModules(state.project?.module_tree?.modules || []).map(module =>
    `<option value="${escapeHtml(module.id)}" ${module.id === selected ? 'selected' : ''}>${escapeHtml(module.id)} · ${escapeHtml(module.name)}</option>`
  ).join('');
}

function caseConversationPanel() {
  const messages = state.project.case_conversation || [];
  const versions = (state.project.case_versions || []).slice().reverse().slice(0, 8);
  return `
    <aside class="module-memory-panel case-memory-panel">
      <div class="module-panel-heading"><div><span class="eyebrow">MEMORY</span><h2>用例协作会话</h2></div><span>${messages.length} turns</span></div>
      <div class="module-chat-log">${messages.map(message => `
        <div class="module-message ${escapeHtml(message.role)}">
          <small>${message.role === 'user' ? 'QA' : 'CASE AGENT'} · ${escapeHtml(message.mode)}</small>
          <p>${escapeHtml(message.content)}</p>
        </div>`).join('') || '<div class="module-message assistant"><p>生成用例后，可以继续补充场景、指定模块重生成，或通过对话调整用例。</p></div>'}</div>
      <form id="case-chat-form" class="module-chat-form">
        <label>作用模块<select id="case-chat-target"><option value="">全部用例</option>${caseModuleOptions()}</select></label>
        <label>修改要求<textarea id="case-chat-input" required placeholder="例如：补充重复回调、乱序消息和终态不可回退的断言"></textarea></label>
        <button class="button primary" type="submit">发送修改</button>
      </form>
      <div class="module-version-list">
        <div class="module-panel-heading"><h3>版本历史</h3><span>${state.project.case_versions?.length || 0} versions</span></div>
        ${versions.map(version => `<div class="module-version"><div><strong>${escapeHtml(version.id)}</strong><span>${escapeHtml(version.mode)} · ${version.cases?.length || 0} cases</span></div><button class="button ghost restore-case-version" data-version="${escapeHtml(version.id)}">恢复</button></div>`).join('') || '<p class="module-empty">暂无版本快照</p>'}
      </div>
    </aside>`;
}

function caseProgressTree(cases) {
  const moduleNames = Object.fromEntries(flattenModules(state.project?.module_tree?.modules || []).map(item => [item.id, item.name]));
  const grouped = {};
  (cases || []).forEach(item => {
    (grouped[item.module_id] ||= []).push(item);
  });
  return Object.entries(grouped).map(([moduleId, items]) => `
    <div class="case-progress-module">
      <strong>${escapeHtml(moduleNames[moduleId] || moduleId)}</strong><span>${items.length} cases</span>
      ${items.map(item => `<p><b>✓</b>${escapeHtml(item.title)}</p>`).join('')}
    </div>`).join('');
}

function renderCases() {
  const cases = state.project.cases || [];
  if (!cases.length) return;
  $('#cases-view').innerHTML = `
    ${humanAcceptanceBanner()}
    <div class="section-intro"><span class="eyebrow">CASE AGENT · CONVERSATION MEMORY · VERSIONED OUTPUT</span><h1>生成、补充，再通过对话打磨用例。</h1><p>支持全量、重生成、继续生成和指定模块生成；每轮会话保存短期 Memory 与完整用例快照，修改后独立评审结果自动失效。</p></div>
    <div class="module-modebar case-modebar">
      <div class="module-mode-actions">
        <button class="button ghost case-operation" data-mode="regenerate">重新生成</button>
        <button class="button ghost case-operation" data-mode="continue">继续生成</button>
      </div>
      <div class="module-target-action">
        <select id="case-target-select" aria-label="指定模块">${caseModuleOptions()}</select>
        <button class="button secondary case-operation" data-mode="targeted">生成指定模块</button>
      </div>
      <span id="case-stream-status">${escapeHtml(state.caseStream || '会话和用例版本已持久化')}</span>
    </div>
    <div class="module-workspace case-workspace">
      <section class="module-tree-pane case-results-pane">
        <div class="case-progress-tree" id="case-stream-tree">${caseProgressTree(cases)}</div>
        <div class="mindmap-actions">
          <button class="button ghost" onclick="toggleMindMap('case-mindmap', 'cases', this)">脑图视图</button>
          <a class="button ghost" href="/api/projects/${state.project.id}/export/xmind">导出 XMind</a>
          <a class="button ghost" href="/api/projects/${state.project.id}/export/mindmap">导出脑图 JSON</a>
          <span>按模块展开详细用例，步骤节点下挂载可观察预期。</span>
        </div>
        <div class="mindmap-shell" id="case-mindmap" hidden></div>
        ${reactTracePanel('case_generation')}
        ${caseTable(cases)}
        <div class="section-actions"><span>对话修改会清空旧评审，确保质量 Agent 检查最新版本。</span><div><a class="button ghost" href="/api/projects/${state.project.id}/export/csv">导出 CSV</a><button class="button primary" id="review-cases">运行独立评审</button></div></div>
      </section>
      ${caseConversationPanel()}
    </div>`;
  $('#review-cases').addEventListener('click', () => runAction(`/api/projects/${state.project.id}/review`, '评审 Agent 正在核对领域覆盖与证据…'));
  document.querySelectorAll('.case-operation').forEach(button => button.addEventListener('click', () => {
    const mode = button.dataset.mode;
    const target = mode === 'targeted' ? $('#case-target-select').value : '';
    runCaseOperation(mode, '', target);
  }));
  $('#case-chat-form').addEventListener('submit', event => {
    event.preventDefault();
    const instruction = $('#case-chat-input').value.trim();
    if (instruction) runCaseOperation('chat', instruction, $('#case-chat-target').value);
  });
  document.querySelectorAll('.restore-case-version').forEach(button => button.addEventListener('click', () =>
    runAction(`/api/projects/${state.project.id}/cases/restore`, '正在恢复用例版本…', {method:'POST', body:JSON.stringify({version_id:button.dataset.version})})
  ));
  bindCaseFeedback('#cases-view');
}

function renderCaseStream(cases) {
  const container = $('#case-stream-tree');
  if (container) container.innerHTML = caseProgressTree(cases);
}

function runCaseOperation(mode, instruction = '', targetModuleId = '') {
  if (!state.project) return;
  setBusy(true, '正在连接用例生成 Agent…');
  state.caseStream = '';
  let streamedChars = 0;
  let streamedCases = [];
  let completed = false;
  let opened = false;
  const fallback = async () => {
    if (completed) return;
    try {
      state.caseStream = 'WebSocket 不可用，正在切换普通请求…';
      setBusy(true, state.caseStream);
      const project = await api(`/api/projects/${state.project.id}/cases/generate`, {method:'POST', body:JSON.stringify({mode, instruction, target_module_id:targetModuleId})});
      completed = true; state.project = project;
      setBusy(false); renderProject(); await loadProjects();
    } catch (error) {
      setBusy(false); notify(error.message || '用例生成失败');
    }
  };
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const socket = new WebSocket(`${protocol}//${location.host}/ws/projects/${state.project.id}/cases`);
  const timeout = window.setTimeout(() => {
    if (!completed && !opened) { socket.close(); fallback(); }
  }, 8000);
  socket.addEventListener('open', () => {
    opened = true; window.clearTimeout(timeout);
    socket.send(JSON.stringify({mode, instruction, target_module_id:targetModuleId}));
  });
  const refreshCaseProject = async () => {
    try {
      const project = await api(`/api/projects/${state.project.id}`);
      if (!project.cases?.length) { setBusy(false); notify('用例 Agent 未返回用例，请重试'); return; }
      completed = true; state.project = project;
      socket.close(); setBusy(false); renderProject(); await loadProjects();
    } catch (error) {
      setBusy(false); notify(error.message || '用例结果刷新失败');
    }
  };
  socket.addEventListener('message', async event => {
    const message = JSON.parse(event.data);
    if (message.type === 'stage') {
      state.caseStream = message.message || '';
      setBusy(true, state.caseStream);
      if ($('#case-stream-status')) $('#case-stream-status').textContent = state.caseStream;
      if (message.stage === 'persisted') await refreshCaseProject();
    } else if (message.type === 'delta') {
      streamedChars += message.delta.length;
      state.caseStream = `LLM 已流式返回 ${streamedChars} 字符`;
      if ($('#case-stream-status')) $('#case-stream-status').textContent = state.caseStream;
    } else if (message.type === 'case') {
      streamedCases.push(message.case);
      state.caseStream = `已解析 ${streamedCases.length} / ${message.total} 条用例`;
      if ($('#case-stream-status')) $('#case-stream-status').textContent = state.caseStream;
      renderCaseStream(streamedCases);
    } else if (message.type === 'complete') {
      completed = true; state.project = message.project;
      socket.close(); setBusy(false); renderProject(); await loadProjects();
    } else if (message.type === 'error') {
      socket.close(); setBusy(false); notify(message.message || '用例生成失败');
    }
  });
  socket.addEventListener('error', () => {
    if (!opened && !completed) fallback();
    else if (!completed) { setBusy(false); notify('用例 Agent 连接中断，请重试'); }
  });
  socket.addEventListener('close', () => {
    if (!completed && opened) refreshCaseProject();
  });
}
