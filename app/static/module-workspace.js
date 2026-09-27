function flattenModules(modules, result = []) {
  (modules || []).forEach(module => {
    result.push(module);
    flattenModules(module.children || [], result);
  });
  return result;
}

function moduleTargetOptions(selected = '') {
  return flattenModules(state.project?.module_tree?.modules || []).map(module =>
    `<option value="${escapeHtml(module.id)}" ${module.id === selected ? 'selected' : ''}>${escapeHtml(module.id)} · ${escapeHtml(module.name)}</option>`
  ).join('');
}

function moduleRows(modules, depth = 0) {
  return (modules || []).map(module => `
    <div class="module-row" data-id="${escapeHtml(module.id)}" style="--module-depth:${depth}">
      <span class="module-id">${escapeHtml(module.id)}</span>
      <label>模块名称<input class="module-name" value="${escapeHtml(module.name)}" ${state.project.module_tree.confirmed ? 'disabled' : ''}></label>
      <label>测试目标<textarea class="module-objective" ${state.project.module_tree.confirmed ? 'disabled' : ''}>${escapeHtml(module.objective)}</textarea></label>
      <div class="module-meta">${(module.requirement_ids || []).map(id => `<span class="tag">${escapeHtml(id)}</span>`).join('')}${(module.risks || []).map(risk => `<span class="tag risk">${escapeHtml(risk)}</span>`).join('')}</div>
    </div>
    ${moduleRows(module.children || [], depth + 1)}
  `).join('');
}

function moduleConversationPanel() {
  const messages = state.project.module_conversation || [];
  const versions = (state.project.module_versions || []).slice().reverse().slice(0, 8);
  return `
    <aside class="module-memory-panel">
      <div class="module-panel-heading"><div><span class="eyebrow">MEMORY</span><h2>模块协作会话</h2></div><span>${messages.length} turns</span></div>
      <div class="module-chat-log">${messages.map(message => `
        <div class="module-message ${escapeHtml(message.role)}">
          <small>${message.role === 'user' ? 'QA' : 'MODULE AGENT'} · ${escapeHtml(message.mode)}</small>
          <p>${escapeHtml(message.content)}</p>
        </div>`).join('') || '<div class="module-message assistant"><p>生成模块后，可以连续提出拆分、合并、补充和调整要求。</p></div>'}</div>
      <form id="module-chat-form" class="module-chat-form">
        <label>作用模块<select id="module-chat-target"><option value="">整棵模块树</option>${moduleTargetOptions()}</select></label>
        <label>修改要求<textarea id="module-chat-input" required placeholder="例如：把审核流程拆成 DeskAgent 和 Librarian 两个子模块，并补充越权风险"></textarea></label>
        <button class="button primary" type="submit">发送修改</button>
      </form>
      <div class="module-version-list">
        <div class="module-panel-heading"><h3>版本历史</h3><span>${state.project.module_versions?.length || 0} versions</span></div>
        ${versions.map(version => `<div class="module-version"><div><strong>${escapeHtml(version.id)}</strong><span>${escapeHtml(version.mode)} · ${escapeHtml(version.created_at)}</span></div><button class="button ghost restore-module-version" data-version="${escapeHtml(version.id)}">恢复</button></div>`).join('') || '<p class="module-empty">暂无版本快照</p>'}
      </div>
    </aside>`;
}

function renderModules() {
  const tree = state.project.module_tree;
  if (!tree) return;
  const moduleReview = state.project.module_review;
  $('#modules-view').innerHTML = `
    <div class="section-intro"><span class="eyebrow">MODULE AGENT · CONVERSATION MEMORY · HUMAN GATE</span><h1>生成、讨论，再确认模块边界。</h1><p>支持全量、重生成、继续生成和指定模块生成；每轮对话都会保存会话与模块快照，并重新运行 Critic。</p></div>
    <div class="module-modebar">
      <div class="module-mode-actions">
        <button class="button ghost module-operation" data-mode="regenerate">重新生成</button>
        <button class="button ghost module-operation" data-mode="continue">继续生成</button>
      </div>
      <div class="module-target-action">
        <select id="module-target-select" aria-label="指定模块">${moduleTargetOptions()}</select>
        <button class="button secondary module-operation" data-mode="targeted">生成指定模块</button>
      </div>
      <span id="module-stream-status">${escapeHtml(state.moduleStream || '会话和模块版本已持久化')}</span>
    </div>
    <div class="module-workspace">
      <section class="module-tree-pane">
        <div class="mindmap-actions">
          <button class="button ghost" onclick="toggleMindMap('module-mindmap', 'modules', this)">脑图视图</button>
          <a class="button ghost" href="/api/projects/${state.project.id}/xmind/modules">导出 XMind</a>
          <span>层级模块、目标、风险和需求关联</span>
        </div>
        <div class="mindmap-shell" id="module-mindmap" hidden></div>
        ${moduleReview ? `<div class="score-layout"><div class="score-box"><strong>${moduleReview.score}</strong><span>模块质量 / 100</span><small>自动优化 ${moduleReview.rounds || 0} 轮</small></div><div><h3>模块发现 · ${moduleReview.findings.length}</h3>${moduleReview.findings.map(item => `<div class="finding"><b>${escapeHtml(item.severity)}</b><span>${escapeHtml(item.message)}</span></div>`).join('') || '<div class="finding"><b>PASS</b><span>模块结构检查通过</span></div>'}</div></div>` : ''}
        <div class="module-list">${moduleRows(tree.modules)}</div>
        <div class="section-actions"><span>${tree.confirmed ? '模块树已确认；继续修改会重新打开人工门禁。' : '确认后才能进入用例生成。'}</span>
          ${tree.confirmed ? '<button class="button primary" id="generate-cases">生成测试用例</button>' : '<button class="button primary" id="confirm-modules">确认模块树</button>'}
        </div>
      </section>
      ${moduleConversationPanel()}
    </div>`;
  if (tree.confirmed) $('#generate-cases').addEventListener('click', () => runCaseOperation('full'));
  else $('#confirm-modules').addEventListener('click', confirmModules);
  document.querySelectorAll('.module-operation').forEach(button => button.addEventListener('click', () => {
    const mode = button.dataset.mode;
    const target = mode === 'targeted' ? $('#module-target-select').value : '';
    runModuleOperation(mode, '', target);
  }));
  $('#module-chat-form').addEventListener('submit', event => {
    event.preventDefault();
    const instruction = $('#module-chat-input').value.trim();
    if (instruction) runModuleOperation('chat', instruction, $('#module-chat-target').value);
  });
  document.querySelectorAll('.restore-module-version').forEach(button => button.addEventListener('click', () =>
    runAction(`/api/projects/${state.project.id}/modules/restore`, '正在恢复模块版本并重新评审…', {method:'POST', body:JSON.stringify({version_id:button.dataset.version})})
  ));
}

function collectEditedModules(modules) {
  const rows = [...document.querySelectorAll('.module-row')];
  return modules.map(module => {
    const row = rows.find(item => item.dataset.id === module.id);
    return {...module, name: row.querySelector('.module-name').value, objective: row.querySelector('.module-objective').value, children: collectEditedModules(module.children || [])};
  });
}

async function confirmModules() {
  const modules = collectEditedModules(state.project.module_tree.modules);
  await runAction(`/api/projects/${state.project.id}/modules/confirm`, '正在固化模块边界…', {method:'PUT', body:JSON.stringify({modules})});
}

function runModuleOperation(mode, instruction = '', targetModuleId = '') {
  if (!state.project) return;
  setBusy(true, '正在连接模块生成 Agent…');
  state.moduleStream = '';
  let streamedChars = 0;
  let completed = false;
  let opened = false;
  const fallback = async () => {
    if (completed) return;
    try {
      state.moduleStream = 'WebSocket 不可用，正在切换普通请求…';
      setBusy(true, state.moduleStream);
      const project = await api(`/api/projects/${state.project.id}/modules/generate`, { method: 'POST', body: JSON.stringify({mode, instruction, target_module_id:targetModuleId}) });
      completed = true; state.project = project;
      setBusy(false); renderProject(); await loadProjects();
    } catch (error) {
      setBusy(false); notify(error.message || '模块生成失败');
    }
  };
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const socket = new WebSocket(`${protocol}//${location.host}/ws/projects/${state.project.id}/modules`);
  const timeout = window.setTimeout(() => {
    if (!completed && !opened) { socket.close(); fallback(); }
  }, 8000);
  socket.addEventListener('open', () => {
    opened = true; window.clearTimeout(timeout);
    socket.send(JSON.stringify({mode, instruction, target_module_id:targetModuleId}));
  });
  const refreshModuleProject = async () => {
    try {
      const project = await api(`/api/projects/${state.project.id}`);
      if (!project.module_tree) { setBusy(false); notify('模块 Agent 未返回模块树，请重试'); return; }
      completed = true; state.project = project;
      socket.close(); setBusy(false); renderProject(); await loadProjects();
    } catch (error) {
      setBusy(false); notify(error.message || '模块结果刷新失败');
    }
  };

  socket.addEventListener('message', async event => {
    const message = JSON.parse(event.data);
    if (message.type === 'stage') {
      state.moduleStream = message.message || '';
      setBusy(true, state.moduleStream);
      if ($('#module-stream-status')) $('#module-stream-status').textContent = state.moduleStream;
      if (message.stage === 'persisted') await refreshModuleProject();
    } else if (message.type === 'delta') {
      streamedChars += message.delta.length;
      state.moduleStream = `LLM 已流式返回 ${streamedChars} 字符`;
      if ($('#module-stream-status')) $('#module-stream-status').textContent = state.moduleStream;
    } else if (message.type === 'complete') {
      completed = true;
      state.project = message.project;
      socket.close(); setBusy(false); renderProject(); await loadProjects();
    } else if (message.type === 'error') {
      socket.close(); setBusy(false); notify(message.message || '模块生成失败');
    }
  });
  socket.addEventListener('error', () => {
    if (!opened && !completed) fallback();
    else if (!completed) { setBusy(false); notify('模块 Agent 连接中断，请重试'); }
  });
  socket.addEventListener('close', () => {
    if (!completed && opened) refreshModuleProject();
  });
}
