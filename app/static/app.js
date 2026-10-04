const state = { project: null, projects: [], parsedDocument: null, traceRuns: [], benchmarkCatalog: null };
const phaseOrder = ['draft', 'analyzed', 'modules_planned', 'modules_confirmed', 'cases_generated', 'reviewed'];

const $ = (selector) => document.querySelector(selector);
const escapeHtml = (value = '') => String(value).replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));

async function api(path, options = {}) {
  const response = await fetch(path, { headers: {'Content-Type': 'application/json'}, ...options });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `请求失败 (${response.status})`);
  }
  return response.json();
}

function setBusy(value, label = 'Agent working...') {
  $('#busy').hidden = !value;
  $('#busy p').textContent = label;
}

function notify(message) {
  const element = $('#notice');
  element.textContent = message;
  element.hidden = false;
  setTimeout(() => { element.hidden = true; }, 5000);
}

function setView(viewName) {
  document.querySelectorAll('.view').forEach(view => view.classList.toggle('active', view.id === `${viewName}-view`));
  const viewPhase = {draft: 'draft', analysis: 'analyzed', modules: 'modules_planned', cases: 'cases_generated', review: 'reviewed'}[viewName];
  document.querySelectorAll('.pipeline-step').forEach(step => step.classList.toggle('active', step.dataset.phase === viewPhase));
}

function phaseView(phase) {
  if (phase === 'draft') return 'draft';
  if (phase === 'analyzed') return 'analysis';
  if (phase === 'modules_planned' || phase === 'modules_confirmed') return 'modules';
  if (phase === 'cases_generated') return 'cases';
  return 'review';
}

function updatePipeline() {
  const currentIndex = phaseOrder.indexOf(state.project?.phase || 'draft');
  document.querySelectorAll('.pipeline-step').forEach(step => {
    const mapped = step.dataset.phase === 'modules_planned' ? 2 : phaseOrder.indexOf(step.dataset.phase);
    step.classList.toggle('done', mapped < currentIndex);
  });
}

async function loadProjects() {
  state.projects = await api('/api/projects');
  const phaseLabels = {draft: '需求输入', analyzed: '需求理解', modules_planned: '模块规划', modules_confirmed: '模块已调整', cases_generated: '用例生成', reviewed: '已评审'};
  $('#project-list').innerHTML = state.projects.map(project => `
    <button type="button" class="project-item ${state.project?.id === project.id ? 'active' : ''}" data-id="${escapeHtml(project.id)}" title="${escapeHtml(project.title)}" ${state.project?.id === project.id ? 'aria-current="page"' : ''}>
      <strong>${escapeHtml(project.title)}</strong><span>${escapeHtml(phaseLabels[project.phase] || project.phase)} · ${escapeHtml(project.updated_at.slice(0, 10))}</span>
    </button>`).join('') || '<div class="compact-item">暂无任务，点击 ＋ 新建</div>';
  document.querySelectorAll('.project-item').forEach(button => button.addEventListener('click', () => openProject(button.dataset.id)));
}

async function openProject(id) {
  state.project = await api(`/api/projects/${id}`);
  renderProject();
  if ($('#trace-panel').classList.contains('open')) await loadTraces();
  await loadRagDatasets();
loadProjects();
}

function renderProject() {
  if (!state.project) return;
  renderAnalysis();
  renderModules();
  renderCases();
  renderReview();
  updatePipeline();
  setView(phaseView(state.project.phase));
  document.dispatchEvent(new Event('project-rendered'));
}

function reactTracePanel(agentName) {
  const traces = (state.project?.traces || []).filter(item => item.agent === agentName);
  const trace = traces.length ? traces[traces.length - 1] : null;
  if (!trace) return '';
  const calls = trace.tool_calls || [];
  const callHtml = calls.map(call => '<div class="react-call"><strong>' + escapeHtml(call.tool) + '</strong><span>' + escapeHtml(call.reason || '') + '</span><small>' + escapeHtml(call.status) + ' · ' + escapeHtml(JSON.stringify(call.arguments || {})) + '</small></div>').join('');
  return '<div class="react-panel"><div class="react-panel-heading"><div><span class="eyebrow">REACT TOOL LOOP</span><h3>' + escapeHtml(agentName) + '</h3></div><span>' + (trace.react_steps || 0) + ' / 4 steps</span></div><div class="react-call-list">' + (callHtml || '<div class="react-call"><span>本轮未调用工具，Agent 直接结束研究。</span></div>') + '</div></div>';
}
function renderAnalysis() {
  const analysis = state.project.analysis;
  if (!analysis) return;
  const interfaces = analysis.interfaces || [];
  const transitions = analysis.state_transitions || [];
  const events = analysis.events || [];
  const evidence = analysis.retrieved_evidence_ids || [];
  $('#analysis-view').innerHTML = `
    <div class="section-intro"><span class="eyebrow">REQUIREMENT AGENT + RAG</span><h1>需求已被还原成可测试的工单契约。</h1><p>${escapeHtml(analysis.summary)}</p></div>
    <div class="fact-band">
      <div><strong>${analysis.atomic_requirements.length}</strong><span>原子需求</span></div>
      <div><strong>${interfaces.length}</strong><span>接口契约</span></div>
      <div><strong>${transitions.length}</strong><span>状态转换</span></div>
      <div><strong>${evidence.length}</strong><span>RAG 证据</span></div>
    </div>
    ${reactTracePanel('requirement_understanding')}
    <div class="summary-grid">
      <div>
        <div class="section-block"><h3>原子需求 · ${analysis.atomic_requirements.length}</h3>
          <div class="table-wrap"><table class="data-table"><thead><tr><th>ID</th><th>陈述</th><th>来源类型</th></tr></thead><tbody>
            ${analysis.atomic_requirements.map(req => `<tr><td>${req.id}</td><td>${escapeHtml(req.statement)}</td><td><small>${escapeHtml(req.category)}</small></td></tr>`).join('')}
          </tbody></table></div>
        </div>
      </div>
      <div>
        <div class="section-block"><h3>工单与角色</h3><div class="tag-list">${(analysis.ticket_types || []).concat(analysis.actors || []).map(item => `<span class="tag">${escapeHtml(item)}</span>`).join('')}</div></div>
        <div class="section-block"><h3>风险提示</h3><div class="tag-list">${analysis.risk_hints.map(item => `<span class="tag risk">${escapeHtml(item)}</span>`).join('')}</div></div>
        <div class="section-block"><h3>RAG 证据</h3><div class="evidence-list">${evidence.map(id => `<span>${escapeHtml(id)}</span>`).join('')}</div></div>
        <div class="section-block"><h3>待确认项</h3><ul class="ambiguity-list">${analysis.ambiguities.map(item => `<li>${escapeHtml(typeof item === 'string' ? item : `${item.id} · ${item.question}`)}</li>`).join('') || '<li>未识别到明显歧义</li>'}</ul></div>
      </div>
    </div>
    ${interfaces.length ? `<div class="domain-section"><div class="domain-heading"><span class="eyebrow">INTERFACES</span><h2>接口契约</h2></div><div class="contract-list">${interfaces.map(item => `<article><div><strong>${escapeHtml(item.name)}</strong><span>${escapeHtml(item.purpose)}</span></div><p><b>成功</b>${escapeHtml(item.success_condition)}</p><p><b>失败</b>${escapeHtml(item.failure_condition)}</p><small>${item.evidence_ids.map(id => escapeHtml(id)).join(' · ')}</small></article>`).join('')}</div></div>` : ''}
    ${transitions.length ? `<div class="domain-section"><div class="domain-heading"><span class="eyebrow">STATE MACHINE</span><h2>审核状态机</h2></div><div class="transition-list">${transitions.map(item => `<div><span>${escapeHtml(item.from_state)}</span><b>${escapeHtml(item.action)}</b><span class="${item.terminal ? 'terminal' : ''}">${escapeHtml(item.to_state)}</span></div>`).join('')}</div></div>` : ''}
    ${events.length ? `<div class="domain-section"><div class="domain-heading"><span class="eyebrow">EVENTS</span><h2>事件契约</h2></div>${events.map(item => `<div class="event-line"><strong>${escapeHtml(item.topic)}</strong><span>event=${escapeHtml(item.event)}</span><span>关联键 ${escapeHtml(item.correlation_key)}</span>${item.statuses.map(status => `<i>${escapeHtml(status)}</i>`).join('')}</div>`).join('')}</div>` : ''}
    <div class="section-actions"><span>下一步将接口、状态、权限、消息和灰度拆成独立测试模块。</span><button class="button primary" id="plan-modules">生成工单测试模块</button></div>`;
  $('#plan-modules').addEventListener('click', () => runModuleOperation('full'));
}

function renderModules() {
  const tree = state.project.module_tree;
  if (!tree) return;
  const moduleReview = state.project.module_review;
  $('#modules-view').innerHTML = `
    <div class="section-intro"><span class="eyebrow">MODULE AGENT · CRITIQUE</span><h1>规划测试范围，自主生成用例。</h1><p>模块代表测试关注点而不是页面目录。Agentic 会在规划后继续生成；你也可以按需编辑模块。</p></div>
    <div class="mindmap-actions">
      <button class="button ghost" onclick="toggleMindMap('module-mindmap', 'modules', this)">脑图视图</button>
      <a class="button ghost" href="/api/projects/${state.project.id}/xmind/modules">导出 XMind</a>
      <span>按模块层级检查测试范围，展开节点查看目标、风险和需求关联。</span>
    </div>
    <div class="mindmap-shell" id="module-mindmap" hidden></div>
    ${moduleReview ? `<div class="score-layout"><div class="score-box"><strong>${moduleReview.score}</strong><span>模块质量 / 100</span><small>自动优化 ${moduleReview.rounds || 0} 轮</small></div><div><h3>模块发现 · ${moduleReview.findings.length}</h3>${moduleReview.findings.map(item => `<div class="finding"><b>${escapeHtml(item.severity)}</b><span>${escapeHtml(item.message)}</span></div>`).join('') || '<div class="finding"><b>PASS</b><span>模块结构检查通过</span></div>'}</div></div>` : ''}
    <div class="module-list">${tree.modules.map(module => `
      <div class="module-row" data-id="${module.id}">
        <span class="module-id">${module.id}</span>
        <label>模块名称<input class="module-name" value="${escapeHtml(module.name)}" ${tree.confirmed ? 'disabled' : ''}></label>
        <label>测试目标<textarea class="module-objective" ${tree.confirmed ? 'disabled' : ''}>${escapeHtml(module.objective)}</textarea></label>
        <div class="module-meta">${module.requirement_ids.map(id => `<span class="tag">${id}</span>`).join('')}${module.risks.map(risk => `<span class="tag risk">${escapeHtml(risk)}</span>`).join('')}</div>
      </div>`).join('')}</div>
    <div class="section-actions"><span>模块规划已就绪，无需人工确认即可生成。编辑后请先保存。</span>
      ${tree.confirmed ? '' : '<button class="button ghost" id="confirm-modules">保存并确认修改（可选）</button>'}
      <button class="button primary" id="generate-cases">生成测试用例</button>
    </div>`;
  $('#generate-cases').addEventListener('click', () => runAction(`/api/projects/${state.project.id}/cases`, '用例 Agent 正在组合测试 Skills…'));
  if (!tree.confirmed) $('#confirm-modules').addEventListener('click', confirmModules);
}

async function confirmModules() {
  const modules = state.project.module_tree.modules.map(module => {
    const row = document.querySelector(`.module-row[data-id="${module.id}"]`);
    return {...module, name: row.querySelector('.module-name').value, objective: row.querySelector('.module-objective').value};
  });
  await runAction(`/api/projects/${state.project.id}/modules/confirm`, '正在固化模块边界…', {method: 'PUT', body: JSON.stringify({modules})});
}

function renderCases() {
  const cases = state.project.cases || [];
  if (!cases.length) return;
  $('#cases-view').innerHTML = `
    ${humanAcceptanceBanner()}
    <div class="section-intro"><span class="eyebrow">CASE GENERATION AGENT</span><h1>${cases.length} 条可追溯工单用例已生成。</h1><p>每条用例都保留需求 ID、RAG 证据、风险标签和可观察的预期结果。逐条采纳 / 修改 / 拒绝会沉淀为 few-shot 示例与 badcase 知识。</p></div>
    <div class="section-actions"><span>使用独立评审 Agent 检查接口、状态、权限、消息与灰度覆盖。</span><div><a class="button ghost" href="/api/projects/${state.project.id}/export/csv">导出 CSV</a><button class="button primary" id="review-cases">运行独立评审</button></div></div>
    <div class="mindmap-actions">
      <button class="button ghost" onclick="toggleMindMap('case-mindmap', 'cases', this)">脑图视图</button>
      <a class="button ghost" href="/api/projects/${state.project.id}/export/xmind">导出 XMind</a>
      <a class="button ghost" href="/api/projects/${state.project.id}/export/mindmap">导出脑图 JSON</a>
      <span>按模块展开详细用例，步骤节点下直接挂载预期结果。</span>
    </div>
    <div class="mindmap-shell" id="case-mindmap" hidden></div>
    ${reactTracePanel('case_generation')}
    ${caseTable(cases)}`;
  $('#review-cases').addEventListener('click', () => runAction(`/api/projects/${state.project.id}/review`, '评审 Agent 正在核对领域覆盖与证据…'));
  bindCaseFeedback('#cases-view');
}

const HUMAN_STATUS_TEXT = {pending: '待评判', adopted: '已采纳', edited: '已修改', rejected: '已拒绝'};
function humanAcceptanceBanner() {
  const acceptance = state.project?.human_acceptance;
  if (!acceptance) return '';
  return `<div class="section-block"><h3>人工验收 · ${acceptance.status === 'accepted' ? '已全部验收' : '尚未全部验收'}</h3><p>当前正文：${acceptance.accepted}/${acceptance.total} 条已采纳，${acceptance.pending} 条待验收，${acceptance.rejected} 条已拒绝。正文修改后需重新验收；采纳不代表接口执行通过。</p></div>`;
}
const FIXABLE_CATEGORIES = ['case_type', 'module_coverage', 'assertion', 'requirement_coverage'];

function caseTable(cases) {
  return `<div class="table-wrap"><table class="data-table"><thead><tr><th>ID</th><th>用例</th><th>类型</th><th>步骤 / 预期</th><th>追溯</th><th>评审</th><th>人工反馈</th></tr></thead><tbody>
    ${cases.map(item => `<tr><td><span class="priority">${item.priority}</span><br><small>${item.id}</small>${item.generated_by === 'revision' ? '<br><small class="revision-mark">修复补充</small>' : ''}</td><td class="case-title"><strong>${escapeHtml(item.title)}</strong><br><small>${escapeHtml(item.module_id)} · 自动化 ${escapeHtml(item.automation_feasibility)}</small></td><td><span class="tag">${escapeHtml(item.case_type)}</span></td><td><ol class="case-steps">${item.steps.map(step => `<li>${escapeHtml(step.action)}<em>预期：${escapeHtml(step.expected)}</em></li>`).join('')}</ol></td><td>${item.requirement_ids.map(id => `<span class="tag">${id}</span>`).join('')}<br><small>${escapeHtml(item.source_evidence[0] || '')}</small></td><td class="review-${item.review_status}">${escapeHtml(item.review_status)}</td>${feedbackCell(item)}</tr>`).join('')}
  </tbody></table></div>`;
}

function feedbackCell(item) {
  const status = HUMAN_STATUS_TEXT[item.human_status] || item.human_status;
  return `<td class="feedback-cell" data-case="${item.id}">
    <span class="human-badge human-${item.human_status}">${status}</span>
    <div class="fb-actions">
      <button class="fb-btn" data-action="adopted" data-case="${item.id}">采纳</button>
      <button class="fb-btn" data-action="edited" data-case="${item.id}">修改</button>
      <button class="fb-btn" data-action="rejected" data-case="${item.id}">拒绝</button>
    </div></td>`;
}

function bindCaseFeedback(scope) {
  document.querySelectorAll(`${scope} .fb-btn`).forEach(button => button.addEventListener('click', () => {
    const caseId = button.dataset.case;
    if (button.dataset.action === 'adopted') return sendFeedback(caseId, 'adopted', '');
    if (button.dataset.action === 'rejected') {
      const reason = window.prompt('拒绝原因（描述"缺少/遗漏"的场景会归为覆盖类 badcase，自动沉淀知识与团队记忆）：');
      if (reason === null) return;
      return sendFeedback(caseId, 'rejected', reason);
    }
    openCaseEditor(scope, caseId);
  }));
}

async function sendFeedback(caseId, action, reason, editedCase) {
  setBusy(true, '正在记录反馈并沉淀知识…');
  try {
    state.project = await api(`/api/projects/${state.project.id}/cases/${caseId}/feedback`, {
      method: 'POST',
      body: JSON.stringify({action, reason: reason || '', edited_case: editedCase || null}),
    });
    renderProject();
  } catch (error) { notify(error.message); }
  finally { setBusy(false); }
}

function openCaseEditor(scope, caseId) {
  const item = state.project.cases.find(c => c.id === caseId);
  const cell = document.querySelector(`${scope} .feedback-cell[data-case="${caseId}"]`);
  if (!item || !cell) return;
  cell.innerHTML = `<div class="case-editor">
    <label>标题<input class="edit-title" value="${escapeHtml(item.title)}"></label>
    <label>步骤（每行：操作 => 预期）<textarea class="edit-steps" rows="6">${escapeHtml(item.steps.map(step => `${step.action} => ${step.expected}`).join('\n'))}</textarea></label>
    <label>修改说明<input class="edit-reason" placeholder="用于 badcase 分类与知识沉淀"></label>
    <div class="fb-actions"><button class="button primary edit-save">保存并采纳</button><button class="button ghost edit-cancel">取消</button></div>
  </div>`;
  cell.querySelector('.edit-cancel').addEventListener('click', renderProject);
  cell.querySelector('.edit-save').addEventListener('click', () => {
    const steps = cell.querySelector('.edit-steps').value.split('\n').map(line => line.trim()).filter(Boolean).map(line => {
      const [action, expected = ''] = line.split('=>').map(part => part.trim());
      return {action, expected};
    });
    if (!steps.length) return notify('至少保留一个步骤');
    sendFeedback(caseId, 'edited', cell.querySelector('.edit-reason').value, {
      title: cell.querySelector('.edit-title').value,
      steps,
    });
  });
}

function renderReview() {
  const review = state.project.review;
  if (!review) {
    $('#review-view').innerHTML = `${humanAcceptanceBanner()}<div class="section-block"><p>当前用例尚无有效评审。请在用例工作区运行独立评审。</p></div>`;
    return;
  }
  const canRepair = item => [...FIXABLE_CATEGORIES, 'semantic'].includes(item.category) && !['suggestion', 'clarification'].includes(item.disposition);
  const findingLabel = item => item.disposition === 'clarification' ? '待澄清（阻塞）' : item.disposition === 'suggestion' && !['high', 'critical', 'error'].includes(item.severity) ? '建议（不阻塞）' : canRepair(item) ? '缺陷 · 可自动修复' : '待核验';
  const fixable = review.findings.filter(canRepair);
  $('#review-view').innerHTML = `
    ${humanAcceptanceBanner()}
    <div class="section-intro"><span class="eyebrow">REVIEW AGENT · CRITIQUE LOOP</span><h1>评审必须是可测量的，且必须被消费。</h1><p>评分来自需求、模块、领域类型、步骤断言和 RAG 证据的结构化检查；可修复的发现可以自动回流到生成 Agent。</p></div>
    <div class="score-layout">
      <div class="score-box"><strong>${review.score}</strong><span>质量评分 / 100</span>${review.added_case_ids?.length ? `<small class="revision-mark">修复补充 ${review.added_case_ids.length} 条</small>` : ''}</div>
      <div><h3>评审发现 · ${review.findings.length}</h3>${review.findings.map(item => `<div class="finding"><b>${escapeHtml(item.severity)}</b><span>${escapeHtml(item.message)} ${item.case_id ? `<small>· ${escapeHtml(item.case_id)}</small>` : ''}<small class="fixable-mark">${escapeHtml(findingLabel(item))}</small>${item.evidence ? `<small>依据：${escapeHtml(item.evidence)}</small>` : ''}</span></div>`).join('') || '<div class="finding"><b>PASS</b><span>未发现结构化质量问题</span></div>'}</div>
    </div>
    <div class="fact-band" id="metrics-band"><div><strong>--</strong><span>生成率</span></div><div><strong>--</strong><span>采纳率</span></div><div><strong>--</strong><span>修改率</span></div><div><strong>--</strong><span>badcase</span></div></div>
    ${evaluationBlock(state.project.evaluation)}
    <div class="section-actions"><span>评审状态已写回每条用例；人工反馈会实时更新指标并沉淀知识。</span><div>
      ${fixable.length ? '<button class="button primary" id="revise-cases">自动修复评审问题</button>' : ''}
      <button class="button ghost" id="evaluate-cases">运行离线评测</button>
      <a class="button ghost" href="/api/projects/${state.project.id}/export/json">导出 JSON</a><a class="button ${fixable.length ? 'ghost' : 'primary'}" href="/api/projects/${state.project.id}/export/csv">导出 CSV</a></div></div>
    ${caseTable(state.project.cases)}`;
  if (fixable.length) $('#revise-cases').addEventListener('click', () => runAction(`/api/projects/${state.project.id}/revise`, '修复 Agent 正在消费评审发现并补齐用例…'));
  $('#evaluate-cases').addEventListener('click', () => runAction(`/api/projects/${state.project.id}/evaluate`, '离线评测 Agent 正在对照黄金数据集…', {method:'POST', body:JSON.stringify({dataset_id:''})}));
  bindCaseFeedback('#review-view');
  loadMetrics();
}

function evaluationBlock(evaluation) {
  if (!evaluation) return '<div class="section-block"><h3>离线评测</h3><p>尚未运行对应工单类型的黄金数据集评测。</p></div>';
  const labels = {interface_recall:'接口召回', case_type_recall:'类型召回', required_term_recall:'关键规则', evidence_ratio:'证据完整', assertion_ratio:'断言质量', duplicate_ratio:'重复率', forbidden_term_ratio:'禁用术语'};
  return `<div class="section-block"><h3>离线评测 · ${escapeHtml(evaluation.dataset_id)} · ${evaluation.score}/100</h3><div class="tag-list">${Object.entries(evaluation.metrics).map(([key,value]) => `<span class="tag">${escapeHtml(labels[key] || key)} ${Math.round(value * 100)}%</span>`).join('')}</div>${evaluation.missing_items.length ? `<p>缺失：${escapeHtml(evaluation.missing_items.join('、'))}</p>` : '<p>黄金基准核心项已覆盖。</p>'}</div>`;
}

async function loadMetrics() {
  try {
    const metrics = await api(`/api/projects/${state.project.id}/metrics`);
    const badcaseTotal = Object.values(metrics.badcase_by_category || {}).reduce((sum, count) => sum + count, 0);
    const badcaseDetail = Object.entries(metrics.badcase_by_category || {}).map(([key, count]) => `${{coverage: '覆盖', quality: '质量', maintenance: '维护'}[key] || key} ${count}`).join(' · ');
    const percent = value => `${Math.round(value * 1000) / 10}%`;
    $('#metrics-band').innerHTML = `
      <div><strong>${percent(metrics.generation_rate)}</strong><span>生成率（完全采纳 / 已评判）</span></div>
      <div><strong>${percent(metrics.adoption_rate)}</strong><span>采纳率（采纳+修改 / 已评判）</span></div>
      <div><strong>${percent(metrics.modification_rate)}</strong><span>修改率</span></div>
      <div><strong>${badcaseTotal}</strong><span>badcase${badcaseDetail ? ` · ${badcaseDetail}` : ''} · 待评判 ${metrics.pending}</span></div>`;
  } catch (error) { /* metrics are best-effort in the UI */ }
}

async function runAction(path, label, options = {method: 'POST'}) {
  setBusy(true, label);
  try {
    state.project = await api(path, options);
    renderProject();
    await loadRagDatasets();
loadProjects();
  } catch (error) { notify(error.message); }
  finally { setBusy(false); }
}

$('#project-form').addEventListener('submit', async event => {
  event.preventDefault();
  setBusy(true, '检索 Agent 与需求 Agent 正在协作…');
  try {
    const project = await api('/api/projects', {method: 'POST', body: JSON.stringify({title: $('#project-title').value, context: $('#project-context').value, requirement: $('#project-requirement').value, source_documents: state.parsedDocument ? [state.parsedDocument] : []})});
    if (event.submitter?.value === 'agentic') {
      state.project = project;
      await window.startAgenticRun(project.id);
    } else {
      state.project = await api(`/api/projects/${project.id}/analyze`, {method: 'POST'});
    }
    renderProject();
    await loadRagDatasets();
loadProjects();
  } catch (error) { notify(error.message); }
  finally { setBusy(false); }
});

$('#project-document').addEventListener('change', async event => {
  const file = event.target.files[0];
  if (!file) return;
  setBusy(true, '文档解析 Agent 正在提取、筛选并压缩测试信息…');
  try {
    const form = new FormData(); form.append('file', file);
    const response = await fetch('/api/documents/parse', {method:'POST', body:form});
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || `解析失败 (${response.status})`);
    state.parsedDocument = body.document;
    $('#project-requirement').value = body.document.normalized_text;
    const extraction = body.document.extraction_summary || {};
const compression = body.document.compression_summary || {};
    const trace = body.trace || {};
    const ocrPages = extraction.ocr_pages || [];
    const ocrStatus = ocrPages.length ? ` · OCR 第${ocrPages.join('、')}页` : '';
    const fallbackReason = trace.mode === 'fallback'
      ? ` · LLM 失败，已回退本地提取${trace.error ? `（${trace.error.slice(0, 120)}）` : ''}`
      : '';
    const compressionStatus = compression.input_chars
      ? ` · ${compression.llm_used ? 'LLM 分层压缩' : '本地提取式压缩'} ${Math.max(0, Math.round((1 - compression.compression_ratio) * 100))}% · 删除 ${compression.excluded_sections || 0} 个无关章节${fallbackReason}`
      : '';
    $('#document-status').textContent = `${body.document.document_type} · ${body.document.chunks.length} 个证据块${ocrStatus}${compressionStatus}${body.document.warnings.length ? ` · ${body.document.warnings.join('；')}` : ''}`;
  } catch (error) { notify(error.message); }
  finally { setBusy(false); }
});

$('#new-project').addEventListener('click', () => { state.project = null; state.parsedDocument = null; $('#supervisor-panel').hidden = true; $('#project-document').value = ''; setView('draft'); updatePipeline(); loadRagDatasets();
loadProjects(); });
$('#knowledge-toggle').addEventListener('click', () => {
  $('#trace-panel').classList.remove('open');
  $('#benchmark-panel').classList.remove('open');
  $('#context-panel').classList.add('open');
  loadContext();
});
$('#context-close').addEventListener('click', () => $('#context-panel').classList.remove('open'));
$('#trace-toggle').addEventListener('click', () => {
  $('#context-panel').classList.remove('open');
  $('#benchmark-panel').classList.remove('open');
  $('#trace-panel').classList.add('open');
  loadTraces();
});
$('#trace-close').addEventListener('click', () => $('#trace-panel').classList.remove('open'));
$('#benchmark-toggle').addEventListener('click', () => {
  $('#context-panel').classList.remove('open');
  $('#trace-panel').classList.remove('open');
  $('#benchmark-panel').classList.add('open');
  if (benchmarkRunning) $('#benchmark-chat').scrollIntoView({block: 'nearest'});
  else loadBenchmarks();
});
$('#benchmark-close').addEventListener('click', () => $('#benchmark-panel').classList.remove('open'));

const benchmarkLabels = {
  worker_execution: '执行 Agent',
  supervisor_decision: '编排决策',
  scope_validation: '证据范围校验',
  schema_validation: '产物结构校验',
  repair_validation: '修复生效校验',
  requirement_understanding: '需求理解',
  module_planning: '模块规划',
  case_generation: '用例生成',
  quality_critic: '质量评审',
  flow_completion_rate: '流程完成率',
  quality_pass_rate: '质量门禁通过率（适用样本）',
  quality_assessed_count: '质量已评定样本数',
  quality_applicable_count: '质量门禁适用样本数',
  technical_failure_rate: '技术失败率',
  degraded_rate: '降级率',
  baseline_finding_count: '基线评审发现数（非误报率）',
  task_success_rate: '任务成功率',
  linked_test_recall: '关联测试召回',
  mean_gold_token_recall: '黄金用例词元召回',
  traceability_ratio: '需求可追溯率',
  assertion_ratio: '断言完整率',
  critic_score: '评审得分',
  actor_accuracy: '角色识别准确率',
  action_recall: '动作召回',
  outcome_recall: '结果召回',
  goal_recall: '目标召回',
  deliverable_module_recall: '交付物模块召回',
  extraction_recall: '文本提取召回',
  relevant_recall: '相关内容召回',
  compression_ratio: '保留文本比例',
  warning_rate: '解析告警率',
  defect_detection_recall: '缺陷检测召回',
  clean_false_positive_count: '干净样本误报数',
  clean_pass_rate: '干净样本通过率',
};

function benchmarkCount(dataset) {
  const fields = [
    dataset.sample_count ? `${dataset.sample_count} 样本` : '',
    dataset.retrieval_case_count ? `${dataset.retrieval_case_count} 检索样本` : '',
    dataset.positive_trace_count ? `${dataset.positive_trace_count} 条正向 Trace` : '',
    dataset.project_count ? `${dataset.project_count} 项目` : '',
    dataset.file_count ? `${dataset.file_count} 文件` : '',
  ].filter(Boolean);
  return fields.join(' · ') || '等待导入';
}

function renderBenchmarkDatasets(datasets) {
  const importable = new Set(['EBT-RAG-V1', 'STORYSEEK-V1', 'PUBLIC-SRS-V1']);
  $('#benchmark-ready-count').textContent = `${datasets.filter(item => item.ready).length} / ${datasets.length} 就绪`;
  $('#benchmark-datasets').innerHTML = datasets.map(dataset => `
    <article class="benchmark-dataset">
      <div class="benchmark-dataset-main">
        <div><strong>${escapeHtml(dataset.name)}</strong><span>${escapeHtml(dataset.dataset_id)} · ${escapeHtml(dataset.license || 'license unknown')}</span></div>
        <b class="benchmark-state ${dataset.ready ? 'ready' : ''}">${dataset.ready ? 'READY' : 'NOT IMPORTED'}</b>
      </div>
      <p>${escapeHtml((dataset.purpose || []).join(' · '))}</p>
      <div class="benchmark-dataset-foot"><span>${escapeHtml(benchmarkCount(dataset))}</span>${importable.has(dataset.dataset_id) ? `<button class="button ghost benchmark-import" type="button" data-dataset-id="${escapeHtml(dataset.dataset_id)}">${dataset.ready ? '刷新' : '导入'}</button>` : ''}</div>
    </article>`).join('');
  document.querySelectorAll('.benchmark-import').forEach(button => button.addEventListener('click', async () => {
    setBusy(true, `正在导入 ${button.dataset.datasetId} 公开数据…`);
    try {
      await api(`/api/benchmarks/${encodeURIComponent(button.dataset.datasetId)}/import`, {method: 'POST'});
      await loadBenchmarks();
      notify(`${button.dataset.datasetId} 已导入`);
    } catch (error) { notify(error.message); }
    finally { setBusy(false); }
  }));
}

function benchmarkMetricValue(key, value) {
  if (value === null || value === undefined) return '未评定';
  if (key.endsWith('_count')) return Number(value).toFixed(0);
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function renderBenchmarkReport(report) {
  const config = report.samples?.find(s => s.llm_config)?.llm_config;
  const knowledgeScope = report.knowledge_scope === 'ebt_sample' ? 'EBT样本级证据' : '知识仅来自样本';
  const configuration = config ? `${config.model} · ${config.stream ? '流式' : '非流式'} · reasoning=${config.reasoning_effort || '默认'} · 读取超时 ${config.timeout_seconds}s · 输出上限 ${config.max_tokens || '服务默认'} · ${knowledgeScope}` : '';
  const metrics = Object.entries(report.metrics || {}).map(([key, value]) => `
    <div><strong>${benchmarkMetricValue(key, value)}</strong><span>${escapeHtml(benchmarkLabels[key] || key)}</span></div>`).join('');
  const samples = (report.samples || []).map(sample => {
    const details = Object.entries(sample)
      .filter(([key, value]) => !['id', 'status', 'error', 'project'].includes(key) && (typeof value === 'number' || typeof value === 'boolean'))
      .slice(0, 3)
      .map(([key, value]) => `${benchmarkLabels[key] || key} ${typeof value === 'boolean' ? (value ? '是' : '否') : (value <= 1 ? `${(value * 100).toFixed(0)}%` : value)}`)
      .join(' · ');
    const failure = sample.technical_failure ? ` · 失败阶段 ${benchmarkLabels[sample.failure_stage] || sample.failure_stage || '未知'}${sample.failure_agent ? ' / ' + (benchmarkLabels[sample.failure_agent] || sample.failure_agent) : ''}${sample.failure_phase ? ' / ' + (benchmarkLabels[sample.failure_phase] || sample.failure_phase) : ''} · 错误 ${sample.failure_error_code || sample.error_code || '未知'}` : '';
    const outcomes = report.schema_version === 2
      ? `流程 ${sample.flow_completed ? '完成' : '未完成'} · 质量门禁 ${sample.quality_passed === null ? '未评定' : sample.quality_passed ? '通过' : '未通过'} · 技术失败 ${sample.technical_failure ? '是' : '否'} · 降级 ${sample.degraded ? '是' : '否'}${failure}` : details;
    const timing = (sample.call_diagnostics || []).filter(d => d.error_code).map(d => {
      const ms = v => v == null ? '未观测到' : `${(v / 1000).toFixed(2)}s`;
      return `调用 ${d.call} ${d.error_code} · 阶段 ${d.phase || '未知'} · 收到响应头 ${ms(d.connected_ms)} · 首事件 ${ms(d.first_event_ms)} · 最后接收 ${ms(d.last_receive_ms)} · 用时 ${ms(d.elapsed_ms)}`;
    }).join('\n');
    const gapLabels = {execution_detail: '执行前待补充', out_of_scope: '原文范围之外', behavior_blocker: '阻塞预期判断', unspecified: '待分类'};
    const readiness = {blocked: '存在阻塞', needs_preparation: '需补充执行条件', not_assessed: '未验收可执行性'};
    const gaps = [...(sample.clarification_items || []).map(g => ({kind: g.kind, text: g.question, reason: g.reason, ids: g.requirement_ids})),
      ...(sample.review_clarifications || []).map(g => ({kind: g.clarification_kind, text: g.message, reason: g.clarification_reason, ids: g.requirement_ids}))];
    const audits = sample.clarification_scope_review || [];
    const scopeAudit = audits.length ? `<details><summary>澄清范围复核（${audits.length} 项）</summary>${audits.map(a => `<p><b>${escapeHtml(a.clarification_id)}</b> · ${escapeHtml(gapLabels[a.original_kind] || a.original_kind)} → ${escapeHtml(gapLabels[a.kind] || a.kind)}<br>原判断：${escapeHtml(a.original_reason)}<br>复核依据：${escapeHtml(a.reason)}<br>引文：${escapeHtml(a.source_quote)}<br>证据：${escapeHtml((a.evidence_ids || []).join('、'))}</p>`).join('')}</details>` : '';
    const scope = sample.case_design_level ? `<small>设计层级：${sample.case_design_level === 'behavior' ? '行为级' : '面向执行'} · 执行准备：${escapeHtml(readiness[sample.execution_readiness] || '未评定')}</small>
      <details><summary>范围与执行准备（${gaps.length} 项）</summary>${gaps.map(g => `<p><b>${escapeHtml(gapLabels[g.kind] || '待分类')}</b> · ${escapeHtml((g.ids || []).join('、'))}<br>${escapeHtml(g.text)}<br>${escapeHtml(g.reason || '')}</p>`).join('') || '<p>未记录澄清项，不代表已完成环境与接口验收。</p>'}
      <p>受阻需求：${escapeHtml((sample.blocked_requirement_ids || []).join('、') || '无记录')}<br>尚无用例关联的需求：${escapeHtml((sample.uncovered_requirement_ids || []).join('、') || '无记录')}</p></details>${scopeAudit}` : '';
    const evidence = sample.knowledge_scope === 'ebt_sample' ? `<small>知识范围：当前 EBT 样本证据（${(sample.knowledge_document_ids || []).length} 份），未写入全局 RAG</small>` : '';
    return `<div class="benchmark-sample"><strong>${escapeHtml(sample.id)}</strong><span>${escapeHtml(sample.status)} · ${escapeHtml(outcomes)}</span>${evidence}${scope}${sample.question ? `<small>${escapeHtml(sample.question)}</small>` : ''}${sample.error || sample.error_code ? `<small>${escapeHtml(sample.error || sample.error_code)}</small>` : ''}${sample.run_id ? `<small>运行 ${escapeHtml(sample.run_id)} · ${sample.steps} 步 · ${escapeHtml(sample.run_status)}</small>` : ''}${timing ? `<small>${escapeHtml(timing)}</small>` : ''}</div>`;
  }).join('');
  $('#benchmark-result').innerHTML = `
    <div class="benchmark-score ${report.schema_version === 2 ? 'benchmark-score-v2' : ''}"><strong>${report.schema_version === 2 ? '分项' : Number(report.score || 0)}</strong><span>${escapeHtml(report.dataset_id)}<br>${report.sample_count || 0} samples · ${escapeHtml(report.mode || 'offline')} · ${escapeHtml(report.execution || '旧版固定流程')} · ${escapeHtml(report.status || '')}</span></div>
    <p class="benchmark-lead">${report.schema_version === 2 ? `${report.mode === 'offline' ? '离线规则回归，不代表模型能力。' : '模型评测；质量通过仅代表内部门禁，不代表独立业务验收。'} ${report.module_confirmation_required === false ? '模块规划后自主推进，无需人工确认。' : report.human_policy === 'simulate_confirm' ? '已选择模拟模块确认，不自动回答业务澄清或追加预算。' : '保留人工确认暂停。'} ${escapeHtml(report.metric_notes || '')}` : '旧版报告：综合分、passed 和成功率沿用旧口径，不能作为完整 Agentic 质量结论。'}</p>
    <div class="benchmark-metrics">${metrics}</div>
    ${configuration ? `<p class="benchmark-lead">实际配置：${escapeHtml(configuration)}</p>` : ''}
    ${report.clarification_policy ? `<p class="benchmark-lead">澄清策略：${report.clarification_policy === 'evidence_only' ? '按原文生成行为级用例；通过只表示行为设计门禁通过，不代表可直接执行。' : '严格澄清；通过不替代实际环境验收。'}</p>` : ''}
    ${samples ? `<details class="benchmark-samples"><summary>查看样本明细</summary>${samples}</details>` : ''}`;
}

function renderBenchmarkReports(reports) {
  $('#benchmark-reports').innerHTML = reports.map(report => `
    <button class="benchmark-report" type="button" data-report-id="${escapeHtml(report.report_id)}">
      <span><strong>${escapeHtml(report.dataset_id)}</strong><small>${escapeHtml(report.created_at || '')} · ${escapeHtml(report.mode || '')}</small></span>
      <b>${report.schema_version === 2 ? escapeHtml(report.execution || 'workflow') : '旧版 ' + Number(report.score || 0)}</b>
    </button>`).join('') || '<p class="benchmark-empty">运行一次评测后，报告会保存在这里。</p>';
  document.querySelectorAll('.benchmark-report').forEach(button => button.addEventListener('click', async () => {
    try {
      renderBenchmarkReport(await api(`/api/benchmarks/reports/${encodeURIComponent(button.dataset.reportId)}`));
    } catch (error) { notify(error.message); }
  }));
}

async function loadBenchmarks() {
  try {
    const selectedSuite = $('#benchmark-suite').value;
    const catalog = await api('/api/benchmarks');
    state.benchmarkCatalog = catalog;
    renderBenchmarkDatasets(catalog.datasets || []);
    $('#benchmark-suite').innerHTML = (catalog.suites || []).map(suite => `<option value="${escapeHtml(suite.id)}">${escapeHtml(suite.label)} · ${escapeHtml(suite.dataset_id)}</option>`).join('');
    if ((catalog.suites || []).some(suite => suite.id === selectedSuite)) $('#benchmark-suite').value = selectedSuite;
    renderBenchmarkReports(catalog.reports || []);
    updateBenchmarkOptions();
  } catch (error) {
    $('#benchmark-datasets').innerHTML = `<p class="benchmark-empty">${escapeHtml(error.message)}</p>`;
  }
}

function updateBenchmarkOptions() {
  const suite = (state.benchmarkCatalog?.suites || []).find(s => s.id === $('#benchmark-suite').value);
  const allowsAgentic = (suite?.executions || []).includes('agentic');
  $('#benchmark-execution').querySelector('[value="agentic"]').disabled = !allowsAgentic;
  if (!allowsAgentic) $('#benchmark-execution').value = 'workflow';
  $('#benchmark-clarification-policy').disabled = !allowsAgentic;
  $('#benchmark-max-steps').disabled = $('#benchmark-execution').value !== 'agentic';
  $('#benchmark-split').disabled = suite?.id !== 'storyseek_pipeline';
  $('#benchmark-limit').disabled = suite?.id === 'critic_mutation';
  $('#benchmark-execution-note').textContent = suite?.id === 'critic_mutation'
    ? '固定运行 4 个结构缺陷样本和 1 个基线；真实模式会调用模型评审。'
    : suite?.id === 'storyseek_pipeline' ? '评测止于需求理解和模块规划；完成模块评审后收尾。'
    : suite?.id === 'srs_document' ? '文档解析专项，不经过 Supervisor。'
    : 'Agentic 使用真实 Supervisor；每个样本独立运行，不自动追加预算。';
}
$('#benchmark-suite').addEventListener('change', updateBenchmarkOptions);
$('#benchmark-execution').addEventListener('change', updateBenchmarkOptions);

const benchmarkChat = new BenchmarkChat($('#benchmark-chat'));
let benchmarkRunning = false;
$('#benchmark-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (benchmarkRunning) return;
  const mode = $('#benchmark-mode').value;
  const payload = {
      suite: $('#benchmark-suite').value,
      split: $('#benchmark-split').value,
      limit: Number($('#benchmark-limit').value),
      mode,
      execution: $('#benchmark-execution').value,
      human_policy: 'pause', // Legacy API field; module confirmation is no longer required.
      clarification_policy: $('#benchmark-clarification-policy').disabled ? 'strict' : $('#benchmark-clarification-policy').value,
      max_steps: Number($('#benchmark-max-steps').value),
      stream: $('#benchmark-stream').value === 'true',
      reasoning_effort: $('#benchmark-reasoning').value,
      timeout_seconds: Number($('#benchmark-timeout').value),
  };
  const label = $('#benchmark-suite').selectedOptions[0]?.textContent || payload.suite;
  const controls = [...$('#benchmark-form').querySelectorAll('input, select, button')];
  const disabled = controls.map(control => control.disabled);
  benchmarkRunning = true;
  controls.forEach(control => { control.disabled = true; });
  $('#benchmark-run').textContent = '评测进行中…';
  $('#benchmark-settings').open = false;
  $('#benchmark-panel').classList.add('chat-active');
  $('#benchmark-result').replaceChildren();
  try {
    const report = await benchmarkChat.run(payload, label);
    renderBenchmarkReport(report);
    await loadBenchmarks();
  } catch (error) { notify(error.message); }
  finally {
    benchmarkRunning = false;
    controls.forEach((control, index) => { control.disabled = disabled[index]; });
    $('#benchmark-run').textContent = '运行 Benchmark';
    updateBenchmarkOptions();
  }
});

async function loadTraces() {
  const empty = $('#trace-empty');
  if (!state.project) {
    empty.hidden = false;
    $('#trace-list').innerHTML = '';
    $('#trace-detail').innerHTML = '';
    return;
  }
  empty.hidden = true;
  try {
    state.traceRuns = await api(`/api/projects/${state.project.id}/traces?limit=30`);
    const totals = state.traceRuns.reduce((acc, run) => {
      acc.spans += run.span_count || 0;
      acc.tokens += run.usage?.total_tokens || 0;
      acc.errors += run.error_count || 0;
      return acc;
    }, {spans: 0, tokens: 0, errors: 0});
    $('#trace-summary').innerHTML = `<div class="trace-metrics"><div><strong>${state.traceRuns.length}</strong><span>运行</span></div><div><strong>${totals.spans}</strong><span>Spans</span></div><div><strong>${totals.tokens}</strong><span>Tokens</span></div><div><strong>${totals.errors}</strong><span>错误</span></div></div>`;
    $('#trace-list').innerHTML = state.traceRuns.map((run, index) => `
      <button class="trace-run ${index === 0 ? 'active' : ''}" data-trace-id="${escapeHtml(run.trace_id)}">
        <span><b>${escapeHtml(run.operation)}</b><small>${escapeHtml(run.transport)} · ${escapeHtml(run.status)}</small></span>
        <span><b>${run.duration_ms}ms</b><small>${run.span_count} spans · ${run.usage?.total_tokens || 0} tokens</small></span>
      </button>`).join('') || '<div class="trace-empty">当前任务还没有新格式 Trace，执行一次分析或生成后即可查看。</div>';
    document.querySelectorAll('.trace-run').forEach(button => button.addEventListener('click', async () => {
      document.querySelectorAll('.trace-run').forEach(item => item.classList.toggle('active', item === button));
      await loadTraceDetail(button.dataset.traceId);
    }));
    if (state.traceRuns.length) await loadTraceDetail(state.traceRuns[0].trace_id);
  } catch (error) {
    $('#trace-list').innerHTML = `<div class="trace-empty">${escapeHtml(error.message)}</div>`;
  }
}

async function loadTraceDetail(traceId) {
  const run = await api(`/api/traces/${traceId}`);
  const byId = Object.fromEntries((run.spans || []).map(span => [span.id, span]));
  const depthOf = span => {
    let depth = 0;
    let parent = byId[span.parent_span_id];
    const seen = new Set();
    while (parent && !seen.has(parent.id) && depth < 8) {
      seen.add(parent.id);
      depth += 1;
      parent = byId[parent.parent_span_id];
    }
    return depth;
  };
  const rows = (run.spans || []).map(span => {
    const usage = span.usage || {};
    const tokens = usage.total_tokens ? `${usage.total_tokens} tok${usage.estimated ? ' 估算' : ''}` : '';
    const attributes = Object.entries(span.attributes || {}).map(([key, value]) => `${key}=${value}`).join(' · ');
    return `<details class="trace-span status-${escapeHtml(span.status)}" style="--trace-depth:${depthOf(span)}" ${span.kind === 'request' ? 'open' : ''}>
      <summary><i>${escapeHtml(span.kind)}</i><b>${escapeHtml(span.name)}</b><span>${span.duration_ms}ms${tokens ? ` · ${tokens}` : ''}</span></summary>
      <div class="trace-span-body">${attributes ? `<small>${escapeHtml(attributes)}</small>` : ''}${span.error ? `<p class="trace-error">${escapeHtml(span.error)}</p>` : ''}${span.input_summary ? `<p><b>Input</b> ${escapeHtml(span.input_summary)}</p>` : ''}${span.output_summary ? `<p><b>Output</b> ${escapeHtml(span.output_summary)}</p>` : ''}${(span.events || []).length ? `<p><b>Events</b> ${span.events.map(event => escapeHtml(event.name)).join(' · ')}</p>` : ''}</div>
    </details>`;
  }).join('');
  $('#trace-detail').innerHTML = `<div class="trace-detail-heading"><span class="eyebrow">TRACE DETAIL</span><b>${escapeHtml(run.trace_id)}</b><small>${escapeHtml(run.started_at)} · ${run.duration_ms}ms</small></div>${rows}`;
}

$('#knowledge-form').addEventListener('submit', async event => {
  event.preventDefault();
  try {
    await api('/api/knowledge', {method:'POST', body:JSON.stringify({title:$('#knowledge-title').value, content:$('#knowledge-content').value, doc_type:$('#knowledge-type').value, ticket_type:$('#knowledge-ticket-type').value, tags:[]})});
    event.target.reset(); await loadContext();
  } catch (error) { notify(error.message); }
});

$('#knowledge-upload-form').addEventListener('submit', async event => {
  event.preventDefault();
  const file = $('#knowledge-file').files[0];
  if (!file) return;
  setBusy(true, '正在解析并构建知识索引…');
  try {
    const form = new FormData();
    form.append('file', file);
    form.append('ticket_type', $('#knowledge-file-ticket-type').value);
    form.append('version', $('#knowledge-file-version').value);
    const response = await fetch('/api/knowledge/ingest', {method:'POST', body:form});
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || `入库失败 (${response.status})`);
    $('#knowledge-upload-status').innerHTML = `<div class="compact-item"><strong>已写入 ${body.changed} 个知识块</strong><span>${escapeHtml(body.document.filename)} · ${body.knowledge_documents.length} chunks · v${escapeHtml($('#knowledge-file-version').value)}</span></div>`;
    await loadContext();
  } catch (error) { notify(error.message); }
  finally { setBusy(false); }
});

$('#rag-search-form').addEventListener('submit', async event => {
  event.preventDefault();
  try {
    const result = await api('/api/rag/search', {method:'POST', body:JSON.stringify({query:$('#rag-query').value})});
    const trace = result.trace || {};
    const summary = `<div class="rag-summary"><b>${escapeHtml(result.retrieval_mode)}</b><span>${escapeHtml(trace.embedding_provider || '')} · 候选 ${result.candidate_count} · 过滤 ${result.filtered_count} · ${trace.latency_ms || 0}ms</span></div>`;
    const hits = result.hits.map(hit => `<div class="compact-item rag-hit"><strong>#${hit.rank} ${escapeHtml(hit.title)}</strong><span>${escapeHtml(hit.document_id)} · score ${Number(hit.score).toFixed(3)} · BM25 ${Number(hit.lexical_score).toFixed(3)} · vector ${Number(hit.vector_score).toFixed(3)}</span><small>${escapeHtml((hit.reasons || []).join(' · '))}</small></div>`).join('');
    $('#rag-result').innerHTML = summary + (hits || '<div class="compact-item">没有召回结果</div>');
  } catch (error) { notify(error.message); }
});

$('#rag-evaluate').addEventListener('click', async () => {
  try {
    const datasetId = $('#rag-dataset').value;
    const report = await api('/api/rag/evaluate', {method:'POST', body:JSON.stringify({dataset_id:datasetId, k:5})});
    $('#rag-result').innerHTML = `<div class="rag-summary"><b>${escapeHtml(report.dataset_id)}</b><span>Recall@5 ${(report.recall_at_k * 100).toFixed(1)}% · Hit Rate ${(report.hit_rate * 100).toFixed(1)}% · MRR ${report.mrr.toFixed(3)}</span><span>污染率 ${(report.contamination_rate * 100).toFixed(1)}% · 过期命中 ${(report.stale_hit_rate * 100).toFixed(1)}% · 样本 ${report.cases.length}</span></div>`;
  } catch (error) { notify(error.message); }
});

$('#ebt-import').addEventListener('click', async () => {
  setBusy(true, '正在下载并校验 EBT 数据集…');
  try {
    const status = await api('/api/ebt/import', {method:'POST'});
    await loadRagDatasets();
    $('#rag-dataset').value = 'EBT-RAG-V1';
    $('#rag-result').innerHTML = `<div class="rag-summary"><b>EBT 已导入</b><span>${status.artifact_count} 个制品 · ${status.positive_trace_count} 条需求用例 Trace · ${status.retrieval_case_count} 个评测问题</span><span>原始数据与工单知识库隔离存储</span></div>`;
  } catch (error) { notify(error.message); }
  finally { setBusy(false); }
});

async function loadRagDatasets() {
  const datasets = await api('/api/rag/datasets');
  $('#rag-dataset').innerHTML = datasets.map(item => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.id)} · ${item.cases.length} cases</option>`).join('');
}
$('#template-form').addEventListener('submit', async event => {
  event.preventDefault();
  try {
    await api('/api/scenario-templates/learn', {method:'POST', body:JSON.stringify({ticket_type:$('#template-ticket-type').value, content:$('#template-content').value, source:'manual_test_plan'})});
    event.target.reset(); await loadContext();
  } catch (error) { notify(error.message); }
});

$('#memory-form').addEventListener('submit', async event => {
  event.preventDefault();
  try {
    await api('/api/memory/rules', {method:'POST', body:JSON.stringify({rule:$('#memory-rule').value, ticket_type:$('#memory-ticket-type').value})});
    event.target.reset(); await loadContext();
  } catch (error) { notify(error.message); }
});

$('#memory-search-form').addEventListener('submit', async event => {
  event.preventDefault();
  try {
    const result = await api('/api/memory/search', {method:'POST', body:JSON.stringify({
      query:$('#memory-query').value,
      ticket_type:$('#memory-ticket-type').value,
      project_id:state.project?.id || '',
      agent_id:$('#memory-agent').value,
      top_k:5,
      token_budget:1000
    })});
    const saving = `${Math.round(Number(result.token_saving_ratio || 0) * 1000) / 10}%`;
    const summary = `<div class="rag-summary"><b>${escapeHtml(result.retrieval_mode)}</b><span>候选 ${result.candidate_count} · 召回 ${result.selected_count} · ${result.latency_ms}ms</span><span>Token ${result.estimated_full_tokens} → ${result.selected_tokens} · 节省 ${saving}</span></div>`;
    const hits = (result.hits || []).map(hit => {
      const memory = hit.memory || {};
      const scope = [memory.project_id && `project=${memory.project_id}`, memory.agent_id && `agent=${memory.agent_id}`, `ticket=${memory.ticket_type || 'COMMON'}`].filter(Boolean).join(' · ');
      return `<div class="compact-item rag-hit"><strong>${escapeHtml(memory.content)}</strong><span>${escapeHtml(memory.memory_type || '')} · score ${Number(hit.score || 0).toFixed(3)} · ${escapeHtml(scope)}</span><small>${escapeHtml((hit.reasons || []).join(' · '))}</small></div>`;
    }).join('');
    $('#memory-result').innerHTML = summary + (hits || '<div class="compact-item">没有召回相关 Memory</div>');
    await loadContext();
  } catch (error) { notify(error.message); }
});

async function loadContext() {
  const [knowledge, memory, templates] = await Promise.all([api('/api/knowledge'), api(`/api/memory?include_inactive=${Boolean($('#memory-show-history')?.checked)}`), api('/api/scenario-templates')]);
  if (window.renderChunkKnowledge) window.renderChunkKnowledge(knowledge);
  const records = memory.records || [];
  const typeCounts = Object.entries(memory.stats?.by_type || {}).map(([key,value]) => `${key} ${value}`).join(' · ');
  $('#memory-stats').innerHTML = `<div class="rag-summary"><b>${memory.stats?.total || 0} memories</b><span>${escapeHtml(typeCounts || '暂无长期记忆')}</span></div>`;
  const visibleMemories = records.slice().reverse();
  $('#memory-list').innerHTML = visibleMemories.map((item, index) => `<div class="compact-item"><strong>${escapeHtml(item.content)}</strong><span>${escapeHtml(item.memory_type)} · ${escapeHtml(item.status)} · ${escapeHtml(item.ticket_type || 'COMMON')} · ${escapeHtml(item.project_id || '团队共享')} · ${escapeHtml(item.agent_id || 'shared')} · 访问 ${item.access_count || 0}</span><div><button class="button ghost" data-memory-edit="${index}" type="button" ${item.status === 'superseded' ? 'disabled' : ''}>修改规则</button> <button class="button ghost" data-memory-revoke="${index}" type="button" ${item.status === 'active' ? '' : 'disabled'}>停用</button></div></div>`).join('') || '<div class="compact-item">尚未形成长期记忆</div>';
  $('#memory-list').querySelectorAll('[data-memory-edit], [data-memory-revoke]').forEach(button => {
    button.addEventListener('click', async () => {
      const editing = button.hasAttribute('data-memory-edit');
      const item = visibleMemories[Number(editing ? button.dataset.memoryEdit : button.dataset.memoryRevoke)];
      const content = editing ? window.prompt('修改后的规则（会保留旧版本）', item.content) : null;
      if (editing && !content?.trim()) return;
      const reason = window.prompt(editing ? '修改原因' : '停用原因（历史记录会保留）');
      if (!reason?.trim()) return;
      button.disabled = true;
      try {
        await api(`/api/memory/${encodeURIComponent(item.id)}/${editing ? 'revise' : 'invalidate'}`, {
          method:'POST', body:JSON.stringify(editing ? {content, reason} : {reason})
        });
        $('#memory-result').textContent = '记忆已更新，请重新检索查看当前结果。';
        await loadContext();
      } catch (error) { notify(error.message); }
      finally { button.disabled = false; }
    });
  });
  $('#memory-list').querySelectorAll('.compact-item').forEach((element, index) => {
    const button = document.createElement('button');
    button.className = 'button ghost';
    button.type = 'button';
    button.textContent = '版本对比 / 回滚';
    button.addEventListener('click', () => window.openFactVersions(visibleMemories[index].id));
    element.appendChild(button);
  });
  $('#template-list').innerHTML = templates.map(item => `<div class="compact-item"><strong>${escapeHtml(item.name)}</strong><span>${escapeHtml(item.status)} · v${item.version} · 支持样本 ${item.support_count}</span></div>`).join('') || '<div class="compact-item">尚未形成场景模板</div>';
}

document.querySelectorAll('.pipeline-step').forEach(step => step.addEventListener('click', () => {
  if (!state.project && step.dataset.phase !== 'draft') return;
  const available = phaseOrder.indexOf(state.project?.phase || 'draft');
  const target = phaseOrder.indexOf(step.dataset.phase);
  if (target <= available) setView(phaseView(step.dataset.phase));
}));

api('/api/health').then(health => {
  const badge = $('#model-status');
  badge.textContent = health.llm_enabled ? `模型 · ${health.model}` : '演示模式 · 无需 Key';
  badge.classList.toggle('live', health.llm_enabled);
});
loadRagDatasets();
loadProjects();


async function toggleMindMap(containerId, view, button) {
  const container = document.getElementById(containerId);
  if (!container.hidden) {
    container.hidden = true;
    button.textContent = '脑图视图';
    return;
  }
  container.hidden = false;
  button.textContent = '收起脑图';
  if (container.dataset.loaded === 'true') return;
  container.innerHTML = '<div class="mindmap-loading">正在转换脑图...</div>';
  try {
    const document = await api(`/api/projects/${state.project.id}/mindmap/${view}`);
    const stats = document.stats || {};
    container.innerHTML = `<div class="mindmap-summary"><strong>${escapeHtml(document.root.label)}</strong><span>${stats.modules || 0} 个模块${view === 'cases' ? ` · ${stats.cases || 0} 条用例` : ''}</span></div><div class="mindmap-tree">${mindMapNode(document.root, 0)}</div>`;
    container.dataset.loaded = 'true';
  } catch (error) {
    container.hidden = true;
    button.textContent = '脑图视图';
    notify(error.message);
  }
}

function mindMapNode(node, depth) {
  const allowed = ['project', 'module', 'case', 'group', 'precondition', 'step', 'expected', 'test_data'];
  const type = allowed.includes(node.node_type) ? node.node_type : 'group';
  const meta = node.metadata || {};
  const badges = [meta.priority, meta.case_type].filter(Boolean);
  if (type === 'module' && meta.direct_case_count !== undefined) badges.push(`${meta.direct_case_count} cases`);
  const badgeHtml = badges.map(value => `<small>${escapeHtml(String(value))}</small>`).join('');
  const label = `<span>${escapeHtml(node.label)}</span>${badgeHtml}`;
  const children = node.children || [];
  if (!children.length) return `<div class="mindmap-node type-${type} mindmap-leaf"><div class="mindmap-label">${label}</div></div>`;
  return `<details class="mindmap-node type-${type}" ${depth < 2 ? 'open' : ''}><summary class="mindmap-label">${label}</summary><div class="mindmap-children">${children.map(child => mindMapNode(child, depth + 1)).join('')}</div></details>`;
}
