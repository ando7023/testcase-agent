(() => {
  const labels = {running: '执行中', waiting_input: '等待补充信息', waiting_confirmation: '等待模块确认', completed: '已完成', budget_exhausted: '步数用尽', failed: '执行失败', needs_attention: '降级产物待核验'};
  const names = {requirement_understanding: '需求理解', module_planning: '模块规划', case_generation: '用例生成', quality_critic: '独立评审', case_revision: '单轮修复', inspect_materials: '检查输入物料', search_knowledge: '检索知识', request_input: '等待人工输入', finish: '完成检查'};
  let runs = [], selected = null, projectId = '', revision = 0;
  function render() {
    const run = selected;
    $('#supervisor-status').textContent = run ? `${labels[run.status] || run.status} · ${run.mode === 'model' ? '模型决策' : '离线规则演示'} · ${run.steps.length}/${run.max_steps} 步` : '尚未运行';
    const paused = run && ['waiting_input', 'waiting_confirmation', 'budget_exhausted'].includes(run.status);
    $('#supervisor-pause').hidden = !paused;
    $('#supervisor-question').textContent = run?.status === 'waiting_confirmation' ? '此历史运行曾等待模块确认；现在可直接继续，预算不足时请追加步数。' : run?.question || '';
    $('#supervisor-continue').disabled = !paused;
    $('#supervisor-log').innerHTML = (run?.error ? `<p class="supervisor-error">${escapeHtml(run.error)}</p>` : '') +
      (state.project?.human_acceptance ? `<p>当前用例人工验收：${state.project.human_acceptance.status === 'accepted' ? '已全部验收' : '尚未全部验收'} · ${state.project.human_acceptance.accepted}/${state.project.human_acceptance.total} 条已采纳。验收针对用例设计，不代表接口执行通过。</p>` : '') +
      (run?.degraded ? '<p>本次运行历史包含本地演示或回退结果；历史状态保留，当前用例的人工验收结果单独显示。</p>' : '') +
      (Object.keys(run?.issue_ledger || {}).length ? `<details><summary>问题跟踪 · 已尝试修复 ${run.repair_rounds || 0}/2 轮</summary>${Object.entries(run.issue_ledger).map(([id, item]) => `<p>${escapeHtml(id)} · ${item.status === 'not_observed' ? '本轮未出现（尚非关闭证明）' : '本轮仍有发现'} · ${escapeHtml(item.finding?.case_id || '')} · ${escapeHtml(item.finding?.message || '')}${item.reopened ? ' · 再次出现' : ''}${item.classification_changed ? ' · 分类有变化，需核对依据' : ''}</p>`).join('')}</details>` : '') +
      (run?.steps || []).map(step => {
        const d = step.decision, o = step.observation || {};
        return `<details class="supervisor-step" ${step.status !== 'success' ? 'open' : ''}>
          <summary>${step.index}. ${escapeHtml(names[d?.capability || d?.action] || '决策校验')} · ${escapeHtml(step.status)}${d?.skills?.length ? ` · ${escapeHtml(d.skills.join(', '))}` : ''}</summary>
          <p>${escapeHtml(d?.reason || '模型决策未通过格式校验')}</p>
          ${d?.instruction ? `<p>任务：${escapeHtml(d.instruction)}</p>` : ''}
          ${o.review ? `<p>评审 ${escapeHtml(o.review.score)} 分 · ${o.review.findings.length} 项发现</p>` : ''}
          <pre>${escapeHtml(JSON.stringify(o, null, 2))}</pre>
        </details>`;
      }).join('');
  }
  async function refresh() {
    if (!state.project) return;
    const id = state.project.id, version = ++revision;
    $('#supervisor-panel').hidden = false;
    if (projectId !== id) {
      projectId = id; runs = []; selected = null;
      $('#supervisor-answer').value = '';
      $('#supervisor-goal').value = '生成满足需求的测试用例，并根据评审反馈修复';
      render();
    }
    try {
      const result = await api(`/api/projects/${id}/agent-runs`);
      if (version !== revision || state.project?.id !== id) return;
      runs = result;
      selected = runs.find(r => r.id === selected?.id) || runs[0] || null;
      $('#supervisor-history').innerHTML = runs.length ? runs.map(r => `<option value="${escapeHtml(r.id)}">${escapeHtml(r.created_at)} · ${escapeHtml(labels[r.status] || r.status)}</option>`).join('') : '<option value="">暂无记录</option>';
      $('#supervisor-history').value = selected?.id || '';
      render();
    } catch (error) { if (state.project?.id === id) notify(error.message); }
  }
  window.startAgenticRun = async (id) => {
    const maxSteps = Number($('#supervisor-budget').value);
    const goal = $('#supervisor-goal').value.trim();
    if (!goal || !Number.isInteger(maxSteps) || maxSteps < 1 || maxSteps > 20) throw new Error('请填写目标和 1–20 之间的决策步数');
    setBusy(true, 'Supervisor 正在自主规划、生成与评审；需要业务补充时会暂停…');
    const run = await api(`/api/projects/${id}/agent-runs`, {method: 'POST', body: JSON.stringify({goal, max_steps: maxSteps})});
    if (state.project?.id === id) {
      selected = run; projectId = id;
      state.project = await api(`/api/projects/${id}`);
      renderProject();
    }
    return run;
  };
  $('#supervisor-start').addEventListener('click', async () => {
    if (!state.project) return;
    try { await window.startAgenticRun(state.project.id); await loadProjects(); }
    catch (error) { notify(error.message); }
    finally { setBusy(false); }
  });
  $('#supervisor-continue').addEventListener('click', async () => {
    if (!selected || !state.project) return;
    const id = state.project.id, runId = selected.id;
    setBusy(true, 'Supervisor 正在根据最新状态继续决策…');
    try {
      const additionalSteps = Number($('#supervisor-additional-steps').value);
      if (!Number.isInteger(additionalSteps) || additionalSteps < 0 || additionalSteps > 20) throw new Error('追加步数必须为 0–20 的整数');
      const run = await api(`/api/projects/${id}/agent-runs/${runId}/continue`, {method: 'POST', body: JSON.stringify({answer: $('#supervisor-answer').value, additional_steps: additionalSteps})});
      if (state.project?.id !== id) return;
      selected = run;
      $('#supervisor-answer').value = '';
      $('#supervisor-additional-steps').value = '0';
      state.project = await api(`/api/projects/${id}`);
      renderProject(); await loadProjects();
    } catch (error) { notify(error.message); }
    finally { setBusy(false); }
  });
  $('#supervisor-history').addEventListener('change', event => {
    selected = runs.find(r => r.id === event.target.value) || null;
    $('#supervisor-answer').value = '';
    render();
  });
  document.addEventListener('project-rendered', refresh);
})();
