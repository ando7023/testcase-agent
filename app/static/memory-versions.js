(() => {
  const dialog = $('#memory-versions-dialog');
  let catalog = {}, projectId = '', fact = null, preview = null, sequence = 0, busy = false;
  const kind = () => $('#memory-version-kind').value;
  const clear = () => { preview = null; $('#memory-version-restore').disabled = true; $('#memory-version-diff').innerHTML = ''; };
  function message(text) { $('#memory-version-result').textContent = text; }
  function options() {
    clear();
    $('#memory-versions-scope').textContent = kind() === 'fact'
      ? `长期事实：${fact?.current_id || fact?.anchor || ''}`
      : state.project ? `项目：${state.project.title}` : '请先选择项目。';
    const entries = kind() === 'fact' ? fact?.versions || [] : catalog[kind() === 'snapshot' ? 'snapshots' : kind()] || [];
    const html = entries.map(v => `<option value="${escapeHtml(v.id)}">${escapeHtml(v.created_at)} · ${escapeHtml(v.label || v.status || '')} · ${escapeHtml(v.id)}</option>`).join('');
    $('#memory-version-left').innerHTML = html;
    $('#memory-version-right').innerHTML = (kind() === 'fact' ? '' : '<option value="current">当前状态</option>') + html;
    if (kind() === 'fact') {
      $('#memory-version-right').value = fact.current_id || fact.anchor;
      const older = entries.find(v => v.id !== $('#memory-version-right').value);
      if (older) $('#memory-version-left').value = older.id;
    }
    $('#memory-version-save-row').hidden = kind() === 'fact';
    $('#memory-version-restore-note').textContent = kind() === 'fact'
      ? '事实回滚会新建版本，保留原历史与作用域。仅允许恢复同一版本链的祖先；过期或无法核对原有效期的版本需显式修订。'
      : '整体恢复包含项目产物、会话与决策记忆、项目长期事实、示例和 badcase；共享团队规则与运行历史保留。恢复前自动备份，恢复后重新确认模块和评审；后续人工拒绝不会被旧快照覆盖。';
    $('#memory-version-compare').disabled = !entries.length;
    message(entries.length ? '选择版本后点击对比差异。' : '暂无版本。可以先保存当前项目快照。');
  }
  async function refresh() {
    const revision = ++sequence;
    projectId = state.project?.id || '';
    clear();
    $('#memory-versions-scope').textContent = state.project ? `项目：${state.project.title}` : '请先选择项目。';
    $('#memory-version-save').disabled = !projectId;
    if (!projectId) { catalog = {}; options(); return; }
    try {
      const result = await api(`/api/projects/${encodeURIComponent(projectId)}/memory-versions`);
      if (revision !== sequence) return;
      catalog = result; options();
    } catch (error) { message(error.message); }
  }
  $('#memory-versions-open').addEventListener('click', () => {
    fact = null;
    $('#memory-version-kind').querySelector('[value="fact"]').disabled = true;
    $('#memory-version-kind').value = 'snapshot'; dialog.showModal(); refresh();
  });
  $('#memory-versions-close').addEventListener('click', () => dialog.close());
  $('#memory-version-kind').addEventListener('change', () => {
    ++sequence;
    if (kind() !== 'fact' && projectId !== state.project?.id) refresh(); else options();
  });
  for (const selector of ['#memory-version-left', '#memory-version-right']) $(selector).addEventListener('change', () => { ++sequence; clear(); });
  $('#memory-show-history').addEventListener('change', () => loadContext().catch(e => notify(e.message)));
  window.openFactVersions = async id => {
    const revision = ++sequence;
    clear(); dialog.showModal();
    try {
      const result = await api(`/api/memory/${encodeURIComponent(id)}/history`);
      if (revision !== sequence) return;
      fact = {...result, anchor: id};
      $('#memory-version-kind').querySelector('[value="fact"]').disabled = false;
      $('#memory-version-kind').value = 'fact';
      $('#memory-versions-scope').textContent = `长期事实：${id}`;
      options();
    } catch (error) { message(error.message); }
  };
  $('#memory-version-compare').addEventListener('click', async () => {
    const revision = ++sequence, selectedKind = kind(), left = $('#memory-version-left').value, right = $('#memory-version-right').value;
    clear();
    try {
      const data = selectedKind === 'fact'
        ? await api(`/api/memory/${encodeURIComponent(right)}/diff?target_id=${encodeURIComponent(left)}`)
        : await api(`/api/projects/${encodeURIComponent(projectId)}/memory-versions/diff`, {method: 'POST', body: JSON.stringify({kind: selectedKind, left_id: left, right_id: right})});
      if (revision !== sequence) return;
      preview = {...data, projectId, kind: selectedKind, left, right};
      message(`新增 ${data.summary.added} · 删除 ${data.summary.removed} · 修改 ${data.summary.modified}`);
      $('#memory-version-diff').innerHTML = data.changes.length ? `<table class="data-table"><thead><tr><th>位置</th><th>旧版本</th><th>新版本</th></tr></thead><tbody>${data.changes.map(c => `<tr><td>${escapeHtml(c.path)}<br>${escapeHtml(c.operation)}</td><td><pre>${escapeHtml(JSON.stringify(c.before, null, 2))}</pre></td><td><pre>${escapeHtml(JSON.stringify(c.after, null, 2))}</pre></td></tr>`).join('')}</tbody></table>` : '<p>没有内容差异。</p>';
      $('#memory-version-restore').disabled = busy || !(selectedKind === 'snapshot' && right === 'current' || selectedKind === 'fact' && right === fact.current_id && left !== right);
    } catch (error) { message(error.message); }
  });
  $('#memory-version-save').addEventListener('click', async () => {
    if (!projectId || busy) return;
    busy = true; $('#memory-version-save').disabled = true;
    try {
      await api(`/api/projects/${encodeURIComponent(projectId)}/memory-versions`, {method: 'POST', body: JSON.stringify({label: $('#memory-version-label').value.trim() || '手动快照'})});
      await refresh(); message('快照已保存。');
    } catch (error) { message(error.message); }
    finally { busy = false; $('#memory-version-save').disabled = !projectId; }
  });
  $('#memory-version-restore').addEventListener('click', async () => {
    if (!preview || busy) return;
    const selected = preview;
    busy = true; $('#memory-version-restore').disabled = true;
    try {
      if (selected.kind === 'fact') {
        const result = await api(`/api/memory/${encodeURIComponent(selected.right)}/rollback`, {method: 'POST', body: JSON.stringify({target_id: selected.left, expected_current_id: selected.right, reason: `人工回滚到 ${selected.left}`})});
        await loadContext(); await window.openFactVersions(result.id); message('已回滚为新版本，历史已保留。');
      } else {
        if (selected.projectId !== state.project?.id) throw new Error('项目已切换，请重新对比。');
        const result = await api(`/api/projects/${encodeURIComponent(selected.projectId)}/memory-versions/restore`, {method: 'POST', body: JSON.stringify({snapshot_id: selected.left, expected_fingerprint: selected.current_fingerprint})});
        if (state.project?.id === selected.projectId) { state.project = result.project; renderProject(); }
        await loadProjects(); await loadContext(); await refresh();
        message(`已恢复。恢复前备份：${result.backup_id}。请重新确认模块并评审。`);
      }
    } catch (error) { message(error.message); clear(); }
    finally { busy = false; }
  });
  document.addEventListener('project-rendered', () => { if (dialog.open && kind() !== 'fact') refresh(); });
})();
