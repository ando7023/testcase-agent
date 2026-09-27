const chunkState = { knowledge: [] };

function selectedChunkIds() {
  return [...document.querySelectorAll('[data-chunk-select]:checked')].map(item => item.value);
}

function updateMergeButton() {
  document.querySelector('#merge-chunks').disabled = selectedChunkIds().length < 2;
}

window.renderChunkKnowledge = function (knowledge) {
  chunkState.knowledge = knowledge;
  const active = knowledge.filter(item => item.status === 'active' && item.chunk_level !== 'parent');
  const parents = knowledge.filter(item => item.status === 'active' && item.chunk_level === 'parent').length;
  const inactive = knowledge.filter(item => item.status === 'inactive').length;
  document.querySelector('#chunk-summary').textContent = active.length + ' 个活动子块 · ' + parents + ' 个父块 · ' + inactive + ' 个历史版本';
  document.querySelector('#knowledge-list').innerHTML = active.map(item => {
    const editable = Boolean(item.source_id && item.parent_id);
    const parent = item.parent_id ? ' · parent ' + escapeHtml(item.parent_id) : '';
    const controls = editable
      ? '<div class="chunk-actions"><label class="chunk-check"><input type="checkbox" data-chunk-select value="' + escapeHtml(item.id) + '">选择</label><button class="icon-button chunk-split" type="button" data-chunk-id="' + escapeHtml(item.id) + '" title="拆分知识块">↥</button></div>'
      : '<div class="chunk-actions"><button class="button ghost chunk-convert" type="button" data-chunk-id="' + escapeHtml(item.id) + '" title="创建父块并迁移为新版子块">转为父子块</button></div>';
    return '<div class="compact-item chunk-item"><div><strong>' + escapeHtml(item.title) + '</strong><span>' + escapeHtml(item.doc_type) + ' · ' + escapeHtml(item.metadata?.ticket_type || '未设置') + ' · ' + escapeHtml(item.id) + ' · v' + (item.chunk_version || 1) + parent + '</span></div>' + controls + '</div>';
  }).join('') || '<div class="compact-item">知识库暂为空</div>';
  updateMergeButton();
};

async function openSplitEditor(documentId) {
  const chunk = chunkState.knowledge.find(item => item.id === documentId);
  if (!chunk) return;
  let parts;
  try {
    const suggestion = await api('/api/knowledge/' + encodeURIComponent(documentId) + '/split/suggest', {method:'POST'});
    parts = suggestion.parts;
  } catch (error) {
    notify(error.message);
    return;
  }
  const editor = document.querySelector('#chunk-editor');
  editor.dataset.documentId = documentId;
  editor.hidden = false;
  const fields = parts.map((part, index) => '<label>子块 ' + (index + 1) + '<textarea rows="5" data-split-part>' + escapeHtml(part.trim()) + '</textarea></label>').join('');
  editor.innerHTML = '<div class="chunk-editor-heading"><strong>拆分 ' + escapeHtml(chunk.title) + '</strong><button class="icon-button" id="close-chunk-editor" type="button" title="关闭">×</button></div><div class="split-strategy">按段落与完整句子建议边界，可继续人工调整</div><div id="split-parts">' + fields + '</div><div class="chunk-editor-actions"><button class="button ghost" id="add-split-part" type="button">增加子块</button><button class="button secondary" id="save-split" type="button">保存新版本</button></div>';}

document.querySelector('#knowledge-list').addEventListener('change', event => {
  if (event.target.matches('[data-chunk-select]')) updateMergeButton();
});

document.querySelector('#knowledge-list').addEventListener('click', async event => {
  const convertButton = event.target.closest('.chunk-convert');
  if (convertButton) {
    try {
      await api('/api/knowledge/' + encodeURIComponent(convertButton.dataset.chunkId) + '/convert', {method:'POST'});
      await loadContext();
    } catch (error) { notify(error.message); }
    return;
  }
  const button = event.target.closest('.chunk-split');
  if (button) await openSplitEditor(button.dataset.chunkId);
});

document.querySelector('#chunk-editor').addEventListener('click', async event => {
  if (event.target.closest('#close-chunk-editor')) {
    document.querySelector('#chunk-editor').hidden = true;
    return;
  }
  if (event.target.closest('#add-split-part')) {
    const count = document.querySelectorAll('[data-split-part]').length + 1;
    document.querySelector('#split-parts').insertAdjacentHTML('beforeend', '<label>子块 ' + count + '<textarea rows="4" data-split-part></textarea></label>');
    return;
  }
  if (!event.target.closest('#save-split')) return;
  const parts = [...document.querySelectorAll('[data-split-part]')].map(item => item.value.trim()).filter(Boolean);
  try {
    const documentId = document.querySelector('#chunk-editor').dataset.documentId;
    await api('/api/knowledge/' + encodeURIComponent(documentId) + '/split', {method:'POST', body:JSON.stringify({parts})});
    document.querySelector('#chunk-editor').hidden = true;
    await loadContext();
  } catch (error) { notify(error.message); }
});

document.querySelector('#merge-chunks').addEventListener('click', async () => {
  try {
    await api('/api/knowledge/merge', {method:'POST', body:JSON.stringify({document_ids:selectedChunkIds()})});
    await loadContext();
  } catch (error) { notify(error.message); }
});