(() => {
  const workspace = document.querySelector('.workspace');
  const toggle = document.querySelector('#project-rail-toggle');
  const list = document.querySelector('#project-list');
  const storageKey = 'caseforge.projectRail.collapsed';
  let collapsed = false;
  try { collapsed = localStorage.getItem(storageKey) === 'true'; } catch (_) { /* Storage may be unavailable. */ }

  function render() {
    if (collapsed && list.contains(document.activeElement)) toggle.focus();
    workspace.classList.toggle('rail-collapsed', collapsed);
    list.hidden = collapsed;
    toggle.setAttribute('aria-expanded', String(!collapsed));
    const label = collapsed ? '展开任务栏' : '收起任务栏';
    toggle.setAttribute('aria-label', label);
    toggle.title = label;
  }

  toggle.addEventListener('click', () => {
    collapsed = !collapsed;
    render();
    try { localStorage.setItem(storageKey, String(collapsed)); } catch (_) { /* Keep the current session usable. */ }
  });
  render();
})();
