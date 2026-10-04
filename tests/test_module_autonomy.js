// Exercise the actual module renderer and old-run resume controls without a browser.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const elements = new Map(), events = new Map(), calls = [];
const $ = selector => {
  if (!elements.has(selector)) elements.set(selector, {
    value: '', listeners: {}, addEventListener(type, fn) { this.listeners[type] = fn; }
  });
  return elements.get(selector);
};
const project = {id: 'demo', module_tree: {confirmed: false, modules: [
  {id: 'M1', name: 'Module', objective: 'Behavior', children: []}
]}};
const run = {id: 'AR-old', status: 'waiting_confirmation', mode: 'model', steps: [], max_steps: 12};
const context = vm.createContext({$, state: {project}, window: {}, escapeHtml: String,
  document: {querySelectorAll: () => [], addEventListener: (type, fn) => events.set(type, fn)},
  api: async () => [run], notify: message => { throw Error(message); },
  runCaseOperation: mode => calls.push(mode)
});
for (const file of ['module-workspace.js', 'supervisor.js']) {
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../app/static', file), 'utf8'), context);
}
(async () => {
  vm.runInContext('renderModules()', context);
  assert.match($('#modules-view').innerHTML, /id="generate-cases"/);
  $('#generate-cases').listeners.click();
  assert.deepEqual(calls, ['full']);
  assert.equal(project.module_tree.confirmed, false);
  await events.get('project-rendered')();
  assert.equal($('#supervisor-continue').disabled, false);
  assert.match($('#supervisor-question').textContent, /直接继续/);
  run.status = 'completed';
  await events.get('project-rendered')();
  assert.equal($('#supervisor-continue').disabled, true);
  console.log('Unconfirmed module generation and legacy pause resume controls passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });
