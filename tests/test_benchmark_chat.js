const assert = require('node:assert/strict');
const {SSEDecoder, preview} = require('../app/static/benchmark-chat.js');

// Split every UTF-8 byte and every SSE boundary; Chinese text must survive.
const events = [];
const frames = new SSEDecoder(event => events.push(event));
const decoder = new TextDecoder();
const bytes = new TextEncoder().encode(': keepalive\r\n\r\ndata: {"event":"delta","text":"模型正文🟢"}\r\n\r\ndata: {"event":"finished"}\n\n');
for (const byte of bytes) frames.feed(decoder.decode(new Uint8Array([byte]), {stream: true}));
frames.feed(decoder.decode(), true);
assert.deepEqual(events, [{event: 'delta', text: '模型正文🟢'}, {event: 'finished'}]);
assert.equal(preview('{"summary":"正在生成'), '摘要：正在生成');
assert.equal(preview('{"summary":"换行\\n中文"}'), '摘要：换行\n中文');
assert.equal(preview('{"summary":"中文\\u4e'), '摘要：中文');
assert.equal(preview('{"question":"<script>alert(1)</script>"}'), '待确认：<script>alert(1)</script>');
assert.equal(preview('{"score":92}'), '{"score":92}');
assert.throws(() => new SSEDecoder(() => {}).feed('data: broken\n\n'), SyntaxError);

// Render the actual report function: handled validation errors must remain visible.
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(require.resolve('../app/static/app.js'), 'utf8');
const labels = source.slice(source.indexOf('const benchmarkLabels ='), source.indexOf('function benchmarkCount('));
const renderer = source.slice(source.indexOf('function benchmarkMetricValue('), source.indexOf('function renderBenchmarkReports('));
const resultNode = {};
const sandbox = {
  $: () => resultNode,
  escapeHtml: value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;'),
};
vm.createContext(sandbox);
vm.runInContext(labels + renderer, sandbox);
sandbox.renderBenchmarkReport({schema_version: 2, dataset_id: 'EBT-RAG-V1', mode: 'live',
  samples: [{id: '103', technical_failure: true, quality_passed: null, status: 'technical_failed',
    failure_stage: 'worker_execution', failure_agent: 'requirement_understanding',
    failure_phase: 'scope_validation', failure_error_code: 'invalid_scope',
    error: '<script>private</script>'}]});
assert.match(resultNode.innerHTML, /需求理解/);
assert.match(resultNode.innerHTML, /证据范围校验/);
assert.match(resultNode.innerHTML, /invalid_scope/);
assert.match(resultNode.innerHTML, /&lt;script&gt;/);
assert.doesNotMatch(resultNode.innerHTML, /<script>/);
sandbox.renderBenchmarkReport({schema_version: 2, mode: 'live', samples: [{id: '104',
  case_design_level: 'behavior', clarification_scope_review: [{clarification_id: 'G1',
    original_kind: 'behavior_blocker', kind: 'out_of_scope', original_reason: '<script>old</script>',
    reason: 'Outside the documented fixture', source_quote: '<img src=x onerror=alert(1)>', evidence_ids: ['DOC-1']}]}]});
assert.match(resultNode.innerHTML, /澄清范围复核/);
assert.match(resultNode.innerHTML, /阻塞预期判断 → 原文范围之外/);
assert.match(resultNode.innerHTML, /Outside the documented fixture/);
assert.doesNotMatch(resultNode.innerHTML, /<script>|<img /);
console.log('Benchmark SSE framing, UTF-8 fragmentation and preview checks passed.');
