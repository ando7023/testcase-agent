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
console.log('Benchmark SSE framing, UTF-8 fragmentation and preview checks passed.');
