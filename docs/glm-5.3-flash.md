# GLM-5.3-Flash 接入

项目使用智谱标准 Chat Completions API，复用现有 HTTP 客户端，无需安装新 SDK。Supervisor、需求理解、模块规划、用例生成、评审以及需要 LLM 的 Memory 操作共用当前模型配置。

## 本地配置

项目根目录 `.env` 已切换至下列设置。请在本地填写 `ZHIPU_API_KEY`，不要把真实密钥提交到仓库或粘贴进聊天。

```dotenv
ZHIPU_API_KEY=填写智谱开放平台的API密钥
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
LLM_MODEL=glm-5.3-flash
LLM_REASONING_EFFORT=max
LLM_THINKING_ENABLED=true
LLM_TEMPERATURE=1
LLM_TOP_P=0.95
LLM_MAX_TOKENS=32768
LLM_TIMEOUT_SECONDS=180
LLM_DISABLED=false
RAG_EMBEDDING_PROVIDER=hashing
MEMORY_EMBEDDING_PROVIDER=hashing
```

模型标识、采样与推理设置来自[模型官方文档](https://docs.bigmodel.cn/cn/guide/models/vlm/glm-5.3-flash)；32768 的输出预算和 180 秒超时是本项目的初始配置，可按实际用例集规模调整。该模型只支持开启思考，客户端会强制发送 `thinking.type=enabled`。

密钥读取：显式 `LLM_API_KEY` 优先，否则在智谱地址下读取 `ZHIPU_API_KEY`，兼容 `ZAI_API_KEY`。显式设置为空的 `LLM_API_KEY` 会禁用请求，可用于隔离测试。智谱地址不会回退使用 `DEEPSEEK_API_KEY`；旧密钥保留在本地，便于以后切回。

填入密钥后重启服务。进程环境变量优先于 `.env`：如果终端中曾设置旧的 `LLM_BASE_URL`、`LLM_MODEL` 或空的 `LLM_API_KEY`，需要先清除这些旧覆盖值。页面模型标识和 `/api/health` 可确认加载结果，`llm_enabled=true` 只表示配置存在，不代表已通过真实请求验证。

密钥从[智谱开放平台](https://bigmodel.cn/usercenter/proj-mgmt/apikeys)创建。接入地址依据[官方兼容接口说明](https://docs.bigmodel.cn/cn/guide/develop/openai/introduction)。

## 输出与当前范围

- 同步和 SSE 流式请求使用相同参数，开启 `response_format={"type":"json_object"}`，把 Schema 写入系统消息；Supervisor 继续使用 Pydantic 校验决策。
- 只消费最终 `content`，不把 `reasoning_content` 当成业务 JSON 或向前端流式输出。
- 发送官方建议的 `thinking.clear_thinking=false`。当前每次调用是独立的 system/user 请求，尚未实现完整 assistant reasoning 历史的跨调用回传，因此不宣称已实现 Preserved Thinking。
- 当前 Supervisor 使用 JSON 动作协议，未使用原生 Function Calling；因此没有额外开启只用于工具参数增量的 `tool_stream`。
- 本轮接通文本生成和调度。图片/视频输入仍未接入该模型；上传文档继续经过原有解析流程。RAG/Memory 保持本地 hashing embedding，不将聊天模型误用作向量模型。

参考：[结构化输出](https://docs.bigmodel.cn/cn/guide/capabilities/struct-output)、[思考模式](https://docs.bigmodel.cn/cn/guide/capabilities/thinking-mode)。

## 长请求超时诊断

`generate_json` 默认仍使用非流式请求；设置 `LLM_STREAM_JSON=true` 可改为 SSE 接收后再统一解析 JSON，业务接口返回值不变。测试脚本支持 `--stream` 临时开启。流式响应仍受连接/读取超时约束，不能保证消除网络或服务端延迟；持续收到数据时，全程耗时可能超过读取超时配置。

先只诊断现有用例的一次语义评审，不运行 Supervisor、不改用例和运行状态：

```powershell
python tests/manual_agentic_smoke.py --root data/live_smoke/glm-20260920-130433 --review-only --stream --timeout-seconds 180
```

脚本输出 `llm_progress`：`connected` 表示已收到响应头；`event_count`、`reasoning_chars`、`content_chars` 表示收到的事件数和文本长度，不输出推理原文。收到数据时约每 15 秒报告一次，静默等待时没有心跳；`reasoning_chars` 增长说明流式推理增量已到达客户端，`content_chars` 增长说明最终正文已到达。没有数据不能单独区分 VPN、代理、服务端排队或响应缓冲。

诊断写入独立 `report-review-*.json`。`reviewed` 仅表示评审调用完成，不代表用例质量通过。响应被截断、流式错误或请求失败不能视为有效评审，会产生高优先级 `review_incomplete`，阻止完成门禁。若响应包含 usage，Trace 记录实际值，否则保留估算标识。

测试脚本在解析前将最终响应正文保存到同目录的 `response-*.txt`，调用记录中的 `response_file` 指向该文件；不保存推理原文或请求凭证，并遮盖当前 API key。该文件可能包含测试需求信息，与其他 smoke 数据一样仅保存在本地忽略目录。解析器兼容 JSON 外层代码围栏，以及完整 JSON 对象之后独立一行的两个或三个反引号（后者包含实测发现的残缺结尾围栏）；先解析完整对象，再严格校验余串，不接受第二个对象、额外说明文字或被截断的 JSON。

单次评审确认成功后，可用 `--restart-run --stream --max-steps 8` 从失败运行的现有产物启动修复流程；不要重复用更大的超时值掩盖未知原因。

## 切回 DeepSeek

修改 `.env` 后重启：

```dotenv
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-v4-flash
DEEPSEEK_API_KEY=填写原有密钥
LLM_REASONING_EFFORT=high
LLM_THINKING_ENABLED=true
LLM_TEMPERATURE=0.2
LLM_TOP_P=
LLM_MAX_TOKENS=0
LLM_TIMEOUT_SECONDS=120
```

不设置 `LLM_API_KEY` 时，DeepSeek 地址继续读取 `DEEPSEEK_API_KEY`。`LLM_MAX_TOKENS=0` 表示不发送该参数。
