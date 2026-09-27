# GLM-5.3 配置与对照测试

官方说明：[GLM-5.3](https://docs.bigmodel.cn/cn/guide/models/text/glm-5.3)。模型 ID 为 `glm-5.3`，可复用 `https://open.bigmodel.cn/api/paas/v4` 的 Chat Completion 接口。必须启用思考，支持 `low`、`high`、`max` 三档。

2026-09-22 已将本地 `.env` 的 `LLM_MODEL` 从 `glm-5.3-flash` 改为 `glm-5.3`，其余配置和凭证保留。`.env.example` 仍保留 Flash 作为原示例，正式模型选择以本地配置或测试参数为准。服务需重启才能载入新配置；已有进程环境变量仍优先于 `.env`。

```dotenv
LLM_MODEL=glm-5.3
LLM_REASONING_EFFORT=max
LLM_THINKING_ENABLED=true
```

脚本支持 `--model` 临时覆盖，因此可在相同项目上比较两个模型而无需反复修改配置：

```powershell
python tests/manual_agentic_smoke.py --root data/live_smoke/glm-20260920-130433 --review-only --stream --model glm-5.3 --timeout-seconds 180
python tests/manual_agentic_smoke.py --root data/live_smoke/glm-20260920-130433 --review-only --stream --model glm-5.3-flash --timeout-seconds 180
```

每条命令会实际调用一次对应模型。`--review-only` 不改项目和原运行。报告记录本次实际模型、采样设置及输入 SHA-256，不沿用 manifest 中的历史模型标签。模型对比时应保持同一版用例、相同提示词和推理强度；仅凭一次评分高低不能判断优劣，还需核对建议是否有需求依据、是否反复、是否可执行以及耗时和用量。

2026-09-22 的模型切换验证未同时修改 FIX-007，以隔离模型变量。2026-09-23 已实现以下评审逻辑并通过离线回归，真实模型闭环仍待验证：

1. 把原始需求、原子需求、业务规则和用户澄清传给 Critic，避免只传摘要和用例。
2. 将确定缺陷、需要补充契约的问题、非阻塞建议区分处理；高风险和评审未完成仍阻止交付，不能用降低评分阈值代替修复。
3. 保存问题依据和出现历史，标记再次出现及阻塞分类变化；未出现不等于已证明关闭。缺少业务信息、修复无变化或两轮修复后仍有阻塞问题时暂停，请求澄清或人工裁定。

`--review-only` 也会读取该运行的目标、回答、历史评审和问题台账，但不会修改原项目或运行。升级后提示词及输入已变化，后续模型对比应统一使用新版本；不能将旧提示词的 82/85 分直接当作修复前后效果对照。新策略说明见 [Harness 评审分类与收敛控制](agentic-harness.md)。

执行结果与验证边界追加到 [修复日志](fix-log.md)。流式请求、超时和解析兼容说明见 [GLM 接入说明](glm-5.3-flash.md)。

2026-09-22 单次真实评审已完成：360.94 秒，82 分，2 项中风险、4 项低风险发现，27,495 tokens，无传输或解析错误。仅证明接入及本次评审成功；尚未验证正式模型完整修复流程，也不能以单次评分判断其优于 Flash。

## 输出长度限制与步骤预算

2026-09-23 续跑第 15 步时，`glm-5.3` 的评审调用返回 `finish_reason=length`，该次配置为 `reasoning_effort=max`、`max_tokens=32768`。这是单次模型输出达到长度限制，不是决策步数不足，也不是读取超时。该运行当时已用 15/22 步，仍余 7 步。失败请求未完整计入 usage，报告总量不能用作该轮完整计费统计。

脚本新增 `--reasoning-effort low|high|max`，只覆盖本次调用，不修改 `.env`。可以先用 `low` 重试原运行，观察是否能返回完整评审及建议质量；降低强度能否解决此次问题仍待实测，不能保证成功。下面命令使用剩余预算，实际调用模型并可能保存修复产物：

```powershell
python tests/manual_agentic_smoke.py --root data/live_smoke/glm-20260920-130433 --continue-run --stream --model glm-5.3 --reasoning-effort low --timeout-seconds 180 --answer "本次降低思考强度重试技术失败的评审，业务约束保持不变；仍按确定缺陷、待澄清和可选建议分类处理。"
```

技术异常现在显示具体安全分类、剩余步数和重试说明；不再将输出截断提示为缺少业务契约。`review_incomplete` 的分数不能证明业务通过。运行明细和历史降级标记仍保留。

后续 `low` 实测在 49.95 秒结束，但第 16 步评审返回完整 JSON 后又附加了 300 字符的围栏/说明，报 `Extra data`；该轮没有长度截断，但不能据此推断稳定成功率或速度提升。当前剩余 6 步。已补充明确 schema 提示和一次 `invalid_json` 自动格式重试；保持严格解析，不丢弃尾部结论。可沿用上述续跑命令，无需追加预算。新增重试的真实成功率尚待验证，格式通过后也可能有业务待澄清项。

2026-09-25 核查更新：第 17 步评审、第 18 步 finish 已成功，45.12 秒，3 次模型调用全部成功；92 分、4 条非阻塞建议，未触发格式重试或新增用例修改。剩余 4 步，无需追加预算。最终 `needs_attention` 是历史第 4、6 步 fallback 保留下来的提醒，当前运行已终止，不能再用 `--continue-run` 续跑。先人工核对当前用例与遗留建议，不需要为了消除状态提醒而重启或清除历史。详情见 [修复日志](fix-log.md)。
