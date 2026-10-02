# 公开评测入口

## 使用方式

启动 `python run.py`，进入工作台的「公开评测」。外部数据需要先点击对应的「导入」；Critic 内置结构缺陷样本无需下载。数据、报告和运行证据均保存在本地数据目录，不随仓库提交。

1. 选择评测套件和离线 / 真实模型模式。
2. EBT、StorySeek 可选 Workflow 或 Agentic。Workflow 按固定顺序调用 Worker；Agentic 调用项目的 `AgenticSupervisor.start/resume`，使用相同的决策、执行、评审及预算门禁。
3. EBT 默认停在模块确认。批量运行完整生成流程需显式选择「模拟确认模块」；报告记录 `benchmark_simulator`，不会生成代表真人操作的确认事实。
4. Agentic 可设置 1–20 步，默认 12 步。模拟确认最多续跑一次，不回答业务澄清、不增加预算。暂停或预算耗尽会作为未完成样本保留。

真实模式要求已配置且启用模型，否则返回配置错误，不静默改成离线。可能产生模型调用费用；先使用 1 个样本验证配置。离线模式关闭样本 Worker 的模型调用，仅验证规则与流程，不代表模型自主能力。

## 各套件测什么

| 套件 | 支持入口 | 完成条件 | 指标边界 |
| --- | --- | --- | --- |
| EBT 用例生成 | Workflow / Agentic | Workflow 生成并评审；Agentic 执行 finish（含需关注的降级收尾） | 词元重合是代理指标；质量门禁是内部 Critic，不是独立业务验收 |
| StorySeek 需求与模块 | Workflow / Agentic | 需求与非空模块规划完成；Agentic 停在模块确认 | 不评测用例生成/修复/收尾；输入含用户故事字段，召回衡量信息保留，不是隐藏答案预测 |
| SRS 文档解析 | Workflow 组件 | 得到非空原文与规范化文本 | 只报告解析指标，质量门禁为未评定，不宣称完成 Agentic |
| Critic 缺陷检测 | Workflow 组件 | 完成该次评审调用 | 固定 4 种结构变异 + 1 个未人工语义标注的基线；真实模式调用模型，但缺陷召回不是语义评审准确率 |

StorySeek 的 split 采用项目划分。其他套件不使用 split；Critic 不使用 limit。Critic 基线的发现数只作为观察结果，不再称为误报数，因为模板本身没有人工确认的「语义无缺陷」标签。

## 报告口径

新报告 `schema_version=2`，取消跨套件混合综合分（`score=null`）。每个样本分别记录：

- `flow_completed`：达到该套件的流程终点；可能完成但质量不通过，或降级收尾。
- `quality_passed`：内部质量门禁通过 / 未通过 / 未评定。评审缺失、技术失败或降级时为 `null`；不会仅凭分数高就通过。离线的通过仅代表结构规则通过。
- `technical_failure`：调用、执行异常或语义评审未完成。不会把 `review_incomplete` 当作业务澄清或有效评审。
- `degraded`：真实模式发现 demo/fallback、Supervisor 历史降级，或样本内发生模型 JSON 调用异常。采取保守口径：即使后续重试成功也保留异常记录并要求复核；不等同于最终产物必然来自回退。

流程完成率、技术失败率、降级率的分母是全部样本。质量通过率分母是全部**适用门禁**的样本（未完成/技术失败/降级也不算通过）；Critic 基线和 SRS 不适用质量门禁。所有适用样本都未评定时显示「未评定」，同时报告适用数与已评定数。四项可以重叠，不能相加当作样本总数。

EBT/StorySeek 词元指标以全部样本计，执行异常的缺失指标按零计；SRS 文本提取召回以全部样本计，相关文本召回仅针对成功且有相关文本标注的样本，压缩率/告警率针对成功解析样本。它们是诊断指标，不能替代独立标注和真实测试执行。

每个样本使用独立的 `benchmark_runs/BR-…/sample-0001/`，包含项目、Memory、Trace 及 Agentic 运行记录；样本之间不共享学习结果，也不写入已有业务项目。初始知识仍沿用当前应用的内置知识包，尚未建立针对各公开数据集的独立知识配置。

报告另存 `run_id`、`run_status`、步骤与动作、模拟确认事件、耗时、逻辑 JSON 调用数及错误次数。`llm_calls` 是客户端 `generate_json` 调用次数，不是 HTTP 请求次数或 token 数。完整模型成本统计与运行耗时预算未在本次实现。

历史报告原样保留，界面标识「旧版」；旧 `passed` / 综合分不能重新解释为 Agentic 质量结论。

## API 示例

服务启动、EBT 导入后，在 PowerShell 执行以下离线 Agentic 样本评测：

```powershell
$payload = @{
    suite = 'ebt_generation'
    limit = 1
    mode = 'offline'
    execution = 'agentic'
    human_policy = 'simulate_confirm'
    max_steps = 12
} | ConvertTo-Json
Invoke-RestMethod -Uri 'http://127.0.0.1:8765/api/benchmarks/run' `
    -Method Post -ContentType 'application/json' -Body $payload
```

将 `execution` 改为 `workflow` 可对比固定流程；将 `human_policy` 改为 `pause` 可检查人工确认门禁。真实调用需要主动设置 `mode='live'`。

接口为同步执行，长评测需保持请求等待；本次尚未实现后台队列、取消或逐样本续跑。后续评测应增加独立业务标注、重复运行及相同模型/数据/预算下的对照，不以这次入口修正宣称模型效果提升。

代码回归：`python tests/run_offline.py`，其中真实调用分支由模拟客户端验证，不访问模型服务。
