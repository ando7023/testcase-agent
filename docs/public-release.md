# 公开演示版

本版保留 Agentic Harness、反馈重规划、Memory、RAG、人工确认、评审与导出能力。默认业务使用新编写的虚构设备借用申请；接口、字段、三条状态转换和演示事件仅用于本仓库的测试，不对应真实服务。

## 发布范围

- 包含 `app/`、`tests/`、`scripts/`、公开说明文档、依赖列表及空凭证配置示例。
- 不包含原业务文档、运行结果、Memory、Trace、数据库、个人配置、临时文件或密钥。
- 原本地文件和旧提交保存在 `.local-private/`，该目录被忽略。不要上传整个工作目录，也不要将本地备份作为 Release 附件。
- 历史修复日志保留原失败事实和验证边界，旧领域名称概括为“历史私有领域”。其中相对路径指向的私有运行证据不随仓库发布。

## 历史与备份

公开 `main` 应为一个全新的根提交，不能以含私有内容的旧提交为父提交。这样普通 `git push origin main` 不会发送旧业务历史。

清理前的提交保存在本地 `.local-private/pre-public.bundle`，文件副本保存在 `.local-private/pre-public-files/`。Git 本地 reflog/不可达对象可能仍能恢复旧内容；它们不属于公开 `main` 的可达历史。不要把 `.git/` 打包发布。

## 验证

```powershell
python tests/run_offline.py
python scripts/check_public_release.py --ref main
git log --oneline main
git status --short
```

发布检查会遍历指定分支可达提交，检查私有文件路径、常见令牌、私钥、个人绝对路径，并在本地 `.env` 存在时检查是否误提交了同值凭证；不打印凭证。它是辅助检查，不能保证发现所有敏感业务信息。

此版本改变了内置领域类型和评测 ID；旧本地项目不做自动迁移。测试公开演示时使用全新的 `CASEFORGE_DATA_DIR`，保留旧数据用于本地历史核查。

本轮不自动发布、不强推远程，也不替项目选择开源许可证。确认本地提交与目标远程后，可正常推送 `main`；远程若已有其他提交，应先核查，不能用强推覆盖。
