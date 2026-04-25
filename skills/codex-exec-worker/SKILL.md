---
name: codex-exec-worker
description: "Launch a local detached PowerShell worker that runs `codex exec` non-interactively against the current workspace, with prompt handoff, run logs, and stop conditions. Use when user wants a second Codex instance to autonomously edit code, continue long-running automation, or work in parallel outside the current chat."
argument-hint: [task-or-prompt-path]
allowed-tools: Bash(*), Read, Write, Edit, Grep, Glob
---

# Codex Exec Worker

在本地另起一个 PowerShell 窗口，用 `codex exec` 启动一个 non-interactive worker。

适用场景：

- 用户明确想让另一个 Codex 实例自动执行修改
- 任务较长，适合脱离当前对话持续跑
- 需要把上下文、约束、验证命令打包交给外部 worker

不适用场景：

- 只改 1-2 个小地方
- 当前对话和外部 worker 会同时改同一批文件
- 任务边界不清晰、没有停止条件

## 当前仓库默认约束

- 本仓库是纯 Markdown skill 体系，skill 只负责调度，不要额外生成 README / CHANGELOG
- 启动外部 worker 时默认使用 `powershell -NoProfile`
  原因：当前机器的 PowerShell profile 会触发 `OpenSpecCompletion.ps1` parser error
- 默认 sandbox 用 `workspace-write`
- 默认把运行产物写到：`<workspace>/.codex-exec-runs/<timestamp_label>/`

## 启动前必须整理的四件事

在写 prompt 之前，先明确：

1. 任务目标
2. 允许写入的文件/目录
3. 禁止修改的文件/目录
4. 停止条件和验证命令

如果当前对话还要继续改代码，必须先切分 ownership：

- 当前对话改哪些文件
- 外部 `codex exec` worker 改哪些文件

不要让两个执行者同时写同一文件。

## Prompt 组织模板

给外部 worker 的 prompt 尽量包含：

```text
Repo root: <absolute path>

Task:
- <要完成的事情>

Allowed write scope:
- <目录或文件>

Do not touch:
- <目录或文件>

Requirements:
- <实现约束>
- <风格约束>

Validation:
- <要运行的命令>

Stop when:
- <完成标准>

Final response:
- 简要说明改了什么
- 列出未解决风险
```

## 启动脚本

使用脚本：

- `skills/codex-exec-worker/scripts/launch_codex_exec_worker.ps1`

最小命令：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\skills\codex-exec-worker\scripts\launch_codex_exec_worker.ps1 `
  -PromptFile .\codex_prompt.txt `
  -Workspace .
```

推荐命令：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\skills\codex-exec-worker\scripts\launch_codex_exec_worker.ps1 `
  -PromptFile .\codex_prompt.txt `
  -Workspace . `
  -Label rotation8-feedback `
  -Json
```

如果只想先生成 prompt / manifest / runner，不立即启动 worker：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\skills\codex-exec-worker\scripts\launch_codex_exec_worker.ps1 `
  -PromptFile .\codex_prompt.txt `
  -Workspace . `
  -Label dry-run `
  -Json `
  -NoLaunch
```

如果需要给 worker 额外可写目录：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\skills\codex-exec-worker\scripts\launch_codex_exec_worker.ps1 `
  -PromptFile .\codex_prompt.txt `
  -Workspace . `
  -AddDir D:\some\other\path `
  -Json
```

## 运行产物

每次启动会生成一个 run 目录，里面至少有：

- `prompt.txt`
- `run_manifest.json`
- `worker-run.ps1`
- `last_message.txt`
- `events.jsonl` 或 `stream.log`
- `exit_status.json`

## 监控方式

查看最后总结：

```powershell
Get-Content .\.codex-exec-runs\<run>\last_message.txt
```

实时看事件流：

```powershell
Get-Content .\.codex-exec-runs\<run>\events.jsonl -Wait
```

如果没开 `-Json`，就看：

```powershell
Get-Content .\.codex-exec-runs\<run>\stream.log -Wait
```

## 完成后必须做的事

外部 worker 结束后：

1. 读 `last_message.txt`
2. 检查 git diff
3. 运行约定的验证命令
4. 再决定是接受结果、继续 refinement，还是重新发起一个 worker

## 续跑说明

如果用户明确要继续上一次 non-interactive 会话，不要重写旧 prompt。直接在目标 workspace 里手动运行：

```powershell
codex exec resume --last -C .
```

是否继续用新的 detached PowerShell，由当前上下文自行判断。
