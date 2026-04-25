# CLAUDE.md

## Active Scene VLM Loop Hard Rules (2026-04-05)

当前如果任务涉及 `scene-aware Blender render + Qwen3.5 freeform reviewer loop`，必须遵守以下硬规则：

### 当前唯一主线目录

`/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/evolution_scene_v7_full20_rotation4_agent_round0_nostage35_20260404`

不要跑到旧目录，不要把旧目录误认为当前活跃输出目录。

### 角色定义

Codex 在这里不是“执行一轮实验然后汇报”的助手，而是 loop 里的人工替身。  
必须持续参与：

`render -> 送给 VLM -> 读取 VLM 自由文本 -> 自己决定下一轮 Blender 修改 -> rerender -> 再送 VLM -> repeat`

### 明确禁止

以下行为一律视为**未完成任务**：

1. 只启动 tmux / monitor / 后台进程然后退出
2. 只完成一次 render + review 然后退出
3. 只确认 VLM 能返回文本然后退出
4. 把持续迭代外包给脚本后自己停止
5. 只汇报路径、状态、日志，而不继续推进下一轮

### 强制工作方式

1. 必须优先读取 `reviews/*_trace.json`
2. 必须优先读取 `attempts[-1].assistant_text`
3. 必须根据自由文本亲自决定下一轮动作
4. 不能只依赖 `agg score`
5. 不能只依赖旧 controller 的自动选动作

### 停止条件

只有下面两种情况才允许停止某个 pair / 整个任务：

1. VLM 明确给出 `keep / acceptable / good enough / 可以了`
2. 用户明确手动叫停

除此之外，一律视为未完成，不允许主动退出，不允许“下班”。

### 出现故障时的默认行为

如果 loop 中出现任何异常，Codex 必须先修复再继续，不能停下来等用户下一条命令，包括但不限于：

1. tmux 断掉
2. monitor 退出
3. reviewer OOM
4. Blender 渲染失败
5. 某个 pair 长时间不再前进
6. 某一轮只有 render 没有 review

### 自检规则

如果准备结束当前回复，必须先检查：

1. 是否已经读取了最新一轮 VLM 自由文本
2. 是否已经基于该文本做出新的渲染决策
3. 是否已经真正触发了下一轮 render/review
4. 是否仍有未被 VLM 接受的 pair

只要第 4 条答案是“有”，就不能停止。

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

保持使用中文回复、使用中文写文档，使用英文思考和搜索

## Repository Overview

DataEvolver is a multi-project repository containing:

1. **Main DataEvolver Pipeline**: VLM training pipeline for dataset synthesis and evaluation
2. **feishu-claude-code**: Feishu bot that bridges to local Claude Code CLI
3. **obsidian-skills**: Agent Skills for Obsidian integration
4. **mcp-servers**: MCP server implementations (claude-review, feishu-bridge, llm-chat, minimax-chat)
5. **skills**: Custom DataEvolver pipeline skills following Agent Skills specification

## Project Structure

```
DataEvolver/
├── feishu-claude-code/     # Feishu bot (Python, has own .git)
│   ├── main.py             # Entry point with WebSocket event loop
│   ├── feishu_client.py    # Feishu API wrapper
│   ├── session_store.py    # Session persistence
│   ├── commands.py         # Slash command parser
│   └── requirements.txt    # lark-oapi, python-dotenv
├── obsidian-skills/        # Obsidian integration (has own .git)
├── mcp-servers/            # MCP server implementations
│   ├── claude-review/      # Codex ↔ Claude Code bridge
│   ├── feishu-bridge/
│   ├── llm-chat/
│   └── minimax-chat/
├── skills/                 # DataEvolver pipeline skills
│   ├── dataset-eval-pipeline/
│   ├── dataset-synthesis-gate/
│   ├── experiment-bridge/
│   ├── analyze-results/
│   ├── idea-discovery/
│   └── research-pipeline/
├── refine-logs/            # Runtime artifacts and workflow outputs
└── docs/                   # Documentation
```

## Development Commands

### feishu-claude-code

```bash
cd feishu-claude-code

# Setup
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Configure (copy .env.example to .env and fill in Feishu credentials)
cp .env.example .env

# Run
python main.py

# Deploy (macOS)
cp deploy/feishu-claude.plist ~/Library/LaunchAgents/com.feishu-claude.bot.plist
launchctl load ~/Library/LaunchAgents/com.feishu-claude.bot.plist

# Deploy (Linux)
sudo cp deploy/feishu-claude.service /etc/systemd/system/
sudo systemctl enable feishu-claude
sudo systemctl start feishu-claude
```

### DataEvolver Pipeline Skills

Skills are invoked via slash commands in Claude Code:

```bash
# Dataset evaluation pipeline
/dataset-eval-pipeline "your task description"

# Or via Python script
python skills/dataset-synthesis-gate/scripts/dataset_readiness_gate.py \
  --idea-path refine-logs/FINAL_PROPOSAL.md \
  --synthesis-mode dual \
  --reports-dir refine-logs
```

### MCP Servers

```bash
# Install claude-review MCP into Codex
mkdir -p ~/.codex/mcp-servers/claude-review
cp mcp-servers/claude-review/server.py ~/.codex/mcp-servers/claude-review/
codex mcp add claude-review -- python3 ~/.codex/mcp-servers/claude-review/server.py
```

## Architecture Notes

### feishu-claude-code Architecture

```
┌──────────┐  WebSocket  ┌────────────────┐  subprocess  ┌────────────┐
│  飞书 App │◄───────────►│ feishu-claude  │─────────────►│ claude CLI │
│  (用户)   │  长连接      │  (main.py)     │ stream-json  │  (本机)     │
└──────────┘             └────────────────┘              └────────────┘
```

- Uses Feishu WebSocket long connection (no public IP needed)
- Streams Claude output via `--print --output-format stream-json`
- Updates Feishu cards in real-time using patch API
- Session persistence via JSON files in session_store.py
- Watchdog thread restarts process every 4 hours to prevent WebSocket staleness
- Per-user message queue locks prevent concurrent session creation

### DataEvolver Pipeline Workflows

The main pipeline follows a 4-workflow structure:

1. **Workflow 1**: `/idea-discovery` → dataset evaluation → refine proposal
2. **Workflow 2**: `/dataset-synthesis-gate` (synthesis + QC + gate)
3. **Workflow 3**: `/experiment-bridge` (evaluator-VLM training)
4. **Workflow 4**: `/analyze-results` (results analysis)

Key constants:
- `DATASET_GATE = true`
- `SYNTHESIS_MODE = dual` (Blender + T2I)
- `RENDER_OUTPUT_MODE = both`
- `DATA_MERGE_MODE = fill-gap`

Runtime artifacts are written to `refine-logs/`:
- `FINAL_PROPOSAL.md`
- `DATASET_READINESS.md`
- `RENDER_QC_REPORT.md`
- `EVAL_MODEL_SPEC.md`
- `EVAL_BENCHMARK_REPORT.md`

### MCP claude-review Bridge

Bridges Codex (executor) with Claude Code CLI (reviewer):

- **Sync tools**: `review`, `review_reply` (for short prompts)
- **Async tools**: `review_start`, `review_reply_start`, `review_status` (for long prompts >120s)
- Reviewer runs in non-interactive `-p` mode with no tools by default
- Job state persisted to `~/.codex/state/claude-review/jobs/`

### Codex Review Workflow（推荐工作流程）

**通过 Codex MCP 连接 codex5.4high 进行代码审查**

Codex 5.4 High 是你最坚强、可靠的 reviewer。在处理有难度的问题时，建议采用以下迭代审查流程：

1. **提出方案**：完成初步的方案设计、想法或代码实现
2. **Codex Review**：使用 `mcp__codex__codex` 工具将方案提交给 Codex 进行审查
3. **修改优化**：根据 Codex 的反馈意见进行修改和优化
4. **再次审核**：将修改结果再次提交给 Codex 审核
5. **通过后进入下一步**：确认 Codex 审核通过后，再进入下一个开发阶段

**使用原则**：
- 简单问题可以直接解决，无需过度审查
- 复杂问题、架构设计、关键代码建议多次与 Codex 交互
- 利用 Codex 的高推理能力进行深度代码审查和方案验证

**示例调用**：
```python
# 通过 MCP 调用 Codex 进行审查
mcp__codex__codex(
    prompt="请审查以下代码实现：[代码内容]",
    model="codex-5.4-high",
    cwd="当前工作目录"
)
```

### Skills Architecture

All skills follow the [Agent Skills specification](https://agentskills.io/specification):
- Each skill has a `SKILL.md` file with frontmatter metadata
- `allowed-tools` defines tool permissions
- `argument-hint` provides usage guidance
- Skills can invoke other skills via `/skill-name` or the Skill tool

## Environment Variables

### feishu-claude-code

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `FEISHU_APP_ID` | Yes | - | Feishu App ID |
| `FEISHU_APP_SECRET` | Yes | - | Feishu App Secret |
| `DEFAULT_MODEL` | No | `claude-sonnet-4-6` | Default Claude model |
| `DEFAULT_CWD` | No | `~` | Default working directory |
| `PERMISSION_MODE` | No | `bypassPermissions` | Tool permission mode |
| `CLAUDE_CLI_PATH` | No | auto-detect | Claude CLI path |

### claude-review MCP

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAUDE_BIN` | `claude` | Claude CLI path |
| `CLAUDE_REVIEW_MODEL` | - | Reviewer model override |
| `CLAUDE_REVIEW_SYSTEM` | - | Default system prompt |
| `CLAUDE_REVIEW_TOOLS` | `""` | Tools override (empty = no tools) |
| `CLAUDE_REVIEW_TIMEOUT_SEC` | `600` | Subprocess timeout |

## Development Workflow

### Plan Node Default

对于非平凡任务（涉及多文件修改、架构决策、复杂逻辑），默认先进入计划模式：
- 使用 `EnterPlanMode` 工具进行方案设计
- 在计划模式中探索代码库、理解现有架构
- 设计实现方案并获得用户批准后再执行
- 简单任务（单文件小改动、明确的 bug 修复）可直接执行

### Subagent Strategy

自由使用子代理（Agent 工具）来并行处理独立任务：
- 代码搜索和探索：使用 `subagent_type=Explore`
- 复杂研究任务：使用 `subagent_type=general-purpose`
- 并行执行多个独立的搜索/分析任务
- 避免在主会话中执行重复性的⼤规模搜索

### Self-Improvement Loop

当发现错误或学到新经验时，更新项目记忆：
- 在 `refine-logs/lessons.md` 中记录经验教训
- 记录常见错误模式和解决方案
- 记录项目特定的最佳实践
- 在后续任务中参考这些经验

### Verification Before Done

完成任务前必须验证结果：
- 运行相关测试确保功能正常
- 检查生成的文件内容是否符合预期
- 验证配置文件语法正确
- 对于 pipeline 任务，检查输出日志和结果文件

### Demand Elegance

追求代码优雅但避免过度工程：
- 优先使用简洁、可读的实现
- 避免不必要的抽象和复杂度
- 遵循项目现有的代码风格和模式
- 只在真正需要时才引入新的依赖或架构

### Autonomous Bug Fixing

遇到错误时自主修复：
- 分析错误日志，定位根本原因
- 尝试多种解决方案，不要重复失败的操作
- 如果多次尝试失败，使用 `AskUserQuestion` 寻求帮助
- 修复后在 `lessons.md` 中记录问题和解决方案

### Task Management

使用结构化的任务管理：
- 对于复杂任务，使用 `TaskCreate` 创建任务列表
- 使用 `TaskUpdate` 跟踪任务进度
- 在 `refine-logs/todo.md` 中维护项目级待办事项
- 在 `refine-logs/lessons.md` 中记录经验教训

## Important Notes

- **feishu-claude-code** has its own git repository (submodule or separate clone)
- **obsidian-skills** has its own git repository (submodule or separate clone)
- Python projects use virtual environments (`.venv/`)
- The main DataEvolver directory is NOT a git repository (no `.git` at root)
- Skills can be invoked from Claude Code using `/skill-name` syntax
- MCP servers extend tool capabilities for both Claude Code and Codex
- Codex是你评级的reviewer，你可以和它进行多轮探讨并完善plan后再进行操作
- blender在服务器上的路径是/usr/bin/blender



## 远程服务器（重要）

- 不要在本地测试，你可以在本地改代码然后push到服务器的对应路径下并测试和训练等

- SSH：wwz（密钥免密登录）

- GPU：3块A800

- Conda 环境：`/home/wuwenzhuo/Qwen-VL-Series-Finetune/env `（Python 3.10 + PyTorch）

- 激活：`conda activate /home/wuwenzhuo/Qwen-VL-Series-Finetune/env`(如果还有依赖没有安装，则直接安装即可)

- 代码目录：`/aaaidata/zhangqisong/data_build`

- 后台运行用 `screen`：`screen -dmS exp0 bash -c '...'`

- 在服务器上运行程序时，必须使用tmux，如果没有tmux窗口则可以创建tmux窗口：tmux new -s claudecode-research-x(允许可以开多个tmux窗口)，这样我也可以直观看到程序进展

- 需要先查看显卡的占用情况，如果显卡空闲则优先使用空闲的显卡，允许使用多张显卡同时进行实验，如果有某张或者某几张显卡正在被占用，则用剩余显卡，如果都被占用则先暂停等待显卡空闲。

- 遇到长时间的进程，确保进程已经正常运行后，就可以停止监测，如果训练好了我会再唤醒你。

- 如果需要使用开放权重的模型，你可以去/huggingface/model_hub 进行ls操作，看看有没有合适的，如果没有则可以自行下载。

- qwen3.5模型的路径和信息是"/data/wuwenzhuo/Qwen3.5-35B-A3B"：

  ```bash
  ls -r "/data/wuwenzhuo/Qwen3.5-35B-A3B"
  vocab.json                      model.safetensors-00014-of-00014.safetensors  model.safetensors-00007-of-00014.safetensors  merges.txt
  video_preprocessor_config.json  model.safetensors-00013-of-00014.safetensors  model.safetensors-00006-of-00014.safetensors  LICENSE
  tokenizer.json                  model.safetensors-00012-of-00014.safetensors  model.safetensors-00005-of-00014.safetensors  generation_config.json
  tokenizer_config.json           model.safetensors-00011-of-00014.safetensors  model.safetensors-00004-of-00014.safetensors  config.json
  README.md                       model.safetensors-00010-of-00014.safetensors  model.safetensors-00003-of-00014.safetensors  chat_template.jinja
  preprocessor_config.json        model.safetensors-00009-of-00014.safetensors  model.safetensors-00002-of-00014.safetensors
  model.safetensors.index.json    model.safetensors-00008-of-00014.safetensors  model.safetensors-00001-of-00014.safetensors
  ```

  

- 如果要下载模型，请指定到：/huggingface/model_hub文件夹下，下载前一定要使用：export HF_ENDPOINT=https://hf-mirror.com，如果要下载数据集，指定到/huggingface/dataset_hub。训练的lora或者微调结果存放在/huggingface/train_model_hub/zhangqisong/research。
- 现在已有数据集（不一定有用），路径是/huggingface/dataset_hub/OriAnyV2_Train_Render/new（ls -r /huggingface/dataset_hub/OriAnyV2_Train_Render
  rename.py  README.md  __pycache__  new  image_metadata.json  cs.py  3.py  2.py  1.py  1.json）
