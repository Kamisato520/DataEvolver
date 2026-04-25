# DataEvolver

DataEvolver 是一个自动化多模态数据集构建流水线，通过自然语言描述驱动 VLM 引导的迭代式 3D 渲染，产出高质量合成训练数据，应用于可控图像编辑任务。

当前流水线涵盖六大阶段：文本扩展 -> T2I 图像生成 -> SAM2 前景分割 -> 3D 重建(Hunyuan3D) -> 场景感知 Blender 渲染 -> VLM 审查 + 反馈闭环 -> 元数据合并。

核心创新：**自由形式 VLM 引导的渲染进化闭环**——VLM 用自然语言审查渲染质量，AI agent 读取反馈并自动调整渲染参数，直至达到质量标准。

## 核心 Skills

- `skills/dataset-synthesis-gate/` — 双路径合成 + 质检 + 门禁
- `skills/dataset-eval-pipeline/` — 端到端数据集评估流水线
- `skills/experiment-bridge/` — 评估型 VLM 微调实验
- `skills/analyze-results/` — 结果分析与报告
- `skills/research-pipeline/` — 总编排（可选）

## 研究工作流

| 工作流 | 命令 | 说明 |
|--------|------|------|
| 1: Idea 发现 | `/idea-discovery` | 文献调研 -> 头脑风暴 -> 查新 -> 试点 -> 精炼 -> 实验计划 |
| 2: 自动审查回路 | `/auto-review-loop` | 外部 LLM 审查 -> 修复 -> 实验 -> 重新审查（最多 4 轮） |
| 3: 论文写作 | `/paper-writing` | 叙事 -> 大纲 -> 图表 -> LaTeX -> PDF -> 自动改进 |
| 端到端 | `/research-pipeline` | 工作流 1-3 顺序执行 |

## 安装与快速开始

```bash
git clone https://github.com/your-org/DataEvolver.git
cd DataEvolver

# 安装所有技能
cp -r skills/* ~/.claude/skills/

# 运行数据构建流水线
python pipeline/stage1_text_expansion.py
bash pipeline/stage4_batch_render.sh

# 使用 skill 入口
/research-pipeline "你的研究方向"
```

## 关键文档

- `docs/FEEDBACK_LOOP_FRAMEWORK.md` — 反馈闭环架构
- `docs/feedback_loop_internals.md` — 内部实现参考
- `docs/server_reference.md` — 服务器环境配置
- `arxiv_report/paper/tech_report_tex/` — 技术报告 LaTeX 源码

## 许可

MIT License — 详见 [LICENSE](LICENSE) 文件。
