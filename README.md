# DataEvolver: Autonomous Synthetic Data Construction via VLM-Guided Iterative Rendering

**An end-to-end framework that evolves multimodal training datasets from natural language descriptions through iterative VLM-guided 3D rendering, with applications in controllable image editing.**

DataEvolver transforms a text description (e.g., "wooden chair") into quality-approved, multi-view rendered images through a six-stage pipeline. The core innovation is a **free-form VLM-guided rendering evolution loop**: a vision-language model critiques rendered scenes using natural language, and an AI agent interprets this feedback to iteratively refine rendering parameters until quality targets are met.

---

## Architecture

DataEvolver uses a **two-tier model architecture**:

- **Executor (Claude Code):** Runs experiments, writes code, manages the pipeline, and invokes the reviewer.
- **Reviewer (external LLM via MCP):** Provides adversarial review, idea critique, and quality assessment. Uses either Codex MCP (stateful GPT-5.4 threads) or llm-chat MCP (stateless, any OpenAI-compatible API).

The six-stage data construction pipeline transforms a natural-language object description into quality-approved rendered images:

```
Text Expansion (Stage 1)
  -> T2I Generation (Stage 2)
    -> SAM2 Segmentation (Stage 2.5)
      -> 3D Reconstruction via Hunyuan3D (Stage 3)
        -> Scene-Aware Blender Rendering (Stage 4)
          -> VLM Review + Feedback Loop (Stages 5.5-5.6)
            -> Metadata Merge (Stage 5)
```

Each stage is an independent Python script, coordinated via shell scripts and Claude Code slash skills.

---

## Directory Structure

| Directory | Purpose |
|-----------|---------|
| `pipeline/` | Six-stage data construction pipeline (text expansion, T2I, SAM2, 3D reconstruction, Blender rendering, VLM review, feedback application, metadata merge) |
| `scripts/` | Feedback loop orchestration (data building, rotation dataset construction, multi-view export, render feedback tuning, scene evolution), 68server H100 training/evaluation scripts |
| `skills/` | Claude Code skill definitions (SKILL.md files): research workflows, dataset synthesis, experiment bridge, scene agent loop, codex execution |
| `configs/` | YAML accelerator configurations (multi-GPU), JSON scene action space / template definitions, VLM review schemas, seed concept lists |
| `mcp-servers/` | MCP server implementations: llm-chat (generic OpenAI-compatible), minimax-chat (MiniMax variant), claude-review (Codex integration), feishu-bridge (notification) |
| `tools/` | Utility scripts: arXiv paper fetching, HuggingFace model download, Codex review override generation |
| `docs/` | Framework documentation: feedback loop internals, server reference, experiment records, team division plans, dataset guides, presentation outlines |
| `arxiv_report/` | Technical report LaTeX source, experiment evaluations, comparison analyses, code replication package |
| `paper/` | NeurIPS paper LaTeX draft, bibliography, figure assets, section files |

---

## Key Features

### 1. Six-Stage Data Construction Pipeline

End-to-end pipeline generating synthetic paired datasets for 3D-aware image editing. Takes a text description (e.g., "wooden chair") and produces 8 multi-view rendered images with quality assurance. Each stage can be run independently or chained via orchestration scripts.

### 2. Free-Form VLM-Guided Rendering Evolution

The distinguishing feature of DataEvolver. Unlike rigid score-based controllers, a VLM (Qwen3.5-35B-A3B) reviews rendered scenes using free-form natural language, identifying specific visual problems (flat lighting, color shift, floating objects, missing shadows). An AI agent interprets this semantic feedback and selects actions from a 24-element discrete rendering action space (adjusting lighting, object placement, scene environment, material properties). The loop repeats until the VLM confirms acceptable quality, with anti-oscillation protection and decaying step sizes for fine-grained convergence.

### 3. Scene-Aware Blender Rendering

3D meshes are inserted into a real Blender scene with preserved scene lighting (HDRI preservation), raycast-based ground detection for physically plausible placement, and Cycles path tracing at 1024x1024 resolution. This avoids common synthetic data artifacts like artificial studio lighting that clashes with the background.

### 4. Three Research Workflows

DataEvolver provides three Claude Code slash skill workflows:

- **Idea Discovery (Workflow 1):** Literature survey -> idea generation -> novelty check -> pilot experiment -> refine -> experiment plan.
- **Auto Review Loop (Workflow 2):** Multi-round review via external LLM, implementing fixes and re-reviewing until positive assessment (max 4 rounds, with state persistence for context compaction).
- **Paper Writing (Workflow 3):** Narrative -> outline -> figure generation -> LaTeX writing -> compilation -> auto-improvement loop.

These can be run individually or end-to-end via the `/research-pipeline` skill.

### 5. Cross-Model Collaboration

The design intentionally separates executor and reviewer roles. Claude Code executes experiments and manages the pipeline, while an external LLM (GPT-5.4 via Codex MCP, or any OpenAI-compatible model via llm-chat MCP) provides independent adversarial review. This avoids the self-play critique limitations of single-model systems.

### 6. Feedback Loop for Iterative Dataset Improvement

A complete outer loop architecture: evaluate a trained model on a benchmark (SpatialEdit-Bench), identify weak angles and underperforming objects, generate new training data for those specific failure modes, merge into an augmented dataset, retrain, and re-evaluate. Per-round diagnostics track weak angles, regression guards, object outliers, and cost.

---

## Quick Start

### Prerequisites

- Python 3.10+
- Blender 3.6+ (for Stage 4 rendering)
- CUDA-capable GPU (for 3D reconstruction and VLM inference)
- Claude Code CLI
- Codex MCP or llm-chat MCP (for external reviewer integration)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/DataEvolver.git
cd DataEvolver

# Install all skills globally
cp -r skills/* ~/.claude/skills/

# Install a single skill
cp -r skills/auto-review-loop ~/.claude/skills/
```

### MCP Server Setup

**For Codex MCP (GPT-5.4 reviewer):**

```bash
npm install -g @openai/codex
codex setup        # set model = "gpt-5.4" in ~/.codex/config.toml
claude mcp add codex -s user -- codex mcp-server
```

**For llm-chat MCP (any OpenAI-compatible API):**

Add to `~/.claude/settings.json`:
```json
{
  "mcpServers": {
    "llm-chat": {
      "command": "/usr/bin/python3",
      "args": ["/path/to/mcp-servers/llm-chat/server.py"],
      "env": {
        "LLM_API_KEY": "your-key",
        "LLM_BASE_URL": "https://api.deepseek.com/v1",
        "LLM_MODEL": "deepseek-chat"
      }
    }
  }
}
```

### Running the Pipeline

```bash
# Stage 1: Text expansion
python pipeline/stage1_text_expansion.py

# Stage 2: T2I image generation
python pipeline/stage2_t2i_generate.py

# Stage 2.5: Foreground segmentation
python pipeline/stage2_5_sam2_segment.py

# Stage 3: 3D reconstruction
python pipeline/stage3_image_to_3d.py

# Stage 4: Blender rendering
bash pipeline/stage4_batch_render.sh

# Stage 5.5: VLM review
python pipeline/stage5_5_vlm_review.py

# Stage 5.6: Feedback application
python pipeline/stage5_6_feedback_apply.py

# Stage 5: Metadata merge
python pipeline/stage5_merge_metadata.py
```

### Using Research Workflows

```bash
# Full research pipeline (Idea Discovery -> Auto Review -> Paper Writing)
/research-pipeline "your research direction"

# Individual workflows
/idea-discovery "research direction"
/auto-review-loop "experiment description"
/paper-writing "paper outline"
```

---

## Research Workflows

| Workflow | Orchestrator Skill | Description |
|----------|--------------------|-------------|
| 1: Idea Discovery | `/idea-discovery` | Literature survey -> brainstorm -> novelty check -> pilot experiment -> refine -> experiment plan |
| 2: Auto Review Loop | `/auto-review-loop` | External LLM review -> fix -> experiment -> re-review (max 4 rounds) with state persistence |
| 3: Paper Writing | `/paper-writing` | Narrative -> outline -> figures -> LaTeX -> PDF compilation -> auto-improvement loop |
| End-to-end | `/research-pipeline` | Workflows 1 through 3 in sequence |

---

## Applications

DataEvolver has been instantiated to construct the **DataEvolver-Rotate** rotation editing dataset (50+ objects, 350+ pairs), where each sample pairs a canonical front-view image with a target view specified by a natural-language prompt. LoRA fine-tuning of Qwen Image Edit 2511 on this dataset shows measurable improvements over the base model on PSNR, SSIM, CLIP-I, and LPIPS, validating the downstream training value of automatically constructed data.

The pipeline is designed to generalize to other geometry-controlled editing tasks: lighting variation, material substitution, scale change, and scene composition.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## References

- Qwen Image Edit 2511: instruction-following image editing model
- Hunyuan3D-2.1: single-image 3D reconstruction
- Qwen3.5-35B-A3B: VLM reviewer with extended thinking
- InstructPix2Pix: learning to follow image editing instructions (CVPR 2023)
- SpatialEdit-Bench: spatial editing benchmark for evaluation
