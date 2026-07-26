# AGENT_SETUP.md

> **You are an AI agent.** This document tells you how to take a fresh Linux
> GPU machine to a state where DataEvolver can build datasets end-to-end. A
> human can read it too, but the prose is written for you.

The user has asked you to set up DataEvolver. Goal state: `bash
src/dataevolver/cli/run_all.sh` runs the asset pipeline, and the VLM review
stages (`python -m dataevolver.annotation.vlm_review_stage`, `python -m
dataevolver.agents.feedback_apply`) can be launched. The user only needs
**Linux, an NVIDIA GPU, git, and Python 3.10+** pre-installed. Everything
else — the Python environment, model weights (~120 GB for the default set),
and Blender — is handled by you, **with the user's explicit confirmation
before anything large is downloaded**.

Typical trigger phrases from the user:

> 请阅读 AGENT_SETUP.md 并帮我完成所有配置
>
> "Read AGENT_SETUP.md and complete the setup for me."

---

## TL;DR — the flow you will execute

```text
probe (read-only)  →  plan (print, no side effects)  →  confirm the bill with the user
                   →  execute (idempotent)           →  verify (doctor + smoke)
```

Never skip the confirm step. Downloading ~120 GB without showing the user
the bill first is a failure, even if the downloads succeed.

---

## 0. Identify yourself

- If you read `CLAUDE.md` first → you are **Claude Code**.
- If you read `AGENTS.md` first → you are **Codex**.
- If your environment supports skills, `.agents/skills/dataevolver-onboarding/`
  contains the setup interview (route, runtime location, install policy,
  model strategy, work/output paths; plus GPU policy questions for
  `world_model_scene` / paper-validation routes). Use it when the user's
  intent is unclear; skip it when they simply said "set everything up with
  defaults".

Defaults when the user does not specify otherwise:

| Setting | Default |
|---|---|
| Profile | `default` (core pipeline models) |
| Workspace root | the directory this repo was cloned into (`$PWD`) |
| Model root | `<workspace>/models` |
| Python env | `uv venv .venv --python 3.10 --system-site-packages` (conda fallback) |

This document covers the core pipeline (`default` profile). For HYWorld /
WorldMirror scene reconstruction use `--profile world_model` and follow the
"Optional HYWorld / WorldMirror Scene Reconstruction" section of the README
after the core setup below.

---

## 1. Probe the machine (read-only)

Run these and summarize the results for the user in a short table. None of
them modify anything.

```bash
uname -a
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv
nvcc --version || cat /usr/local/cuda/version.json 2>/dev/null
df -h . && df -h <MODEL_ROOT parent>
lscpu | head -20 && free -g
python3 --version; git --version
command -v uv uvx conda hf aria2c blender
```

Fit rules to apply:

| Resource | Requirement | If not met |
|---|---|---|
| NVIDIA GPU | ≥ 24 GB VRAM for generation/render stages | Stop and ask — CPU-only setup is not supported |
| VRAM for VLM reviewer | ≥ 80 GB (single card or sharded across cards) | Offer to set up everything except the local VLM reviewer; the review stage then needs a remote/API VLM the user provides |
| Free disk at model root | ≥ 200 GB recommended (~120 GB models + workspace) | Ask the user for a different `--model-root` or a reduced model set |
| CUDA driver | Supports one of cu118 / cu121 / cu124 / cu128 wheels | Stop and ask before touching drivers — never install drivers yourself |

GPUs already running other users' jobs or serving vLLM/VLM services are
off-limits: record them as reservations, never kill them.

---

## 2. Produce the plan (no side effects)

Run the planner. It prints the preflight, GPU plan, environment plan, the
model manifest with exact download commands, and the config plan; it
executes nothing:

```bash
bash src/dataevolver/cli/bootstrap_dataevolver_default.sh \
  --profile default \
  --model-root <MODEL_ROOT> \
  --workspace-root "$PWD" \
  --dry-run \
  --write-local-config
```

`--write-local-config` writes three non-sensitive local files under
`.dataevolver/local/` (`ENVIRONMENT.md`, `env.config.json`,
`env.sh.example`). That directory is gitignored.

**User-provided models:** if the user already has weights on disk, do not
plan a download for them. Record the existing path and use it for the
corresponding environment variable in step 4.6. Replacement models that are
not the defaults should be recorded as `custom` with a note that
compatibility is unverified.

---

## 3. Show the bill and get confirmation

Present a bill in this shape and wait for an explicit yes:

| Item | Source | Size | Target | Env var |
|---|---|---|---|---|
| Qwen-Image-2512 | HF `Qwen/Qwen-Image-2512` | ~56 GB | `<MODEL_ROOT>/Qwen-Image-2512` | `QWEN_IMAGE_MODEL_PATH` |
| SAM3 checkpoint | HF `facebook/sam3` (**gated**) | ~2 GB | `<MODEL_ROOT>/sam3` | `SAM3_CKPT` |
| SAM3 source | `github.com/facebookresearch/sam3` | small | `<MODEL_ROOT>/src/sam3` | `SAM3_DIR` |
| Hunyuan3D-2.1 weights | HF `tencent/Hunyuan3D-2.1` | ~20 GB | `<MODEL_ROOT>/Hunyuan3D-2.1` | `MODEL_HUB`, `PAINT_MODEL_HUB` |
| Hunyuan3D-2.1 source | `github.com/Tencent-Hunyuan/Hunyuan3D-2.1` | small | `<MODEL_ROOT>/src/Hunyuan3D-2.1` | `HUNYUAN3D_REPO` |
| RealESRGAN x4plus | Real-ESRGAN release asset | ~65 MB | `$HUNYUAN3D_REPO/hy3dpaint/ckpt/` | `REALESRGAN_CKPT` |
| DINOv2 Giant | HF `facebook/dinov2-giant` | ~5 GB | `<MODEL_ROOT>/dinov2-giant` | `DINO_MODEL_PATH` |
| Qwen3.5-35B-A3B | HF `Qwen/Qwen3.5-35B-A3B` | ~35 GB | `<MODEL_ROOT>/Qwen3.5-35B-A3B` | `VLM_MODEL_PATH` |
| Blender 5.2 | blender.org official release | ~350 MB | `<MODEL_ROOT>/blender-5.2.0` | `BLENDER_BIN` |

Rules:

- Do not start any download over 1 GB before the user confirms the bill.
- Default download route is the **hf-mirror.com mirror with `hfd`**
  (section 4.4). Direct huggingface.co also works when the network allows;
  gated access is always granted on huggingface.co itself either way.
- **SAM3 is gated on Hugging Face.** The user must request access on the
  model page themselves and have `HF_TOKEN` exported in their shell. Never
  ask the user to paste a token into chat, and never write a token to disk.
- Items the user already has on disk are listed in the bill as
  `existing — skipped`.

---

## 4. Execute (idempotent — skip anything that already exists)

### 4.1 Python environment and package entry points

Prefer `uv`. On shared GPU servers use `--system-site-packages` so an
already-validated PyTorch build is reused:

```bash
uv venv .venv --python 3.10 --system-site-packages
source .venv/bin/activate
python -m pip install -e .
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Only if that torch check fails or reports no CUDA, install the wheel
matching the driver you probed (pick the highest supported): `cu128`,
`cu124`, `cu121`, or `cu118`:

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4.2 Python dependencies

```bash
uv pip install \
  "numpy<2" "tokenizers==0.22.1" \
  diffsynth transformers accelerate diffusers safetensors \
  pillow opencv-python scipy scikit-image imageio trimesh \
  rembg[gpu] anthropic qwen-vl-utils lpips basicsr realesrgan \
  iopath timm ftfy moviepy==1.0.3 nerfview rtree \
  opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
```

### 4.3 Source checkouts

```bash
mkdir -p <MODEL_ROOT>/src
git clone https://github.com/facebookresearch/sam3.git <MODEL_ROOT>/src/sam3
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git <MODEL_ROOT>/src/Hunyuan3D-2.1
```

### 4.4 Model weights

Default download route: the **hf-mirror.com mirror with the `hfd`
downloader** — multi-connection via `aria2c`, resumable, and much faster on
networks where huggingface.co is slow or unreachable. If the machine has
good direct access to huggingface.co, you may instead run exactly the
`hf download` commands the planner printed, unchanged.

```bash
export HF_ENDPOINT=https://hf-mirror.com
curl -L -o hfd.sh https://hf-mirror.com/hfd/hfd.sh && chmod a+x hfd.sh
./hfd.sh --help   # confirm current flags — hfd evolves with the mirror

# One call per model on the confirmed bill, for example:
./hfd.sh Qwen/Qwen-Image-2512 --local-dir <MODEL_ROOT>/Qwen-Image-2512
./hfd.sh tencent/Hunyuan3D-2.1 --local-dir <MODEL_ROOT>/Hunyuan3D-2.1
./hfd.sh facebook/dinov2-giant --local-dir <MODEL_ROOT>/dinov2-giant
./hfd.sh Qwen/Qwen3.5-35B-A3B --local-dir <MODEL_ROOT>/Qwen3.5-35B-A3B

# Gated repo (SAM3): access must already be granted on huggingface.co
./hfd.sh facebook/sam3 --hf_username <hf_username> --hf_token "$HF_TOKEN" \
  --local-dir <MODEL_ROOT>/sam3
```

Requirements:

- `hfd` uses `aria2c` when available (fastest) and falls back to `wget`
  (`--tool wget`). Installing `aria2` is a system package — ask the user
  before installing anything system-wide.
- `HF_TOKEN` must come from the user's shell environment; pass it through
  as `"$HF_TOKEN"`, never a literal value.
- Interrupted downloads are safe to re-run — `hfd` resumes where it left off.
- Skip any target directory that already exists and is non-empty; say so.
- The standard CLI also works through the mirror:
  `HF_ENDPOINT=https://hf-mirror.com hf download <repo> --local-dir <target>`.
- Download RealESRGAN into the Hunyuan3D checkout:

```bash
curl -L -o <MODEL_ROOT>/src/Hunyuan3D-2.1/hy3dpaint/ckpt/RealESRGAN_x4plus.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
```

### 4.5 Blender

Blender is delivered as an official portable tarball — do not ask the user
to install it system-wide:

```bash
curl -L -o /tmp/blender-5.2.0-linux-x64.tar.xz \
  https://download.blender.org/release/Blender5.2/blender-5.2.0-linux-x64.tar.xz
tar -xf /tmp/blender-5.2.0-linux-x64.tar.xz -C <MODEL_ROOT>
<MODEL_ROOT>/blender-5.2.0-linux-x64/blender --version
```

Any Blender ≥ 4.2 the user already has also works; point `BLENDER_BIN` at it.

### 4.6 Write the local environment file

Copy `.dataevolver/local/env.sh.example` to
`.dataevolver/local/env.remote.sh`, fill in every path from the bill
(including user-provided ones), and add `BLENDER_BIN`. Paths only — never
tokens. Tell the user this file must be sourced before every run.

### 4.7 Optional native extensions (only after nvcc check)

Compile Hunyuan3D's texture-paint extensions only if `nvcc` is present and
matches the torch CUDA version; otherwise skip and note that the
image-to-3D stage can run `--shape-only` until this is done:

```bash
uv pip install --no-build-isolation -e "$HUNYUAN3D_REPO/hy3dpaint/custom_rasterizer"
cd "$HUNYUAN3D_REPO/hy3dpaint/DifferentiableRenderer" && bash compile_mesh_painter.sh
```

---

## 5. Verify

```bash
source .dataevolver/local/env.remote.sh
"$BLENDER_BIN" --version
python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available(), "bf16:", torch.cuda.is_bf16_supported())
from transformers import AutoProcessor
from opentelemetry import trace
PY
ls "$QWEN_IMAGE_MODEL_PATH" "$MODEL_HUB" "$DINO_MODEL_PATH" "$VLM_MODEL_PATH" >/dev/null && echo "model paths OK"
```

Then, with the user's OK (this uses GPU time), run the single-object smoke:

```bash
python -m dataevolver.workflows.stages.t2i_generate --ids obj_001 --steps 1 --height 512 --width 512 --device cuda:0
python -m dataevolver.workflows.stages.sam_segment --ids obj_001 --device cuda:0
python -m dataevolver.workflows.stages.image_to_3d --ids obj_001 --shape-only --device cuda:0
```

Finish by giving the user a short setup report: what was installed, what was
skipped as already present, every env var and its path, anything deferred
(e.g. paint extensions), and the exact next commands
(`source .dataevolver/local/env.remote.sh && bash src/dataevolver/cli/run_all.sh`).

---

## 6. Authorization scope

You ARE authorised to:

- Run the probe commands, the planner, and the `dataevolver` package entry
  points in this repo
- Create the project venv and install Python packages **for this project**
- Download the models on the confirmed bill into `<MODEL_ROOT>`
- Clone the SAM3 and Hunyuan3D source repos
- Extract Blender into `<MODEL_ROOT>`
- Write non-sensitive config under `.dataevolver/local/`

You are NOT authorised to:

- Write API keys or tokens to any file, or echo them into chat
- Install system packages, NVIDIA drivers, or CUDA toolkits
- Edit `~/.bashrc`, `~/.zshrc`, SSH config, or anything in `~` outside
  standard model/cache directories
- Download anything not on the confirmed bill
- Bypass Hugging Face gated-access checks
- Commit, push, or modify files outside this repo and `<MODEL_ROOT>`
- Occupy or kill GPUs that are running other users' jobs or serving
  vLLM/VLM services (check `nvidia-smi` first; record them as reservations)

---

## 7. If something is off-script, stop and ask

- No NVIDIA GPU, or driver too old for every supported wheel
- Disk shortfall that survives choosing a different `--model-root`
- SAM3 gated access not yet granted (offer to continue with the rest and
  return to segmentation later)
- The user wants replacement models — record them, warn that compatibility
  is unverified, and do not silently swap pipeline stages
- Air-gapped machine (downloads impossible — ask how weights will arrive)
- Anything that would require sudo

Self-modifying the user's environment without telling them is a worse
failure mode than pausing for one round trip.

---

## 8. After setup

- One-shot asset pipeline: `bash src/dataevolver/cli/run_all.sh`
- VLM review and bounded repair: `python -m dataevolver.annotation.vlm_review_stage`
  and `python -m dataevolver.agents.feedback_apply` (see README "Production
  Smoke Tests" and the pipeline overview)
- HYWorld / WorldMirror scene reconstruction: rerun the planner with
  `--profile world_model` and follow the README's HYWorld section
- Setup interview skill: `.agents/skills/dataevolver-onboarding/`
- Multi-GPU convention: `--gpus "0,1;2,3"` — `;` separates workers, `,`
  separates GPUs within a worker
