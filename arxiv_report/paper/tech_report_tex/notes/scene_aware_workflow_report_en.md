# From Manual Blender Tuning to Self-Evolving Data Construction: A VLM- and Agent-Guided Workflow-as-Skill Framework for Object-Level Image Editing

## Abstract

Synthetic dataset construction for object-level image editing is often treated as a forward generation pipeline, but in practice it behaves like a closed-loop process of observation, diagnosis, parameter adjustment, regeneration, and validation. This report abstracts that process as a workflow-as-skill data engine: a reusable, stateful, and inspectable framework driven by staged artifacts, VLM/CV review, AI-agent actions, downstream probes, and explicit verdicts. The framework contains an inner loop for generation-time self-correction and an outer loop for validation-time self-improvement. We validate the idea on the original motivating task of scene-aware object rotation data construction, where the workflow produces train-ready paired data and reveals both useful weak-angle signal and an unresolved render-quality bottleneck. The resulting `inspect` verdict is not a failure, but a diagnostic outcome showing that quality gating must be strengthened before further scaling.

## 1. Introduction

This report begins from a small but persistent bottleneck in synthetic data construction. To build scene-aware object rotation pairs, each object must be rendered in a fixed environment with stable lighting, grounding, scale, viewpoint, and background consistency. In practice, this requires repeated Blender-based tuning [4]: render samples, inspect failures, diagnose whether the problem comes from placement, lighting, camera framing, material quality, or reconstruction artifacts, adjust parameters or scripts, and render again. The bottleneck is not merely that Blender is inconvenient. The deeper issue is that manual synthetic-data construction is already a human-in-the-loop closed process, but the loop is often implicit, slow, and difficult to scale.

This pain point becomes costly when the goal is modest but task-specific. A user may only want to add a small batch of new objects or quickly build a dataset for a new object-level edit, yet still needs rendering expertise, repeated trial and error, and sometimes exploratory code-level tuning through Blender MCP-style tooling or direct code edits. Without an explicit workflow, the same failures recur across objects and dataset rounds. Worse, if downstream training regresses, it is hard to trace whether the cause was a bad asset, a poor render, a weak instruction, a split issue, or the training configuration.

The central observation is that synthetic dataset construction should not be modeled as a one-shot pipeline. It is an iterative workflow: observe, diagnose, adjust, regenerate, and validate. This report proposes a workflow-as-skill data engine that makes this loop explicit. The engine turns a dataset request into staged artifacts, review signals, feedback actions, and acceptance outcomes. VLM/CV review and AI coding agents replace part of the human-driven Blender tuning process, while downstream training and evaluation provide feedback at the dataset-round level.

The framework is evaluated on the original motivating task: scene-aware object rotation data construction. This task is appropriate because it requires stable background context, object identity preservation, viewpoint control, and lighting and grounding consistency. It also forces a clear distinction between object rotation and camera-orbit multiview rendering. The case study is therefore not the general framework itself, but a concrete validation setting for testing whether the workflow can produce useful train-ready data and diagnose its own bottlenecks.

The main finding is diagnostic: the workflow surfaces useful weak-angle signal while exposing an unresolved render-quality bottleneck, demonstrating that both loops actively shape the outcome. Concrete metrics are deferred to Section 9.

This report makes three contributions:

- It reframes autonomous visual dataset construction as a closed-loop workflow-as-skill data engine rather than a one-shot generation pipeline.
- It defines a dual-loop design in which VLM/CV feedback and AI-agent actions support generation-time self-correction, while downstream probes support validation-time self-improvement.
- It instantiates the framework on scene-aware object rotation editing and shows that a mixed R1 result can become actionable diagnostic evidence rather than wasted trial and error.

## 2. Motivation: From Manual Blender Tuning to Agentic Dataset Construction

The motivating engineering problem is repeated render tuning. A scene-aware synthetic dataset cannot be judged only by whether an image file exists. The object must be complete, visible, grounded, plausibly lit, correctly scaled, and consistent with the target instruction. When any of these conditions fails, the next step is rarely a single deterministic fix. The builder must inspect the render, infer the failure mode, decide whether to adjust lighting, object scale, camera distance, support-plane handling, material settings, or filtering rules, and then regenerate the sample.

This pattern reveals why ad-hoc pipelines are fragile. A script sequence can generate images, masks, meshes, renders, and train pairs, but it does not by itself encode why an artifact should be trusted. It also does not preserve enough state to explain downstream regressions. If a model trained on the resulting dataset improves viewpoint alignment but loses identity consistency, the system needs to know which dataset round, object subset, render-quality signal, or validation subgroup produced the conflict.

Agentic dataset construction is a response to this closed-loop nature. The goal is not to remove all human judgment or claim that agents can replace expert dataset design. The goal is narrower and more practical: make the observation-diagnosis-adjustment loop explicit enough that VLM/CV feedback and AI coding agents can handle part of the routine tuning, while humans retain oversight at verdict boundaries. This turns manual Blender iteration into an inspectable workflow with persistent artifacts and bounded actions.

The same motivation applies beyond Blender rendering. Object-level editing data often requires controlled changes and invariant preservation, whether the source is synthetic rendering, collected real images, or a mixture of both. The framework developed here is therefore task-neutral at the abstraction level, even though the first validation case comes from scene-aware object rotation.

The report builds on the setting of instruction-guided image editing and editing datasets, where InstructPix2Pix and MagicBrush connect natural-language instructions with paired source-target examples [1,2]. It is also related to synthetic data factory work such as DatasetGAN, which uses generative models to produce labeled visual data with reduced manual annotation effort [3]. The present report differs in emphasis: it treats data construction as a stateful closed-loop workflow rather than only a dataset-generation procedure.

## 3. Related Work

**Instruction-guided image editing.** InstructPix2Pix [1] and MagicBrush [2] established the modern paradigm of natural-language image editing, both relying on curated paired supervision. Our work targets the upstream question of how such pairs are produced.

**Synthetic data factories.** DatasetGAN [3] and OmniGen [13] demonstrate that generative models can fabricate task-specific training data. We extend this line by treating fabrication as a stateful workflow with explicit verdict gating, rather than a one-shot pass.

**VLM-as-judge and feedback-driven scaling.** ImageReward [15] uses a learned reward to score generated images; VIEScore [8] applies a VLM as a structured evaluator. Our work adopts a similar evaluator role, but places it inside a dataset-construction workflow where review signals route regeneration and augmentation decisions..

**3D-aware view synthesis.** Zero-1-to-3 [14] renders Objaverse assets to train novel-view diffusion. Our pipeline shares the render-then-train recipe but inserts a VLM quality gate so that low-quality 3D reconstructions are filtered before they pollute downstream supervision.

**Self-improving training pipelines.** STaR [16] and Self-Rewarding Language Models [17] popularised iterative self-improvement loops in NLP. The dual-loop design here ports the same template to a multi-stage visual data engine, where the "judge" is a multimodal VLM and the "actions" are concrete pipeline-stage operations rather than token-level generations.

## 4. Workflow-as-Skill Data Engine

The proposed system is a workflow-as-skill data engine. It replaces a loose script chain with a reusable, stateful, and inspectable procedure that turns a dataset request into staged artifacts, feedback signals, actions, and verdicts. In the current formulation, the skill can be written as:

```text
Skill = (Stages, Controller, Review, Verdict)
```

`Stages` are the ordered construction modules and the artifacts they must produce. `Controller` is the AI agent or workflow manager that tracks state, routes samples, edits scripts, adjusts parameters, and decides which stage should run next. `Review` contains automatic CV signals, VLM assessments, human-readable traces, and downstream validation metrics. `Verdict` maps the observed evidence into actions such as `continue`, `inspect`, `regenerate`, `reject`, `stop_or_revert`, or `no_signal`.

This formalization is deliberately lightweight. It is not intended to prove dataset correctness. Its purpose is to make the operational contract explicit: every stage emits inspectable artifacts, every artifact remains attached to persistent state, every review signal can influence routing, and every construction round ends with a decision that prevents ambiguous interpretation.

**Table 1. Operational contract of the workflow-as-skill data engine**

| Component | Contract |
|---|---|
| Stage outputs | Construction modules emit prompts, masks, meshes, renders, review traces, train pairs, checkpoints, and comparison reports that can be inspected or reused. |
| Controller actions | The controller routes samples, adjusts parameters, edits workflow code when needed, triggers rerendering or regeneration, and manages state across rounds. |
| Review signals | Quality is assessed through automatic CV signals, VLM feedback during construction, and downstream benchmark behavior after training. |
| Feedback bounds | Updates remain bounded to actions such as rerendering, regenerating, filtering, rejecting, or delaying acceptance. |
| Verdict logic | A round is accepted only if useful improvements do not trigger regression guards; mixed evidence is routed to `inspect`. |

The value of this abstraction is traceability. A dataset engine should not only produce examples; it should explain why those examples are trustworthy enough to enter training. Persistent state makes failures traceable, review signals make quality visible, and verdicts make iteration decisions explicit. Downstream training remains important, but it functions as a validation probe for the data engine rather than the main method.

```text
Figure 1 (text mock-up): Workflow-as-Skill abstraction.

  Request ──> [ Stage_1 -> Stage_2 -> ... -> Stage_K ]
                            │
                            ▼
                    [ VLM Review ]
                            │
                            ▼
                    [ Verdict ∈ {accept, regenerate, reject} ]
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
          accept        regenerate      reject
       -> Dataset    -> Action a∈A    -> drop sample
                       -> re-enter
                          earlier stage
```

## 5. Dual-Loop Self-Evolving Dataset Construction

The workflow contains two connected loops. The inner loop operates during generation, while the outer loop operates after training and evaluation. Both loops share the same structure: an agent receives feedback from an oracle, acts within a bounded action space, and stops according to a convergence or verdict criterion. They differ mainly in time scale and target.

The inner loop is generation-time self-correction. A sample is rendered, then reviewed by VLM and CV signals for lighting, grounding, viewpoint, scale, object completeness, material plausibility, and background consistency. The controller uses this feedback to adjust Blender parameters, modify rendering scripts, improve asset placement, change camera or framing settings, filter low-quality samples, or trigger regeneration. This loop replaces part of the manual Blender tuning process that would otherwise require repeated expert inspection and code edits.

Concretely, the agent action space is deliberately practical rather than open-ended. It covers lighting and exposure adjustment, camera and framing correction, object scale and position correction, support-plane grounding, render filtering, asset regeneration, and limited script edits when repeated failures indicate that a rendering rule or controller behavior must change. These actions are bounded so that the workflow can improve samples without turning each failure into an untraceable manual rewrite.

The outer loop is deployment-time self-improvement. After a dataset round is exported, a downstream probe is trained, inference is run, metrics are computed, weak subgroups are identified, and the result is converted into an augmentation or filtering plan. That plan feeds the next data-construction round. In the current case study, this means weak-angle behavior can motivate targeted object additions, while regression guards can prevent automatic acceptance when new data introduces harmful noise.

**Table 2. Inner and outer loops in the data engine**

| Loop | Level | Feedback oracle | Action space | Verdict role |
|---|---|---|---|---|
| Inner loop | sample/render level | VLM/CV review of generated artifacts | adjust render parameters, edit scripts, rerender, regenerate, filter | decide whether a sample or asset is usable |
| Outer loop | dataset/training-round level | downstream LoRA probe and benchmark metrics | add targeted data, filter subsets, change quality gates, inspect or revert a round | decide whether a dataset round should continue |

```text
Figure 2 (text mock-up): Two isomorphic loops.

  Inner loop (generation-time):
     Render ─> VLM ─> verdict ─> action ─> Re-render ─> ...

  Outer loop (deployment-time):
     Train ─> Eval ─> compare.py ─> verdict ─> dataset_feedback_plan ─> Augment ─> Train ─> ...

  Both share: (agent, feedback oracle, action space, verdict).
```

The two-loop design explains why `inspect` is a productive outcome. A mixed result is not merely a failed experiment; it is a signal that the outer loop found a useful direction but the inner loop did not yet enforce sufficient quality. The correct response is therefore not to discard the framework, but to improve the inner quality gates before scaling further.

## 6. Data-Build Pipeline Instantiation

The data-build pipeline instantiates the workflow abstraction as a sequence of artifact-producing stages. It begins with object concept expansion because the data engine needs explicit object identities, descriptions, and synthesis prompts before visual generation can be controlled. White-background object generation then produces isolated object images, which are easier to segment and reconstruct than images with scene clutter.

Foreground segmentation converts each generated object image into an RGBA asset suitable for image-to-3D reconstruction. This stage is aligned with promptable segmentation methods in the SAM family [5,6], but the report treats segmentation as an implementation component rather than a new segmentation contribution. The reconstructed mesh gives the workflow an object that can be inserted into a scene, transformed, grounded, lit, and rendered from controlled viewpoints; image-to-3D asset-generation systems such as Hunyuan3D 2.1 provide relevant background for this stage [7]. These stages are not independent scripts in the conceptual framework; they are artifact contracts that define what the controller can inspect and reuse.

Scene-aware rendering then places the reconstructed object into a fixed Blender environment [4]. This stage is where the original manual tuning pain point appears most strongly. The render must preserve the scene, use stable lighting, avoid floating or clipping, and keep the object at a plausible scale. VLM/CV review provides construction-time feedback, while controller actions adjust parameters, trigger rerendering, or mark samples for regeneration or filtering.

After a usable canonical source state is selected, the workflow exports target views and constructs train-ready source-target pairs. Each pair records the source image, target image, instruction, target rotation, object identifier, object name, and prompt version. The schema is intentionally close to the downstream image-editing interface: the source image and instruction form the input, and the target image provides the supervised edit target. This generic pair schema becomes concrete in the horizontal rotation case study, where the source is a canonical front view and each target is a controlled object-yaw view in the same scene. This design makes the dataset directly usable by the LoRA validation probe while keeping the construction trace inspectable.

## 7. Case Study: Scene-Aware Object Rotation

We return to the original motivating task to validate whether the workflow can produce useful train-ready data and diagnose its own bottlenecks. Scene-aware object rotation is an appropriate first case because it requires stable background context, object identity preservation, viewpoint control, lighting consistency, and grounding consistency. It also exposes an important ambiguity: object rotation is not the same as camera-orbit multiview rendering.

The task-specific rendering contract is:

```text
canonical yaw000 state -> rotate object yaw -> keep scene and camera fixed
```

This contract belongs to the case study rather than the general framework. It prevents two failure modes. First, selecting a separate best render for each target angle can introduce cross-angle drift in material, lighting, scale, or grounding. Second, moving the camera changes the task into camera-orbit multiview rendering. By fixing the scene and camera and rotating only the object yaw, the pair more directly encodes object-level rotation editing.

The baseline dataset uses the canonical front view as the source and non-zero horizontal viewpoints as targets. Each training object contributes seven target views, producing 245 training pairs from 35 training objects. Validation and test partitions are object-disjoint, with 49 validation pairs and 56 test pairs. In R1, evaluation feedback selects weak angles, and the added objects are train-only. The round contributes 60 weak-angle pairs at 90, 180, and 270 degrees, expanding the train set to 305 pairs without contaminating validation or test objects.

This case study therefore connects the two loops. The inner loop constructs and reviews renderable object data. The outer loop uses validation feedback to decide where new data should be added. The result is a concrete train-ready dataset round that can be evaluated without changing the held-out protocol.

## 8. Validation Protocol

Validation asks whether the constructed data changes downstream behavior in the intended direction and whether the workflow prevents premature acceptance when regressions appear. The protocol uses LoRA fine-tuning [9] on Qwen-Image-Edit-2511 as a downstream probe; the Qwen-Image technical report and Qwen-Image-Edit-2511 model card document the underlying image-editing backbone used in this probe [10,11]. The report does not claim a new LoRA method; LoRA is used as an instrument for measuring whether the data engine produces useful supervision.

The exp5 baseline is the comparison point for R1 because the feedback-loop state uses the object-info prompt checkpoint. The baseline and R1 datasets follow the pair schema and object-disjoint split described above. The object split is frozen across rounds so that changes in validation or benchmark behavior can be attributed to the training-data intervention rather than to leakage or split drift.

The tracked training probe uses rank 32, learning rate 1e-4, 30 training epochs, and the epoch 29 checkpoint for comparison. The prompt format follows the train-ready instruction field, using view-language instructions that ask the model to rotate an object from the front view to a target side or back view. These settings are reported for reproducibility, but the main claim remains about data construction and diagnosis.

**Table 3. Validation configurations**

| Configuration | Training objects | Train pairs | Val pairs | Test pairs | Data intervention |
|---|---:|---:|---:|---:|---|
| exp5 baseline | 35 original train objects | 245 | 49 | 56 | baseline object-info prompt checkpoint |
| v2 Scaling R1 | 35 original + 20 train-only objects | 305 | 49 | 56 | +60 weak-angle pairs at 90, 180, and 270 degrees |

The external evaluation uses SpatialEdit-Bench [12] with 488 rotation pairs across 61 objects and eight angle slots. Metrics cover pixel fidelity with PSNR, structural similarity with SSIM, perceptual distance with LPIPS, semantic image similarity with CLIP-I, representation similarity with DINO, distribution-level quality with FID, and VLM-based view and consistency judgments with Score_view, Score_cons, and VIE Overall. The use of VLM-based scoring is related to recent visual-instruction evaluation work such as VIEScore [8]. For angle-slot reporting, this report follows the `compare.py` mapping from indices 0..7 to 45, 90, 135, 180, 225, 270, 315, and 360 degrees.

## 9. Results and Diagnostic Analysis

The validation reveals a mixed but informative result. R1 improves several semantic and viewpoint-oriented metrics, which suggests that targeted weak-angle data changes model behavior in the intended direction. At the same time, consistency and distribution metrics regress, which prevents automatic acceptance. This is the expected behavior of the two-loop framework: the outer loop identifies useful signal, while the verdict exposes that the inner loop's render-quality gating is insufficient.

**Table 4. Overall SpatialEdit-Bench metrics for exp5 baseline and v2 Scaling R1**

| Metric | exp5 baseline | v2 R1 | Delta | Interpretation |
|---|---:|---:|---:|---|
| PSNR ↑ | 16.63 | 16.68 | +0.05 | slight fidelity improvement |
| SSIM ↑ | 0.7296 | 0.7310 | +0.0014 | slight structural improvement |
| LPIPS ↓ | 0.2564 | 0.2546 | -0.0018 | slight perceptual improvement |
| CLIP-I ↑ | 0.9050 | 0.9499 | +0.0449 | strong semantic improvement |
| DINO ↑ | 0.8895 | 0.8837 | -0.0058 | consistency regression |
| FID ↓ | 50.83 | 55.93 | +5.10 | distribution-quality regression |
| Score_view ↑ | 0.7705 | 0.7828 | +0.0123 | viewpoint improvement |
| Score_cons ↑ | 0.9709 | 0.9676 | -0.0033 | slight consistency regression |
| VIE Overall ↑ | 0.8649 | 0.8703 | +0.0054 | small overall VIE improvement |

The positive metrics indicate that weak-angle augmentation is not arbitrary noise. CLIP-I improves substantially, Score_view rises, and VIE Overall increases. PSNR, SSIM, and LPIPS also move in the desired direction overall, though the magnitudes are small and should not be overinterpreted. These results support the outer-loop decision to look for targeted weak-angle data.

The regressions explain why the dataset round should not be accepted as a clean success. DINO decreases, FID worsens, and Score_cons declines slightly. The comparison report also records a strong-angle DINO regression at 45 degrees, with a delta of -0.00994 against a threshold of -0.008. These regressions matter because object-level editing requires identity and appearance preservation, not only semantic or viewpoint alignment.

The render-quality diagnosis identifies the immediate bottleneck. All 20 newly added R1 objects have `hybrid_score < 0.6`, with a median around 0.46. This indicates that the outer loop found a useful scaling direction, but the inner loop allowed low-quality synthetic objects to enter the training intervention. The appropriate verdict is therefore `inspect`, not `continue`.

### Why `inspect` Is Not Failure

The `inspect` verdict is a contribution of the workflow rather than a sign that the workflow failed. In a traditional ad-hoc dataset pipeline, a mixed data round is often just wasted time: some metrics improve, others regress, and the builder has little guidance about what to fix next. In this workflow, the same mixed result becomes actionable evidence. The system indicates that weak-angle scaling is promising, but that render-quality gating, asset regeneration, and category-aware rendering priors must be improved before further scaling.

This interpretation preserves both sides of the evidence. R1 is not a full success because consistency and distribution metrics regress. It is also not a failure because it identifies a useful data direction and a concrete quality bottleneck. The value of the workflow is that it turns this ambiguity into a routed next action.

## 10. Discussion: Generalization and Boundaries

The framework-first view separates the data-construction method from the rotation case study. The broader idea is a reusable data engine for object-level editing tasks that require controlled changes and preserved invariants. The specific yaw contract does not transfer automatically to other tasks, but the workflow structure can: staged artifact construction, review, bounded feedback actions, downstream validation, and explicit verdicts.

This generalization must be stated cautiously. The workflow does not automatically solve resizing, repositioning, material editing, insertion, pose editing, or real-data collection. Each task must define its own change/invariance contract, artifact schema, quality signals, action space, and acceptance logic. The framework provides a disciplined way to express those requirements and inspect whether the resulting dataset satisfies them.

The next extension path is mixed or real-data construction. A future version could use VLM feedback to review collected real samples, identify missing coverage or quality issues, and convert downstream weaknesses into either synthetic augmentation plans or real-data collection plans. The same quality-gating logic could then apply across data sources. This report does not claim that the system already solves real-data construction; it identifies a plausible extension of the same closed-loop abstraction.

The practical lesson is that quality gates belong inside the data engine. If low-quality samples can enter training merely because they were generated, scaling becomes unsafe. The next version of the workflow should treat render quality, asset integrity, view-instruction consistency, and category-aware rendering priors as first-class acceptance conditions.

## 11. Limitations and Future Work

The current validation is an initial single-round `inspect` case. It shows that the framework can expose useful signal and harmful noise, but it does not prove that the closed-loop system is fully solved or robust across repeated scaling rounds. Additional rounds with stronger inner-loop quality gates are needed to test whether the identified bottleneck can be removed.

Synthetic asset quality remains the immediate boundary. The R1 objects fall below the desired hybrid-score threshold, making the round valuable as a diagnostic case but weak as a clean scaling result. The next concrete step is to add render-quality gating before export or train-set merging, regenerate or reject low-quality synthetic assets, and improve category-aware rendering priors so that new object classes do not inherit inappropriate lighting, material, or scale assumptions.

The framework also needs validation beyond scene-aware rotation. Future tasks should not assume that the rotation-specific rendering contract transfers directly. Each task should define its own transformation, invariants, review signals, and verdict logic. Mixed synthetic/real-data settings are especially important future work, because they would test whether the same workflow can guide both synthetic augmentation and real-data collection.

Several release artifacts remain incomplete for a public technical-report package. Qualitative grids, pipeline diagrams, failure-case figures, loss curves, and completed checkpoint hashes are not yet included. Angle-label consistency also requires care because historical notes use different labels for some angle slots; final per-angle tables should be regenerated from a single authoritative script.

## 12. Conclusion

This report presented a workflow-as-skill framework for autonomous synthetic dataset construction in object-level image editing. The framework grows out of a concrete manual-construction pain point: scene-aware synthetic data requires repeated observation, diagnosis, Blender parameter adjustment, regeneration, and validation. By making that loop explicit, the data engine turns manual tuning into a stateful workflow with staged artifacts, VLM/CV review, AI-agent actions, downstream probes, and explicit verdicts.

The scene-aware object rotation case study validates the framework on its original motivating task. The workflow produces train-ready source-target pairs, expands weak-angle data in R1, and evaluates the result through a fixed LoRA probe and SpatialEdit-Bench protocol. The evidence is mixed in a useful way: semantic and viewpoint metrics improve, while consistency and distribution metrics regress. The resulting `inspect` verdict identifies render-quality gating as the immediate bottleneck rather than treating the round as either a clean success or a dead end.

The broader conclusion is that dataset construction should be treated as an inspectable closed-loop system. The next step is to strengthen the inner loop with stricter render-quality gating and then test whether the same workflow can support further synthetic scaling and future mixed or real-data construction tasks.

## 13. Reproducibility and Implementation Details

The reproducible unit of the framework is a dataset-construction round. A round starts from a dataset goal and a set of object concepts, then produces prompts, white-background object images, foreground masks, reconstructed meshes, scene-aware rendered views, review records, train-ready pairs, a downstream checkpoint, and a comparison report. These artifacts form a linked state trace that lets later inspection connect a benchmark regression to the samples and construction stages that produced it.

The case-study dataset uses a source-target pair schema. A source image is the canonical front view, and each target image corresponds to a requested horizontal rotation. Each row records the source image, target image, instruction text, target rotation, object identifier, object name, and prompt version. The instruction format uses explicit view names, for example asking the model to rotate a named object from the front view to a right-side or back view.

Split construction is object-disjoint. The baseline configuration uses 35 train objects, 7 validation objects, and 8 test objects, yielding 245 train pairs, 49 validation pairs, and 56 test pairs. The R1 augmented configuration adds 20 new train-only objects and 60 weak-angle pairs while leaving validation and test objects fixed. This design prevents new synthetic objects from leaking into held-out splits and makes the R1 comparison a targeted train-data intervention.

The training probe uses Qwen-Image-Edit-2511 with LoRA fine-tuning under a fixed comparison protocol [9-11]. The tracked runs use rank 32, learning rate 1e-4, 30 epochs, and the epoch 29 checkpoint for evaluation. The probe is evaluated on SpatialEdit-Bench [12], which contains 488 rotation pairs across 61 objects and eight angle slots. The metric set combines traditional image metrics, representation metrics, distribution-level quality, and VLM-based view/consistency scores, matching the interpretation used in Section 9.

Checkpoint and artifact availability remains partial in this Markdown technical-report package. The tracked comparison uses exp5 baseline and v2 R1 LoRA checkpoints, but the large checkpoint files and full inference image dumps are not bundled because of size. A complete release should attach qualitative grids, pipeline diagrams, loss curves, failure-case figures, and checkpoint MD5 hashes.

### Artifact Index

Code, intermediate artifacts, and evaluation logs will be released at an anonymous repository upon publication. Key hyperparameters: LoRA rank=32, learning rate=1e-4, training epoch=29 (uniform across all rounds). Dataset split: N_train ≈ 305 pairs, N_val ≈ 49, N_test ≈ 56. Checkpoint hashes will be provided in the released repository.

## 14. References

[1] Tim Brooks, Aleksander Holynski, and Alexei A. Efros. 2023. *InstructPix2Pix: Learning to Follow Image Editing Instructions*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). arXiv:2211.09800.

[2] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. 2023. *MagicBrush: A Manually Annotated Dataset for Instruction-Guided Image Editing*. NeurIPS 2023 Datasets and Benchmarks. arXiv:2306.10012.

[3] Yuxuan Zhang, Huan Ling, Jun Gao, Kangxue Yin, Jean-Francois Lafleche, Adela Barriuso, Antonio Torralba, and Sanja Fidler. 2021. *DatasetGAN: Efficient Labeled Data Factory With Minimal Human Effort*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

[4] Blender Foundation. *Blender: Free and Open Source 3D Creation Software*. Official project website: https://www.blender.org/.

[5] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. 2023. *Segment Anything*. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).

[6] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár, and Christoph Feichtenhofer. 2024. *SAM 2: Segment Anything in Images and Videos*. arXiv:2408.00714.

[7] Team Hunyuan3D et al. 2025. *Hunyuan3D 2.1: From Images to High-Fidelity 3D Assets with Production-Ready PBR Material*. arXiv:2506.15442.

[8] Max Ku, Dongfu Jiang, Cong Wei, Xiang Yue, and Wenhu Chen. 2024. *VIEScore: Towards Explainable Metrics for Conditional Image Synthesis Evaluation*. Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL), Long Papers. DOI:10.18653/v1/2024.acl-long.663.

[9] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685.

[10] Chenfei Wu et al. 2025. *Qwen-Image Technical Report*. arXiv:2508.02324.

[11] Qwen Team. *Qwen-Image-Edit-2511*. Hugging Face model card: https://huggingface.co/Qwen/Qwen-Image-Edit-2511.

[12] Yicheng Xiao, Wenhu Zhang, Lin Song, Yukang Chen, Wenbo Li, Nan Jiang, Tianhe Ren, Haokun Lin, Wei Huang, Haoyang Huang, Xiu Li, Nan Duan, and Xiaojuan Qi. 2026. *SpatialEdit: Benchmarking Fine-Grained Image Spatial Editing*. arXiv:2604.04911.

[13] Shitao Xiao et al. 2024. *OmniGen: Unified Image Generation*. arXiv:2409.11340.

[14] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. 2023. *Zero-1-to-3: Zero-shot One Image to 3D Object*. arXiv:2303.11328.

[15] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinghao Li, Ming Ding, Jie Tang, and Yuxiao Dong. 2023. *ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation*. arXiv:2304.05977.

[16] Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah D. Goodman. 2022. *STaR: Bootstrapping Reasoning With Reasoning*. arXiv:2203.14465.

[17] Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Sainbayar Sukhbaatar, Jing Xu, and Jason Weston. 2024. *Self-Rewarding Language Models*. arXiv:2401.10020.
