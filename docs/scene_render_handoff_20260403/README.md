# Scene Render Handoff 2026-04-03

本目录是对 `2026-04-02` 交接文档的续写，重点不再是 `v2 -> v7` 的历史回顾，而是：

- 当前真正要继续推进的主线是什么
- 当前有哪些“用户已经认可、可作为兜底的视觉基线”
- 远端正在跑什么
- 下一位 AI 应该从哪里接手
- 哪些坑已经踩过，不要再踩

优先阅读：

1. `CURRENT_SUCCESS_PATH_HANDOFF_20260403.md`
2. `../scene_render_handoff_20260402/README.md`
3. `../scene_render_handoff_20260402/SCENE_RENDER_WORK_SUMMARY_20260402.md`

如果只是为了继续当前工作，不需要先通读所有旧文档。  
先看本目录主文档即可。

当前需要同时记住 3 条“用户已认可”的结果路径：

1. 当前继续推进的 rotation4 主线  
   `/aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_rotation4_agent_round0_obj001_obj003_obj007_obj008_20260402`
2. 当前最好、最稳的非旋转基础 scene render 兜底  
   `/aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_pseudoref_multigpu_20260401_teachercam3_light1`
3. 当前视觉上也可接受、但 controller 结论不可信的 full10 兜底  
   `/aaaidata/zhangqisong/data_build/pipeline/data/evolution_scene_v7_qwen35_freeform_full10_20260402`
