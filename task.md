我们现在将要验证当前数据集的有效性，验证方法是通过训练qwen image edit 2511 模型，训练一个八方位角的旋转lora。其中在训练角度的时候，我们要将角度的方位映射为短语： 
                         0° 
                    (front view)
                         │
         315°            │            45°
    (front-left)         │       (front-right)
              ╲          │          ╱
               ╲         │         ╱
                ╲        │        ╱
   270° ─────────────── ● ─────────────── 90°
   (left side)        OBJECT         (right side)
                ╱        │        ╲
               ╱         │         ╲
              ╱          │          ╲
         225°            │            135°
     (back-left)         │       (back-right)
                         │
                        180°
                    (back view)
```markdown
| 角度 (Angle) | 描述短语 (Descriptor) |
| :--- | :--- |
| 0° | front view |
| 45° | front-right quarter view |
| 90° | right side view |
| 135° | back-right quarter view |
| 180° | back view |
| 225° | back-left quarter view |
| 270° | left side view |
| 315° | front-left quarter view |
```
可以参考代码DiffSynth-Studio\train_clockwise.py和DiffSynth-Studio\train_counter_clockwise.sh，调整为服务器上对应的数据集和模型的路径，保持训练时使用的prompt不变，推送到服务器的指定位置上（/aaaidata/zhangqisong/DiffSynth-Studio），然后进行训练，只需要修改这两个代码和脚本就可以直接开始训练了。如果训练出来的lora能够实现旋转，则说明数据集有效，否则需要对数据集进行调整
服务器的配置是三块A800,显存是80G，要求需要分片部署在三张卡上一起训练，这样才不会oom并且能加快训练速度