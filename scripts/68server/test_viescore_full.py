import sys, os, time, math
os.chdir('/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build/SpatialEdit-Bench-Eval/object_level_eval')
sys.path.insert(0, '.')
from viescore import VIEScore, vie_prompts
from PIL import Image
import json

print('Loading VIEScore with qwen35vl backbone...')
t0 = time.time()
vie_score = VIEScore(backbone='qwen35vl', task='tie', key_path='secret.env')
print(f'Loaded in {time.time()-t0:.0f}s')

meta_path = '/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build/SpatialEdit-Bench/SpatialEdit_Bench_Meta_File.json'
with open(meta_path) as f:
    meta = json.load(f)
item = [x for x in meta if x['type'] == 'rotate'][0]
print(f'Test item: {item["image_id"]} edit_id={item["edit_id"]}')

src_path = os.path.join('/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build/SpatialEdit-Bench/images', item['image_path'])
pred_path = f'/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build/SpatialEdit-Bench-Eval/base/fullset/rotate/en/{item["image_id"]}/{item["edit_id"]}.png'

pil_src = Image.open(src_path).convert('RGB')
pil_pred = Image.open(pred_path).convert('RGB')

num_id = None
for k, v in vie_prompts.VIEW_PROMPT.items():
    if v in item['prompt']:
        num_id = k
        break
print(f'num_id={num_id}, object={item["object"]}')

question_prompt = vie_prompts.SC_rotate[num_id].format(object_name=item['object'])
Score_view_prompt = '\n'.join([vie_prompts._prompts_0shot_tie_rule_SC_rotate, question_prompt])

print('Testing Score_view...')
t1 = time.time()
Score_view = vie_score.evaluate([pil_pred], Score_view_prompt) / 10
print(f'Score_view = {Score_view} ({time.time()-t1:.1f}s)')

context = vie_prompts._context_no_delimit
Score_cons_prompt = '\n'.join([context, vie_prompts._prompts_0shot_in_context_generation_rule_SC_Scene])
Score_cons_prompt = Score_cons_prompt.replace('<instruction>', item['prompt'])

print('Testing Score_cons...')
t2 = time.time()
Score_cons = vie_score.evaluate([pil_src, pil_pred], Score_cons_prompt) / 10
print(f'Score_cons = {Score_cons} ({time.time()-t2:.1f}s)')

overall = math.sqrt(Score_view * Score_cons)
print(f'Overall = {overall:.4f}')
print('SUCCESS')
