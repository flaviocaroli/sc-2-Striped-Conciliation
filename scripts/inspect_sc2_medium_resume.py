from pathlib import Path
import json
import torch

ckpt = Path('/home/3159436/sc2/outputs/sc2_medium_lung_large/checkpoints/last.pt')
metrics = Path('/home/3159436/sc2/outputs/sc2_medium_lung_large/metrics_partial.json')

if not ckpt.exists():
    raise SystemExit(f'No checkpoint found: {ckpt}')

obj = torch.load(ckpt, map_location='cpu', weights_only=False)
print('checkpoint:', ckpt)
print('completed_global_epoch:', obj.get('completed_global_epoch', obj.get('epoch')))
print('partial_epoch:', obj.get('partial_epoch'))
print('interrupted:', obj.get('interrupted'))
print('stage_name:', obj.get('stage_name'))
print('stage_idx:', obj.get('stage_idx'))
print('stage_epoch_idx:', obj.get('stage_epoch_idx'))
print('step:', obj.get('step'), '/', obj.get('n_steps'))
print('best_epoch:', obj.get('best_epoch'))
print('best_val_total:', obj.get('best_val_total'))

if metrics.exists():
    data = json.loads(metrics.read_text())
    print('metrics_partial_best_epoch:', data.get('best_epoch'))
    print('metrics_partial_best_val_total:', data.get('best_val_total'))
    print('history_rows:', len(data.get('history', [])))
