import os
import torch
import numpy as np

SEQ_PATH    = 'outputs/world_model_data/sample_sequence_0.pt'
FUTURE_PATH = 'outputs/world_model_data/future_prediction.pt'
OUTPUT_DIR  = 'outputs/evaluation'

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(SEQ_PATH) or not os.path.exists(FUTURE_PATH):
    print("World model output files not found.")
    print("Run 03_world_model files first.")
    exit()

real_seq = torch.load(SEQ_PATH,    weights_only=False)
pred_seq = torch.load(FUTURE_PATH, weights_only=False)

print(f"Real sequence shape : {real_seq.shape}")
print(f"Pred sequence shape : {pred_seq.shape}")
print()


def compute_proxy_fid(real, pred):
    """Proxy FID using feature mean difference."""
    real_f  = real.reshape(real.shape[0], -1).float()
    pred_f  = pred.reshape(pred.shape[0], -1).float()
    r_mean  = real_f.mean(dim=0)
    p_mean  = pred_f.mean(dim=0)
    fid     = float(torch.mean((r_mean - p_mean) ** 2))
    return fid


def compute_proxy_fvd(real, pred):
    """Proxy FVD using temporal difference."""
    real_f = real.reshape(real.shape[0], -1).float()
    pred_f = pred.reshape(pred.shape[0], -1).float()
    min_n  = min(real_f.shape[0], pred_f.shape[0])
    fvd    = float(torch.mean(torch.abs(real_f[:min_n] - pred_f[:min_n])))
    return fvd


fid = compute_proxy_fid(real_seq, pred_seq)
fvd = compute_proxy_fvd(real_seq, pred_seq)

print('=' * 40)
print('EVALUATION RESULTS')
print('=' * 40)
print(f"Proxy FID : {fid:.6f}  (lower is better)")
print(f"Proxy FVD : {fvd:.6f}  (lower is better)")
print()
print("Note: These are proxy metrics for development.")
print("For paper-quality FID/FVD use pytorch-fid library.")

import pickle
results = {'fid': fid, 'fvd': fvd}
out_path = os.path.join(OUTPUT_DIR, 'fid_fvd_results.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(results, f)
print(f"\nResults saved: {out_path}")