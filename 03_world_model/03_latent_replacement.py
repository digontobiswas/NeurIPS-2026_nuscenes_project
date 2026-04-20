import os
import pickle
import numpy as np
import torch
import torch.nn as nn

OUTPUT_DIR = 'outputs/world_model_data'

os.makedirs(OUTPUT_DIR, exist_ok=True)


class SimpleEncoder(nn.Module):
    """
    Simple MLP encoder — encodes agent state into latent vector.
    In full implementation replace with SVD / diffusion encoder.
    """
    def __init__(self, input_dim=9, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class LatentReplacementModule(nn.Module):
    """
    Implements the latent replacement approach from Vista:
    replace noisy latent with clean condition frame latent.
    n_hat = m * z + (1 - m) * n
    where m is the condition mask,
          z is clean latent,
          n is noisy latent.
    """
    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim

    def forward(self, clean_latent, noisy_latent, mask):
        """
        clean_latent : (B, T, latent_dim)
        noisy_latent : (B, T, latent_dim)
        mask         : (B, T, 1) — 1 for condition frames
        """
        replaced = mask * clean_latent + (1 - mask) * noisy_latent
        return replaced


def create_condition_mask(seq_len, n_condition=3):
    """
    Create condition frame mask.
    First n_condition frames are condition (mask=1),
    rest are prediction frames (mask=0).
    """
    mask = torch.zeros(seq_len, 1)
    mask[:n_condition] = 1.0
    return mask


if __name__ == '__main__':
    priors_path = os.path.join(OUTPUT_DIR, 'dynamic_priors.pkl')

    if not os.path.exists(priors_path):
        print("Dynamic priors not found.")
        print("Run 02_dynamic_prior.py first.")
        exit()

    with open(priors_path, 'rb') as f:
        all_priors = pickle.load(f)

    encoder   = SimpleEncoder(input_dim=9, latent_dim=64)
    lrm       = LatentReplacementModule(latent_dim=64)

    B          = 4
    T          = 10
    latent_dim = 64

    print("Testing Latent Replacement Module...")
    print(f"  Batch size   : {B}")
    print(f"  Sequence len : {T}")
    print(f"  Latent dim   : {latent_dim}")
    print()

    clean_latent = torch.randn(B, T, latent_dim)
    noisy_latent = torch.randn(B, T, latent_dim)

    mask = create_condition_mask(T, n_condition=3)
    mask = mask.unsqueeze(0).expand(B, -1, -1)

    replaced = lrm(clean_latent, noisy_latent, mask)

    print(f"  Input shape  : {clean_latent.shape}")
    print(f"  Mask shape   : {mask.shape}")
    print(f"  Output shape : {replaced.shape}")
    print()

    cond_frames = (mask[0, :, 0] == 1).sum().item()
    pred_frames = (mask[0, :, 0] == 0).sum().item()
    print(f"  Condition frames  : {cond_frames}")
    print(f"  Prediction frames : {pred_frames}")

    sample_seq = torch.randn(B, T, latent_dim)
    torch.save(sample_seq, os.path.join(OUTPUT_DIR, 'sample_sequence_0.pt'))

    future = replaced
    torch.save(future, os.path.join(OUTPUT_DIR, 'future_prediction.pt'))

    print(f"\nSaved sample_sequence_0.pt and future_prediction.pt")
    print("Latent replacement module working correctly.")