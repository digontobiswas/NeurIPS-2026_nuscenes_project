import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

OUTPUT_DIR = 'outputs/world_model_data'
FIGURES    = 'outputs/figures'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES,    exist_ok=True)


class SimpleFuturePredictor(nn.Module):
    """
    Simple LSTM-based future predictor.
    Input  : sequence of agent states (position + velocity)
    Output : predicted future positions
    Replace with Vista diffusion model for full implementation.
    """
    def __init__(self, input_dim=6, hidden_dim=128,
                 output_dim=3, n_future=5):
        super().__init__()
        self.n_future = n_future
        self.lstm     = nn.LSTM(input_dim, hidden_dim,
                                num_layers=2, batch_first=True)
        self.head     = nn.Linear(hidden_dim, output_dim * n_future)

    def forward(self, x):
        out, _   = self.lstm(x)
        last     = out[:, -1, :]
        pred     = self.head(last)
        pred     = pred.view(-1, self.n_future, 3)
        return pred


def load_trajectories(trajectory_dir):
    traj_files = [
        f for f in os.listdir(trajectory_dir)
        if f.endswith('.pkl')
    ]
    if len(traj_files) == 0:
        return None
    with open(os.path.join(trajectory_dir, traj_files[0]), 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    trajectories = load_trajectories('outputs/trajectories')

    if trajectories is None:
        print("No trajectories found.")
        exit()

    model     = SimpleFuturePredictor(
        input_dim=6, hidden_dim=128,
        output_dim=3, n_future=5
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    sequences = []
    targets   = []
    SEQ_LEN   = 8
    N_FUTURE  = 5

    for inst_token, traj in trajectories.items():
        if len(traj) < SEQ_LEN + N_FUTURE:
            continue

        for start in range(len(traj) - SEQ_LEN - N_FUTURE + 1):
            hist   = traj[start:start + SEQ_LEN]
            future = traj[start + SEQ_LEN:start + SEQ_LEN + N_FUTURE]

            pos = [[p['x'], p['y'], p['z']] for p in hist]
            vel = []
            for i in range(len(pos)):
                if i == 0:
                    vel.append([0, 0, 0])
                else:
                    v = [
                        pos[i][0] - pos[i-1][0],
                        pos[i][1] - pos[i-1][1],
                        pos[i][2] - pos[i-1][2]
                    ]
                    vel.append(v)

            feat = [pos[i] + vel[i] for i in range(SEQ_LEN)]
            tgt  = [[p['x'], p['y'], p['z']] for p in future]

            sequences.append(feat)
            targets.append(tgt)

    if len(sequences) == 0:
        print("Not enough trajectory data for training.")
        print("Creating dummy prediction output...")
        dummy_pred = torch.randn(10, N_FUTURE, 3)
        torch.save(
            dummy_pred,
            os.path.join(OUTPUT_DIR, 'future_prediction.pt')
        )
        print("Dummy prediction saved.")
        exit()

    X = torch.tensor(sequences, dtype=torch.float32)
    Y = torch.tensor(targets,   dtype=torch.float32)

    print(f"Training samples : {len(X)}")
    print(f"Input shape      : {X.shape}")
    print(f"Target shape     : {Y.shape}")
    print()

    EPOCHS     = 30
    BATCH_SIZE = 8
    losses     = []

    print("Training future predictor...")
    model.train()

    for epoch in range(EPOCHS):
        idx       = torch.randperm(len(X))
        X_shuf    = X[idx]
        Y_shuf    = Y[idx]
        epoch_loss = 0.0
        n_batches  = 0

        for i in range(0, len(X_shuf), BATCH_SIZE):
            xb = X_shuf[i:i + BATCH_SIZE]
            yb = Y_shuf[i:i + BATCH_SIZE]

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:>3}/{EPOCHS}  Loss: {avg_loss:.6f}")

    model.eval()
    with torch.no_grad():
        predictions = model(X[:10])

    torch.save(
        predictions,
        os.path.join(OUTPUT_DIR, 'future_prediction.pt')
    )
    torch.save(
        Y[:10],
        os.path.join(OUTPUT_DIR, 'sample_sequence_0.pt')
    )
    print(f"\nPredictions saved to {OUTPUT_DIR}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(losses, color='steelblue', linewidth=2)
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].grid(True, alpha=0.3)

    pred_np = predictions[0].numpy()
    gt_np   = Y[0].numpy()
    axes[1].plot(pred_np[:, 0], pred_np[:, 1],
                 'r-o', label='Predicted', linewidth=2)
    axes[1].plot(gt_np[:, 0],   gt_np[:, 1],
                 'g-o', label='Ground truth', linewidth=2)
    axes[1].set_title('Predicted vs Ground Truth Trajectory')
    axes[1].set_xlabel('X (m)')
    axes[1].set_ylabel('Y (m)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(FIGURES, 'future_prediction.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {out_path}")
    plt.show()
    plt.close()