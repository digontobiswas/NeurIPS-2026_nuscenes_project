import numpy as np


def compute_fid_fvd(real_sequence, predicted_sequence):
    """
    Compute proxy FID and FVD scores.
    For full computation use pytorch-fid library.
    This is a simplified proxy for development.
    """
    if hasattr(real_sequence, 'numpy'):
        real = real_sequence.numpy()
    else:
        real = np.array(real_sequence)

    if hasattr(predicted_sequence, 'numpy'):
        pred = predicted_sequence.numpy()
    else:
        pred = np.array(predicted_sequence)

    # Proxy FID — mean squared difference of feature means
    real_flat = real.reshape(real.shape[0], -1).astype(float)
    pred_flat = pred.reshape(pred.shape[0], -1).astype(float)

    real_mean = np.mean(real_flat, axis=0)
    pred_mean = np.mean(pred_flat, axis=0)

    fid = float(np.mean((real_mean - pred_mean) ** 2))

    # Proxy FVD — temporal consistency score
    fvd = float(np.mean(np.abs(real_flat - pred_flat)))

    return {'fid': fid, 'fvd': fvd}


def compute_trajectory_difference(gt_positions, pred_positions):
    """
    Compute L2 trajectory difference (RMSE) between
    ground truth and predicted positions.
    gt_positions  : list of [x, y] pairs
    pred_positions: list of [x, y] pairs
    """
    gt   = np.array(gt_positions,   dtype=float)
    pred = np.array(pred_positions, dtype=float)

    min_len = min(len(gt), len(pred))
    gt      = gt[:min_len]
    pred    = pred[:min_len]

    diff = gt - pred
    l2   = np.sqrt(np.sum(diff ** 2, axis=1))
    rmse = float(np.sqrt(np.mean(l2 ** 2)))
    mae  = float(np.mean(l2))

    return {
        'rmse'       : rmse,
        'mae'        : mae,
        'per_step_l2': l2.tolist()
    }


def compute_reward(pred_positions, safe_distance=5.0):
    """
    Compute a simple reward based on predicted trajectory.
    Higher reward = safer and smoother trajectory.
    """
    positions = np.array(pred_positions, dtype=float)

    if len(positions) < 2:
        return {'total_reward': 0.0, 'smoothness': 0.0, 'safety': 0.0}

    # Smoothness reward — penalize sharp direction changes
    diffs      = np.diff(positions, axis=0)
    speeds     = np.sqrt(np.sum(diffs ** 2, axis=1))
    smoothness = float(1.0 / (1.0 + np.std(speeds)))

    # Safety reward — based on distance from origin
    distances  = np.sqrt(np.sum(positions ** 2, axis=1))
    safety     = float(np.mean(np.clip(distances / safe_distance, 0, 1)))

    total_reward = 0.5 * smoothness + 0.5 * safety

    return {
        'total_reward': total_reward,
        'smoothness'  : smoothness,
        'safety'      : safety
    }