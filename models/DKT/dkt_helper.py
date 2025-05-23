from sklearn.metrics import roc_auc_score           # <- NEW for DKT eval
import numpy as np                                  # <- NEW for masking
import torch
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# DKT helpers – kept outside the Model_Config class for clarity
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_STEP = 50          # block length for one-hot sequences
BATCH_SIZE = 64
EPOCHS = 10
HIDDEN = 100

def df_to_sequences(df):
    """Return a list of (questions, answers) per user, ordered by timestamp."""
    seqs = []
    df = df.sort_values("timestamp")
    skill_map = {s: i for i, s in enumerate(df.skill_name.unique())}
    for _, u_df in df.groupby("user_id"):
        q = [skill_map[s] for s in u_df.skill_name]
        a = u_df.correct.tolist()
        seqs.append((q, a))
    return seqs, len(skill_map)

def sequences_to_onehot(seqs, n_skill):
    """Pad to MAX_STEP and convert Q/A into one-hot blocks suitable for DKT."""
    blocks = []
    for q, a in seqs:
        L = len(q)
        pad = (MAX_STEP - L % MAX_STEP) % MAX_STEP
        X = np.zeros((L + pad, 2 * n_skill), dtype=np.float32)
        for i, q_id in enumerate(q):
            X[i, q_id if a[i] else q_id + n_skill] = 1
        blocks.append(X)
    X = np.concatenate(blocks).reshape(-1, MAX_STEP, 2 * n_skill)
    return torch.tensor(X)

def make_loaders(train_df, test_df):
    """Return PyTorch DataLoaders (yielding plain tensors) and n_skill."""
    train_seqs, n_skill = df_to_sequences(train_df)
    test_seqs, _ = df_to_sequences(test_df)

    train_x = sequences_to_onehot(train_seqs, n_skill)
    test_x  = sequences_to_onehot(test_seqs, n_skill)

    train_loader = DataLoader(train_x, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_x,  batch_size=BATCH_SIZE)

    return train_loader, test_loader, n_skill
