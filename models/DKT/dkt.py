# models/dkt.py
# ============================================================================
# GRU–based Deep-Knowledge Tracing
# – numerically stable BCEWithLogitsLoss
# – automatic masking (no padding rows in loss)
# – AUC + RMSE metrics
# – gradient clipping
# ============================================================================

import logging
import numpy as np
import torch, tqdm
from torch import nn
from sklearn.metrics import (
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,       
)
from EduKTM import KTM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 1.  Network
# ---------------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self, n_skill: int, hidden: int, n_layers: int, dropout: float):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=2 * n_skill,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden, n_skill)

    def forward(self, x):
        h0 = torch.zeros(
            self.rnn.num_layers, x.size(0), self.rnn.hidden_size, device=x.device
        )
        out, _ = self.rnn(x, h0)
        return self.fc(out)  # logits


# ---------------------------------------------------------------------------
# 2.  Helpers
# ---------------------------------------------------------------------------
def gather_logits_truth(onehot, logits, n_skill):
    """Return (logit_vec, truth_vec) for loss – *no detach*, keeps grad."""
    idx = (onehot.sum(-1) > 0).nonzero(as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        return None, None

    q_ids = torch.argmax(onehot[idx, :n_skill] + onehot[idx, n_skill:], dim=1)

    truth = (
        onehot[idx, :n_skill]
        .gather(1, q_ids.view(-1, 1))
        .float()
        .squeeze()
    )
    logit = logits[idx, :].gather(1, q_ids.view(-1, 1)).squeeze()
    return logit, truth


def gather_probs_truth(onehot, logits, n_skill):
    """Return (prob, truth) detached – for evaluation/metrics."""
    logit, truth = gather_logits_truth(onehot, logits, n_skill)
    if logit is None:
        return None, None
    prob = torch.sigmoid(logit)
    return prob.detach(), truth.detach()


# ---------------------------------------------------------------------------
# 3.  DKT wrapper
# ---------------------------------------------------------------------------
class DKT(KTM):
    def __init__(
        self,
        num_questions: int,
        hidden_size: int = 100,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_skill = num_questions
        self.dkt_model = Net(
            n_skill=num_questions,
            hidden=hidden_size,
            n_layers=num_layers,
            dropout=dropout,
        ).to(DEVICE)

    # ----------------------------------------------------------------------
    def train(
        self,
        train_loader,       # yields X only (padding rows allowed)
        test_data=None,     # optional DataLoader for validation
        *,
        epoch: int = 50,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
    ):
        loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
        opt = torch.optim.Adam(
            self.dkt_model.parameters(), lr=lr, weight_decay=weight_decay
        )

        for e in range(1, epoch + 1):
            self.dkt_model.train()
            epoch_losses = []

            for X in tqdm.tqdm(train_loader, desc=f"Epoch {e:02d}", leave=False):
                X = X.to(DEVICE)
                logits = self.dkt_model(X)

                sample_losses = []
                for b in range(X.size(0)):
                    lg, tr = gather_logits_truth(X[b], logits[b], self.n_skill)
                    if lg is not None:
                        sample_losses.append(loss_fn(lg, tr))

                if not sample_losses:    # skip empty batches
                    continue

                batch_loss = torch.stack(sample_losses).mean()
                opt.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.dkt_model.parameters(), 5)
                opt.step()

                epoch_losses.append(batch_loss.item())

            print(f"[Epoch {e:02d}] train-loss: {np.mean(epoch_losses):.4f}", end="")

            if test_data is not None:
                mae, rmse, auc = self.eval(test_data)
                print(f"  |  val-AUC: {auc:.4f}  RMSE: {rmse:.4f}  MAE: {mae:.4f}")
            else:
                print()

    # ----------------------------------------------------------------------
    def eval(self, data_loader):
        """Compute (AUC, RMSE) on a loader (expects X only)."""
        self.dkt_model.eval()
        y_pred, y_true = [], []

        # --- inside eval() ----------------------------------------------------
        with torch.no_grad():
            for X in data_loader:
                X = X.to(DEVICE)
                logits = self.dkt_model(X)
                for b in range(X.size(0)):
                    p, t = gather_probs_truth(X[b], logits[b], self.n_skill)
                    if p is not None:
                        # ensure 1-D before storing
                        y_pred.append(p.view(-1).cpu())    # ← changed
                        y_true.append(t.view(-1).cpu())    # ← changed


        y_pred = torch.cat(y_pred).numpy()
        y_true = torch.cat(y_true).numpy()

        auc = roc_auc_score(y_true, y_pred)
        try:
            rmse = mean_squared_error(y_true, y_pred, squared=False)
        except TypeError:                        
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        mae  = mean_absolute_error(y_true, y_pred) 
        return mae, rmse, auc

    # ----------------------------------------------------------------------
    def save(self, path):  torch.save(self.dkt_model.state_dict(), path)
    def load(self, path):  self.dkt_model.load_state_dict(torch.load(path))


# ============================================================================
# End of file
# ============================================================================