import torch
import random
from random import shuffle
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import roc_auc_score, accuracy_score, mean_absolute_error, mean_squared_error
from packaging import version
import sklearn
import numpy as np

SKL_NEW = version.parse(sklearn.__version__) >= version.parse("0.22.0")

def set_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def get_data(df, train_split=0.8, randomize=True):
    """Extract sequences from dataframe.

    Arguments:
        df (pandas Dataframe): output by prepare_data.py
        train_split (float): proportion of data to use for training
    """
    item_ids = [torch.tensor(u_df["problem_id"].values, dtype=torch.long)
                for _, u_df in df.groupby("user_id")]
    skill_ids = [torch.tensor(u_df["skill_name"].values, dtype=torch.long)
                 for _, u_df in df.groupby("user_id")]
    labels = [torch.tensor(u_df["correct"].values, dtype=torch.long)
              for _, u_df in df.groupby("user_id")]

    item_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), i))[:-1] for i in item_ids]
    skill_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), s))[:-1] for s in skill_ids]
    label_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), l))[:-1] for l in labels]

    data = list(zip(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels))
    if randomize:
        shuffle(data)

    # Train-test split across users
    train_size = int(train_split * len(data))
    train_data, val_data = data[:train_size], data[train_size:]
    return train_data, val_data

class Metrics:
    """Keep track of metrics over time in a dictionary.
    """
    def __init__(self):
        self.metrics = {}
        self.counts = {}

    def store(self, new_metrics):
        for key in new_metrics:
            if key in self.metrics:
                self.metrics[key] += new_metrics[key]
                self.counts[key] += 1
            else:
                self.metrics[key] = new_metrics[key]
                self.counts[key] = 1

    def average(self):
        average = {k: v / self.counts[k] for k, v in self.metrics.items()}
        self.metrics, self.counts = {}, {}
        return average


# def prepare_batches(data, batch_size, randomize=True):
#     """Prepare batches grouping padded sequences.

#     Arguments:
#         data (list of lists of torch Tensor): output by get_data
#         batch_size (int): number of sequences per batch

#     Output:
#         batches (list of lists of torch Tensor)
#     """
#     if randomize:
#         shuffle(data)
#     batches = []

#     for k in range(0, len(data), batch_size):
#         batch = data[k:k + batch_size]
#         seq_lists = list(zip(*batch))
#         inputs_and_ids = [pad_sequence(seqs, batch_first=True, padding_value=0)
#                           for seqs in seq_lists[:-1]]
#         labels = pad_sequence(seq_lists[-1], batch_first=True, padding_value=-1)  # Pad labels with -1
#         batches.append([*inputs_and_ids, labels])

#     return batches

def prepare_batches(data, batch_size, randomize=True, pad_val=-1):
    if randomize:
        shuffle(data)
    batches = []
    for k in range(0, len(data), batch_size):
        item_in, skill_in, label_in, item_id, skill_id, lbl = zip(*data[k:k+batch_size])
        pad = lambda seqs, v: pad_sequence(seqs, batch_first=True, padding_value=v)
        batches.append([
            pad(item_in,   0),
            pad(skill_in,  0),
            pad(label_in,  0),
            pad(item_id,   0),
            pad(skill_id,  0),
            pad(lbl,     pad_val)       # labels padded with –1
        ])
    return batches

def compute_loss(preds, labels, criterion):
    preds = preds[labels >= 0].flatten()
    labels = labels[labels >= 0].float()
    return criterion(preds, labels)

def compute_auc(preds, labels):
    preds = preds[labels >= 0].flatten()
    labels = labels[labels >= 0].float()
    if len(torch.unique(labels)) == 1:  # Only one class
        auc = accuracy_score(labels, preds.round())
    else:
        auc = roc_auc_score(labels, preds)
    return auc


def fmt(v):          # pretty print AUC when it exists
    return f"{v:.4f}" if v is not None else "—"

@torch.no_grad()
def evaluate_split(model, data, batch_size, pad_val=-1):
    model.eval()
    all_probs, all_labels = [], []

    for batch in prepare_batches(data, batch_size, randomize=False, pad_val=pad_val):
        item_in, skill_in, label_in, item_id, skill_id, lbl = [
            t.to(model.lin1.weight.device) for t in batch
        ]
        probs = torch.sigmoid(model(item_in, skill_in, label_in, item_id, skill_id))

        mask = lbl != pad_val
        all_probs.append(probs[mask].cpu())       # shape (num_true_steps,)
        all_labels.append(lbl  [mask].cpu().float())

    return get_metrics(torch.cat(all_probs), torch.cat(all_labels), pad_val=None)


@torch.no_grad()
def get_metrics(probs: torch.Tensor,
                labels: torch.Tensor,
                pad_val: int | None = -1):
    """
    Returns dict {'auc', 'mae', 'rmse'}.
    Works on *any* scikit-learn version.
    """
    # 1. mask pads --------------------------------------------------------
    if pad_val is not None:
        mask   = labels != pad_val
        probs  = probs [mask]
        labels = labels[mask]

    y_pred = probs.cpu().numpy()
    y_true = labels.float().cpu().numpy()

    # 2. MAE --------------------------------------------------------------
    mae = mean_absolute_error(y_true, y_pred)

    # 3. RMSE (version-agnostic) -----------------------------------------
    try:                                   # new API (>=0.22)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:                      # old API
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # 4. AUC --------------------------------------------------------------
    auc = None
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_pred)

    return {"auc": auc, "mae": mae, "rmse": rmse}