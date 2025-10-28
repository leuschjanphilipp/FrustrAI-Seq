import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

def get_callbacks(config: dict):
    early_stop = EarlyStopping(
        monitor=config["monitored_metric"],
        patience=config['early_stop_patience'],
        mode='min',
        verbose=True
    )
    checkpoint = ModelCheckpoint(
        monitor=config["monitored_metric"],
        dirpath=config['path_study'],
        filename=f"trial_{config["trial_number"]}_model",
        save_top_k=1,
        mode='min',
        save_weights_only=True
    )
    return [early_stop, checkpoint]

def bootstrapping_regression(true, preds, n_bootstrap=1000):
    spearman_list = []
    mae_list = []
    r2_list = []
    n = len(true)
    for _ in range(n_bootstrap):
        indices = np.random.choice(range(n), size=n, replace=True)
        true_sample = true[indices]
        preds_sample = preds[indices]
        mae_sample = mean_absolute_error(true_sample, preds_sample)
        spearman_sample, _ = spearmanr(true_sample, preds_sample)
        mae_list.append(mae_sample)
        spearman_list.append(spearman_sample)
        r2_list.append(r2_score(true_sample, preds_sample))
    return {
        "spearman_r": spearman_list,
        "mae": mae_list,
        "r2": r2_list
    }

def bootstrapping_classification(true, preds, n_bootstrap=1000):
    precision_list = []
    recall_list = []
    f1_list = []
    n = len(true)
    for _ in range(n_bootstrap):
        indices = np.random.choice(range(n), size=n, replace=True)
        true_sample = true[indices]
        preds_sample = preds[indices]
        report = classification_report(true_sample, preds_sample, labels=range(3), output_dict=True, zero_division=0)
        precision_list.append(report["weighted avg"]["precision"])
        recall_list.append(report["weighted avg"]["recall"])
        f1_list.append(report["weighted avg"]["f1-score"])
    return {
        "precision": precision_list,
        "recall": recall_list,
        "f1": f1_list
    }