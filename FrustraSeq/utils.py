import signal
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.callbacks import Callback


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

def run_eval_metrics(preds_file, regression=True, classification=True, bin_regression_for_classification=False, return_cls_report_dict=False, max_seq_length=512):

    if type(preds_file) is not dict:
        preds_dict = {key: preds_file[key] for key in preds_file.files}
    else:
        preds_dict = preds_file
    padded_seqs = []
    for seq in preds_dict["full_seqs"]:
        if len(seq) < max_seq_length:
            # Pad the sequence
            padding = "X" * (max_seq_length - len(seq))
            padded_seqs.append(seq + padding)
        else:
            # Truncate the sequence
            padded_seqs.append(seq[:max_seq_length])
    padded_seqs = np.array(padded_seqs)
    seq_array = np.array([list(seq) for seq in padded_seqs])
    preds_dict['masked_residues'] = seq_array[preds_dict["masks"]]
    
    output_metrics = {"preds_dict": preds_dict}
    
    if classification:
        if bin_regression_for_classification:
            preds_dict["masked_cls_preds"] = pd.cut(preds_dict["masked_regr_preds"].astype(float), bins=[-np.inf, -1, 0.55, np.inf], labels=[0,1,2])

        output_metrics["cls_report"] = classification_report(preds_dict["masked_cls_targets"], 
                                        preds_dict["masked_cls_preds"], 
                                        labels=range(3), 
                                        digits=4, 
                                        zero_division=0, 
                                        output_dict=return_cls_report_dict)
        output_metrics["confusion_matrix"] = pd.crosstab(preds_dict["masked_cls_targets"], preds_dict["masked_cls_preds"], rownames=['True'], colnames=['Predicted'])
    if regression:
        output_metrics["spearman_r"] = spearmanr(preds_dict["masked_regr_targets"], preds_dict["masked_regr_preds"])
        output_metrics["pearson_r"] = pearsonr(preds_dict["masked_regr_targets"], preds_dict["masked_regr_preds"])
        output_metrics["mean_absolute_error"] = mean_absolute_error(preds_dict["masked_regr_targets"], preds_dict["masked_regr_preds"])
        output_metrics["r2_score"] = r2_score(preds_dict["masked_regr_targets"], preds_dict["masked_regr_preds"])
    return output_metrics

################################################
### JUELICH SLURM REQUEUE HANDLING UTILITIES ###
################################################

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

def get_plugins(is_haicore):
    if not is_haicore:
        return []
    return [SLURMEnvironment(requeue_signal=signal.SIGUSR1, auto_requeue=False)]

class HaicoreCheckpointOnExit(Callback):
    def on_train_end(self, trainer, pl_module):
        if trainer.should_stop:
            if trainer.global_rank == 0:
                hpc_save_path = trainer.default_root_dir + "/hpc_ckpt.ckpt"
                print(f'Rank {trainer.global_rank}: Saving to {hpc_save_path}', flush=True)
                trainer.save_checkpoint(hpc_save_path)
                print("Checkpoint saved. Exiting gracefully.")
            else:
                print(f'Rank {trainer.global_rank}: Not saving checkpoint, only rank 0 saves.', flush=True)

def haicore_signal_connector(trainer):
    def _handle_signal(signum, frame):
        trainer.should_stop = True
    return _handle_signal

def set_signal(trainer):
    signal_connector = haicore_signal_connector(trainer)
    signal.signal(signal.SIGUSR1, signal_connector)

# Define a signal handler
def handle_sigusr1(signum, frame):
    print("Received SIGUSR1: saving checkpoint...")
    trainer.save_checkpoint("checkpoints/manual_checkpoint.ckpt")
    print("Checkpoint saved. Exiting gracefully.")
    exit(0)
