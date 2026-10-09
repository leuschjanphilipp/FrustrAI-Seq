import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import classification_report, mean_absolute_error, root_mean_squared_error, r2_score


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
        output_metrics["root_mean_squared_error"] = root_mean_squared_error(preds_dict["masked_regr_targets"], preds_dict["masked_regr_preds"])
        output_metrics["r2_score"] = r2_score(preds_dict["masked_regr_targets"], preds_dict["masked_regr_preds"])
    return output_metrics
