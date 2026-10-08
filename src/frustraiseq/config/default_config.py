DEFAULT_CONFIG = {
    "experiment_name": "FrustrAI-Seq_Prediction",
    "fit_dataset": "leuschj/Funstration",
    "inference_dataset": None,
    "max_seq_length": 512,
    "batch_size": 1,
    "num_workers": 1,
    "pLM_model": None,  # base pLM, only needed for training or Lightning checkpoints
    "checkpoint_path": None,  # Lightning checkpoint; None -> FrustrAISeq.from_pretrained("leuschj/FrustrAI-Seq")
    "no_label_token": -100,
    "use_cls_heads_output_for_class_pred": False,  # False: frustration_class = binned frustration_index

    "precision": "half",
    "verbose": True,
    "notes": "",

    "lora_r": 4,
    "lora_alpha": 1,
    "lora_modules": ["q", "k", "v", "o"],
    "ce_weighting": [2.65750085, 0.68876299, 0.8533673],
    "architecture": {
        "lr": 1e-4,
        "dropout": 0.1,
        "pLM_dim": 1024,
        "kernel_1": 7,
        "padding_1": 7 // 2,  # kernel_1 // 2 to keep same length
        "kernel_2": 7,
        "padding_2": 7 // 2,  # kernel_2 // 2 to keep same length
        "hidden_dim_0": 64,
        "hidden_dim_1": 10}
}