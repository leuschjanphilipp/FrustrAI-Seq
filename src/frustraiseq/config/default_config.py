DEFAULT_CONFIG = {
    "experiment_name": "FrustrAI-Seq_Prediction",
    "fit_dataset": "leuschj/Funstration",
    "inference_dataset": None,
    "max_seq_length": 512,
    "batch_size": 1,
    "num_workers": 1,
    "pLM_model": None, #HF: Rostlab/prot_t5_xl_half_uniref50-enc
    "checkpoint_path": None, #HF: leuschj/FrustrAI-Seq/FrustraSeq_CW.ckpt
    "no_label_token": -100,

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