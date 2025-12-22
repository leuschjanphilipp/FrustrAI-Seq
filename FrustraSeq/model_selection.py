import os
import sys
import time
import torch
import signal
import optuna
import optuna_integration
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.strategies import DDPStrategy
from lightning.pytorch.utilities.rank_zero import rank_zero_only

sys.path.append('..')
sys.path.append('FrustraSeq')
from FrustraSeq.models.frustraSeq import FrustraSeq
from FrustraSeq.dataloader import FrustrationDataModule
from FrustraSeq.utils import set_signal, HaicoreCheckpointOnExit, run_eval_metrics


def objective(single_trial, config):

    trial = optuna_integration.TorchDistributedTrial(single_trial,)
    config = config.copy()
    config["experiment_name"] = config["experiment_name"] + f"/trial_{trial.number}"
    print(f"RANK {os.environ.get('RANK', -1)}: Starting trial {trial.number} with experiment name {config['experiment_name']}")

    os.makedirs(config["experiment_name"], exist_ok=True)
    print(f"RANK {os.environ.get('RANK', -1)}: Created directory {config['experiment_name']} for trial {trial.number}")

    architecture = FrustraSeq.suggest_params(trial)
    config["architecture"] = architecture
    trial.set_user_attr("config", config)

    torch.set_float32_matmul_precision("high")
    trainer_precision = "32"

    data_module = FrustrationDataModule(df=None,
                                        parquet_path=config["parquet_path"], 
                                        batch_size=config["batch_size"],
                                        set_key=config["set_key"],
                                        max_seq_length=config["max_seq_length"], 
                                        num_workers=config["num_workers"], # 0 
                                        persistent_workers=True, # Flase
                                        pin_memory=True, # Flase
                                        prefetch_factor=2, #!
                                        sample_size=None,
                                        cath_sampling_n=config["cath_sampling_n"])
    
    early_stop = EarlyStopping(monitor="val_loss",
                               patience=5,
                               mode='min',
                               verbose=True)
    checkpoint = ModelCheckpoint(monitor="val_loss",
                                 dirpath=f"./{config['experiment_name']}",
                                 filename=f"best_val_model",
                                 save_top_k=1,
                                 mode='min',
                                 save_weights_only=True)
    logger = WandbLogger(project="FrustraSeq_FrustraSeq_",
                         name=config["experiment_name"],
                         save_dir=f"./{config['experiment_name']}",
                         log_model=False,
                         offline=True)
    lr_logger = LearningRateMonitor(logging_interval='step')

    #plugins = [SLURMEnvironment(requeue_signal=signal.SIGUSR1, auto_requeue=False)]

    trainer = Trainer(default_root_dir=f"./{config['experiment_name']}",
                    accelerator="gpu",
                    devices=4,
                    strategy=DDPStrategy(find_unused_parameters=True), # true bc of finetune
                    max_epochs=10,
                    logger=logger,
                    log_every_n_steps=10,
                    val_check_interval=0.2,
                    callbacks=[early_stop, checkpoint, lr_logger], # HaicoreCheckpointOnExit()], #!
                    #plugins=plugins, #!
                    precision=trainer_precision,
                    gradient_clip_val=1,
                    enable_progress_bar=False,
                    deterministic=False,
                    accumulate_grad_batches=1, # chatgpt suggestion is to have this at 1 as LoRA likes smaller batch sizes
                    )
    #set_signal(trainer)

    if os.path.exists(f"./{config['experiment_name']}/hpc_ckpt.ckpt"):
        print(f"RANK {os.environ.get('RANK', -1)}: Resuming from HPC checkpoint.")
        model = FrustraSeq.load_from_checkpoint(checkpoint_path=f"./{config['experiment_name']}/hpc_ckpt.ckpt",
                                                config=config)
        trainer.fit(model, ckpt_path=f"./{config['experiment_name']}/hpc_ckpt.ckpt", datamodule=data_module)
    else:
        print(f"RANK {os.environ.get('RANK', -1)}: Initializing new model.")
        model = FrustraSeq(config=config)
        trainer.fit(model, datamodule=data_module)

    test_trainer = Trainer(accelerator="gpu", # gpu
                            devices=1, # only use one gpu for inference
                            max_epochs=2, 
                            logger=logger,
                            log_every_n_steps=5,
                            val_check_interval=50, 
                            precision=trainer_precision, #!, config["inference_precision"],
                            gradient_clip_val=1,
                            enable_progress_bar=False,
                            )
    #tune on val set
    data_module.test_dataloader = data_module.val_dataloader
    test_trainer.test(model, datamodule=data_module)
    model.save_preds_dict(set="val")
    metrics = run_eval_metrics(np.load(f"./{config['experiment_name']}/val_preds.npz"), return_cls_report_dict=True)
    return metrics["cls_report"]["macro avg"]["f1-score"]

@rank_zero_only
def run_model_selection(config, n_trials=10):
    os.makedirs(config["experiment_name"], exist_ok=True)
    storage = f"sqlite:///{config['experiment_name']}/optuna_study.db"
    sampler = optuna.samplers.TPESampler() 
    study = optuna.create_study(direction="maximize",
                                study_name="FrustraSeq_ms",
                                sampler=sampler,
                                storage=storage,
                                load_if_exists=True)
    study.optimize(lambda trial: objective(trial, config), 
                    n_trials=n_trials,
                    gc_after_trial=True)

    print("Number of finished trials: ", len(study.trials))
    print(f"Best trial: {study.best_trial.number}. Value: {study.best_trial.value}")

torch.distributed.init_process_group(backend='cpu:gloo,cuda:nccl')
RANK = int(os.environ.get("RANK", -1))
print(f"OS RANK {RANK}: Launched model selection script.")

n_trials = 1
config = {
    "experiment_name": "it5_protT5",
    "parquet_path": "FrustraSeq/data/frustration/v8_frustration_v2.parquet.gzip",
    "set_key": "split_test", # split_test (gonzalos prots in test) or set_old (split for previous dataset) or split0-3
    "cath_sampling_n": None, # 100,  # None for no sampling
    "batch_size": 16, #16 for all modules for FT; 64 maybe more for no FT
    "num_workers": 10,
    "max_seq_length": 512,
    "precision": "full",
    "pLM_model": "FrustraSeq/data/protT5",
    "prefix_prostT5": "<AA2fold>",
    "pLM_dim": 1024,
    "no_label_token": -100,
    "finetune": True,
    "lora_r": 4,
    "lora_alpha": 1,
    "lora_modules": ["q", "k", "v", "o", "wi", "wo", "w1", "w2", "w3", "fc1", "fc2", "fc3"], # ["query", "key", "value", "fc1", "fc2"] for esm
    "ce_weighting": [10.0, 2.0, 2.5],
    "notes": "",
}
if RANK == 0:
    print(f"RANK {RANK}: Creating study...")
    run_model_selection(config=config, n_trials=n_trials)
else:
    print(f"RANK {RANK}: Waiting for RANK 0 to create study...")
    for _ in range(n_trials):
        try:
            objective(None, config)
        except Exception as e:
            pass

