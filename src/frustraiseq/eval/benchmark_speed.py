"""
Benchmark FrustrAI-Seq inference speed on a FASTA file (e.g. the human proteome).

Example usage:
    python src/frustraiseq/eval/benchmark_speed.py -i uniprotkb_human.fasta --checkpoint model.ckpt --plm-path /path/to/plm
"""

import copy
import time
import argparse
import pandas as pd
from lightning.pytorch import Trainer

from frustraiseq.cli import load_fasta_to_dataframe, load_config_from_yaml
from frustraiseq.config.default_config import DEFAULT_CONFIG
from frustraiseq.data.dataloader import FunstrationDataModule
from frustraiseq.model.frustraiseq import FrustrAISeq


def main():
    parser = argparse.ArgumentParser(description="Benchmark FrustrAI-Seq inference speed")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input FASTA file")
    parser.add_argument("-o", "--output", type=str, default=None, help="Optional output CSV for the predictions")
    parser.add_argument("--config", type=str, default=None, help="Config YAML (default: DEFAULT_CONFIG)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint (default: checkpoint_path in config)")
    parser.add_argument("--plm-path", type=str, default=None, help="pLM directory (default: pLM_model in config)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    args = parser.parse_args()

    config = load_config_from_yaml(args.config) if args.config else copy.deepcopy(DEFAULT_CONFIG)
    if args.checkpoint is not None:
        config["checkpoint_path"] = args.checkpoint
    if args.plm_path is not None:
        config["pLM_model"] = args.plm_path
    assert config.get("checkpoint_path") and config.get("pLM_model"), "Provide --checkpoint and --plm-path (or set them in the config)."
    config["batch_size"] = args.batch_size
    config["num_workers"] = args.num_workers
    config["verbose"] = False

    df = load_fasta_to_dataframe(args.input)
    # sort by length so batches are similarly sized (non-standard residues are mapped to X by the DataModule)
    df = df.iloc[df["sequence"].str.len().argsort()].reset_index(drop=True)

    data_module = FunstrationDataModule(config=config,
                                        inference_dataset=df,
                                        batch_size=config["batch_size"],
                                        num_workers=config["num_workers"],
                                        persistent_workers=config["num_workers"] > 0,
                                        pin_memory=args.accelerator != "cpu")

    model = FrustrAISeq.load_from_checkpoint(checkpoint_path=config["checkpoint_path"], config=config)
    model.eval()

    trainer = Trainer(accelerator=args.accelerator,
                      devices=1,
                      precision=args.precision,
                      logger=False,
                      enable_progress_bar=True)

    start_time = time.time()
    predictions = trainer.predict(model, datamodule=data_module)
    total_time = time.time() - start_time

    n_residues = df["sequence"].str.len().sum()
    print(f"Total time for predicting {len(df)} sequences ({n_residues} residues): {total_time:.2f} seconds "
          f"({len(df) / total_time:.2f} seqs/s, {n_residues / total_time:.0f} residues/s)")

    if args.output is not None:
        pd.DataFrame([row for batch in predictions if batch is not None for row in batch]).to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
