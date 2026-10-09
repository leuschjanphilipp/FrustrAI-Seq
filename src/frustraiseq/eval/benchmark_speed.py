"""
Benchmark FrustrAI-Seq inference speed on a FASTA file (e.g. the human proteome).

Example usage:
    python src/frustraiseq/eval/benchmark_speed.py -i uniprotkb_human.fasta
    python src/frustraiseq/eval/benchmark_speed.py -i uniprotkb_human.fasta --checkpoint model.ckpt --plm-path /path/to/plm
"""

import copy
import time
import argparse

from frustraiseq.cli import load_fasta_to_dataframe, load_config_from_yaml, load_model
from frustraiseq.config.default_config import DEFAULT_CONFIG


def main():
    parser = argparse.ArgumentParser(description="Benchmark FrustrAI-Seq inference speed")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input FASTA file")
    parser.add_argument("-o", "--output", type=str, default=None, help="Optional output CSV for the predictions")
    parser.add_argument("--config", type=str, default=None, help="Config YAML (default: DEFAULT_CONFIG)")
    parser.add_argument("--model", type=str, default=None, help="HF repo id or exported model directory (default: leuschj/FrustrAI-Seq)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Lightning checkpoint, used instead of --model")
    parser.add_argument("--plm-path", type=str, default=None, help="Base pLM directory, only needed with --checkpoint")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default=None, help="torch device (default: cuda > mps > cpu)")
    args = parser.parse_args()

    config = load_config_from_yaml(args.config) if args.config else copy.deepcopy(DEFAULT_CONFIG)
    model = load_model(config, args.model, args.checkpoint, args.plm_path)

    df = load_fasta_to_dataframe(args.input)
    # sort by length so batches are similarly sized (non-standard residues are mapped to X by predict)
    df = df.iloc[df["sequence"].str.len().argsort()].reset_index(drop=True)

    start_time = time.time()
    predictions = model.predict(df, batch_size=args.batch_size, device=args.device)
    total_time = time.time() - start_time

    n_residues = df["sequence"].str.len().sum()
    print(f"Total time for predicting {len(df)} sequences ({n_residues} residues): {total_time:.2f} seconds "
          f"({len(df) / total_time:.2f} seqs/s, {n_residues / total_time:.0f} residues/s)")

    if args.output is not None:
        predictions.to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
