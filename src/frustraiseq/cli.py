#!/usr/bin/env python3
"""
FrustrAI-Seq Command Line Interface

This module provides a CLI for running FrustrAI-Seq predictions on protein sequences.

Example usage:
    frustraiseq predict -i input.fasta -o output.csv
    frustraiseq predict -i input.fasta -o output.csv --model leuschj/FrustrAI-Seq --batch-size 32 --accelerator cuda
    frustraiseq predict -i input.fasta -o output.csv --config config.yml --checkpoint model.ckpt --plm-path /path/to/plm
"""

import argparse
import copy
import sys
import yaml
import pandas as pd
from typing import Optional, Dict, Any
from Bio import SeqIO

from frustraiseq.model.frustraiseq import FrustrAISeq, HF_MODEL_REPO
from frustraiseq.config.default_config import DEFAULT_CONFIG

# config entries that are passed on to from_pretrained models (everything else comes from the model's config.json)
RUNTIME_CONFIG_KEYS = ("batch_size", "verbose", "use_cls_heads_output_for_class_pred")

def load_fasta_to_dataframe(fasta_path: str) -> pd.DataFrame:
    """
    Load a FASTA file and convert it to a pandas DataFrame.
    
    Args:
        fasta_path: Path to the input FASTA file
        
    Returns:
        DataFrame with columns 'id' and 'sequence'
    """
    sequences = []
    ids = []
    
    try:
        with open(fasta_path, 'r') as handle:
            for record in SeqIO.parse(handle, "fasta"):
                ids.append(record.id)
                sequences.append(str(record.seq))
        
        if len(ids) == 0:
            raise ValueError(f"No sequences found in FASTA file: {fasta_path}")
            
        df = pd.DataFrame({
            'id': ids,
            'sequence': sequences
        })
        
        print(f"Loaded {len(df)} sequences from {fasta_path}")
        return df
        
    except FileNotFoundError:
        print(f"Error: FASTA file not found: {fasta_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading FASTA file: {e}")
        sys.exit(1)


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML config file
        
    Returns:
        Dictionary containing configuration
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)


def load_model(
    config: Dict[str, Any],
    model_name_or_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    pLM_path: Optional[str] = None,
) -> FrustrAISeq:
    """
    Load a FrustrAI-Seq model, either a `from_pretrained` model (HF repo id or exported directory)
    or a Lightning checkpoint from training (requires the base pLM).

    Precedence: model_name_or_path > checkpoint_path > config["checkpoint_path"] > HF default model.

    Args:
        config: configuration dictionary; for from_pretrained models only the runtime settings
            (RUNTIME_CONFIG_KEYS) are taken from it
        model_name_or_path: HF repo id or directory created with FrustrAISeq.save_pretrained
        checkpoint_path: Lightning checkpoint (.ckpt)
        pLM_path: base pLM directory (only for Lightning checkpoints, overrides config["pLM_model"])

    Returns:
        Loaded FrustrAISeq model in eval mode
    """
    if model_name_or_path is not None and checkpoint_path is not None:
        print("Error: use either --model or --checkpoint, not both.")
        sys.exit(1)
    if model_name_or_path is None:
        checkpoint_path = checkpoint_path or config.get("checkpoint_path")

    try:
        if checkpoint_path is not None:
            if pLM_path is not None:
                config["pLM_model"] = pLM_path
            if config.get("pLM_model") is None:
                print("Error: loading a Lightning checkpoint requires the base pLM (--plm-path or pLM_model in config).")
                sys.exit(1)
            print(f"Loading Lightning checkpoint {checkpoint_path} with pLM {config['pLM_model']}...")
            model = FrustrAISeq.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config, map_location="cpu")
        else:
            model_name_or_path = model_name_or_path or HF_MODEL_REPO
            print(f"Loading pretrained model {model_name_or_path}...")
            overrides = {k: config[k] for k in RUNTIME_CONFIG_KEYS if k in config}
            model = FrustrAISeq.from_pretrained(model_name_or_path, **overrides)
    except FileNotFoundError as e:
        print(f"Error: file not found: {e}")
        sys.exit(1)
    model.eval()
    return model


def run_prediction(
    input_fasta: str,
    output_csv: str,

    config_path: Optional[str] = None,
    model_name_or_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    pLM_path: Optional[str] = None,

    batch_size: Optional[int] = None,
    accelerator: Optional[str] = "auto",
    verbose: Optional[bool] = None
) -> None:
    """
    Run FrustrAI-Seq prediction on sequences from a FASTA file.

    Args:
        input_fasta: Path to input FASTA file
        output_csv: Path to output CSV file
        config_path: Path to config YAML (if None, uses DEFAULT_CONFIG)
        model_name_or_path: HF repo id or exported model directory (default: leuschj/FrustrAI-Seq)
        checkpoint_path: Lightning checkpoint from training (alternative to model_name_or_path)
        pLM_path: base pLM directory, only needed with checkpoint_path
        batch_size: Batch size for inference (if None, uses config)
        accelerator: Accelerator to use ('auto', 'cpu', 'cuda', 'mps')
        verbose: Whether to print verbose output (if None, uses config)
    """

    print("=" * 80)
    print("FrustrAI-Seq. Per Residue Local Energetic Frustration Prediction")
    print("=" * 80)

    print("\n[1/3] Loading input sequences...")
    df_input = load_fasta_to_dataframe(input_fasta)

    print("\n[2/3] Loading model...")
    if config_path is not None:
        config = load_config_from_yaml(config_path)
    else:
        config = copy.deepcopy(DEFAULT_CONFIG)
    if batch_size is not None:
        config["batch_size"] = batch_size
    if verbose is not None:
        config["verbose"] = verbose
    model = load_model(config, model_name_or_path, checkpoint_path, pLM_path)

    print("\n[3/3] Running predictions...")
    device = None if accelerator == "auto" else accelerator
    df_output = model.predict(df_input, batch_size=config["batch_size"], device=device)

    print(f"\nPredictions complete. Saving to {output_csv}...")
    df_output.to_csv(output_csv, index=False)
    
    print(f"Results saved to {output_csv}")
    print("\n" + "=" * 80)
    print("Prediction completed successfully!")
    print("=" * 80)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="FrustrAI-Seq - Predict per-residue local energetic frustration from protein sequences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic prediction (model is downloaded from HuggingFace: leuschj/FrustrAI-Seq)
  frustraiseq predict -i input.fasta -o output.csv --batch-size 16 --accelerator cuda

  # Lightning checkpoint from your own training run (needs the base pLM)
  frustraiseq predict -i input.fasta -o output.csv --config my_run/config.yaml \\
      --checkpoint my_run/best_val_model.ckpt --plm-path /path/to/plm
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Predict subcommand
    predict_parser = subparsers.add_parser(
        "predict",
        help="Run prediction on protein sequences"
    )
    
    # Required arguments
    predict_parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input FASTA file containing protein sequences"
    )
    
    predict_parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output CSV file for predictions"
    )
    
    # Optional arguments
    predict_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file."
    )

    predict_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"HuggingFace repo id or local directory of a pretrained model (default: {HF_MODEL_REPO})"
    )

    predict_parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Lightning checkpoint (.ckpt) from your own training run, used instead of --model. Requires --plm-path"
    )

    predict_parser.add_argument(
        "--plm-path",
        type=str,
        default=None,
        help="Base pLM directory, only needed with --checkpoint"
    )
    
    predict_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for inference (default: from config, 1)"
    )
    
    predict_parser.add_argument(
        "--accelerator",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Accelerator to use for prediction/inference (default: auto)"
    )
    
    predict_parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Print verbose output and progress bar (default: from config, True)"
    )

    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "predict":
        run_prediction(
            input_fasta=args.input,
            output_csv=args.output,
            config_path=args.config,
            model_name_or_path=args.model,
            checkpoint_path=args.checkpoint,
            pLM_path=args.plm_path,
            batch_size=args.batch_size,
            accelerator=args.accelerator,
            verbose=args.verbose
        )


if __name__ == "__main__":
    main()
