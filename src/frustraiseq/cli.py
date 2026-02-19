#!/usr/bin/env python3
"""
FrustrAISeq Command Line Interface

This module provides a CLI for running FrustrAISeq predictions on protein sequences.

Example usage:
    frustraiseq predict -i input.fasta -o output.csv
    frustraiseq predict -i input.fasta -o output.csv --config config.yml --checkpoint model.ckpt --plm-path /path/to/plm --batch-size 32 --accelerator cuda --verbose True
"""

import argparse
import os
import sys
from transformers import T5Tokenizer
import yaml
import pandas as pd
import torch
from pathlib import Path
from typing import Optional, Dict, Any
from Bio import SeqIO

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from transformers import T5EncoderModel, T5Tokenizer

from frustraiseq.data.dataloader import FunstrationDataModule
from frustraiseq.model.frustraiseq import FrustrAISeq
from frustraiseq.config.default_config import DEFAULT_CONFIG


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


def load_model_from_checkpoint(
    checkpoint_path: str,
    config: Optional[Dict[str, Any]] = None,
) -> FrustrAISeq:
    """
    Load a FrustrAISeq model from a checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint (.ckpt file)
        config: Optional configuration dictionary
        
    Returns:
        Loaded FrustrAISeq model
    """
    try:
        model = FrustrAISeq.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config)
        model.eval()
        return model
        
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model checkpoint: {e}")
        sys.exit(1)


def download_from_huggingface(repo_id: str = "leuschj/FrustrAI-Seq", 
                              filename: str = "FrustraSeq_CW.ckpt", 
                              local_dir: str = "./FrustrAI-Seq") -> str:
    """
    Download a file from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repository ID (e.g., 'leuschj/FrustrAI-Seq')
        filename: Name of the file to download
        local_dir: Local directory to save the file
        
    Returns:
        Path to the downloaded file
    """
    try:
        from huggingface_hub import hf_hub_download
        
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
        )
        print(f"Downloaded {filename} from {repo_id}")
        return local_path
        
    except ImportError:
        print("Error: huggingface_hub not installed. Install with: pip install huggingface_hub")
        sys.exit(1)
    except Exception as e:
        print(f"Error downloading from HuggingFace: {e}")
        sys.exit(1)


def run_prediction(
    input_fasta: str,
    output_csv: str,

    config_path: Optional[str] = None,
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
        checkpoint_path: Path to model checkpoint (if None, downloads from HF)
        config_path: Path to config YAML (if None, downloads from HF)
        batch_size: Batch size for inference
        accelerator: Accelerator to use ('cpu', 'cuda', 'mps')

        use_huggingface: Whether to download from HuggingFace
        hf_repo: HuggingFace repository ID
        hf_checkpoint: Checkpoint filename on HuggingFace
        hf_config: Config filename on HuggingFace

        verbose: Whether to print verbose output
    """
    
    print("=" * 80)
    print("FrustrAI-Seq. Per Residue Local Energetic Frustration Prediction")
    print("=" * 80)
    
    # Step 1: Load input sequences
    print("\n[1/5] Loading input sequences...")
    df_input = load_fasta_to_dataframe(input_fasta)
    
    # Step 2: Load or download checkpoint and config
    print("\n[2/5] Loading model checkpoint and configuration...")

    if config_path is not None:
        print(f"Config path provided: {config_path}. Loading config...")
        config = load_config_from_yaml(config_path)
    else:
        print("No config path provided. Falling back to default config...")
        #config_path = download_from_huggingface(hf_repo, hf_config)
        config = DEFAULT_CONFIG
    
    if pLM_path is not None:
        print(f"pLM path provided: {pLM_path}. (arg plm-path overrides path in config)")
        config["pLM_model"] = pLM_path
    elif config.get("pLM_model") is not None:
        print(f"pLM model specified in config: {config['pLM_model']}.")
    else:
        print("No pLM path provided. Downloading Rostlab/prot_t5_xl_half_uniref50-enc from HuggingFace...")
        encoder = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")

        encoder.save_pretrained("./prot_t5_xl_half_uniref50-enc")
        tokenizer.save_pretrained("./prot_t5_xl_half_uniref50-enc")
        del encoder
        del tokenizer
        config["pLM_model"] = "./prot_t5_xl_half_uniref50-enc"

    if checkpoint_path is not None:
        print(f"Checkpoint path provided: {checkpoint_path}. (arg checkpoint overrides path in config)")
        config["checkpoint_path"] = checkpoint_path
    elif config.get("checkpoint_path") is not None:
        print(f"Checkpoint path specified in config: {config['checkpoint_path']}.")
    else:
        print("No checkpoint path provided. Downloading from HuggingFace...")
        checkpoint_path = download_from_huggingface()
        config["checkpoint_path"] = checkpoint_path

    if batch_size is not None:
        config["batch_size"] = batch_size
    if verbose is not None:
        config["verbose"] = verbose

    config["inference_dataset"] = input_fasta #path of input fasta

    # Step 4: Setup data module
    print("\n[3/5] Preparing data...")
    data_module = FunstrationDataModule(
        config=config,
        inference_dataset=df_input,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        persistent_workers=False,  # Disable for CLI to avoid hanging on exit
        pin_memory=True if accelerator != "cpu" else False
    )

    # Step 5: Load model
    print("\n[4/5] Loading model...")
    model = load_model_from_checkpoint(config["checkpoint_path"], config)

    # Step 6: Run predictions
    print("\n[5/5] Running predictions...")

    # TODO enable multi-GPU. 
    trainer = Trainer(
        accelerator=accelerator,
        logger=False,
        enable_progress_bar=verbose,
    )
    
    predictions = trainer.predict(model, datamodule=data_module)
    
    # Process predictions and save to CSV
    print(f"\nPredictions complete. Saving to {output_csv}...")
    
    if predictions is None:
        print("Error: No predictions were generated")
        sys.exit(1)
    
    results = []
    for pred_batch in predictions:
        if pred_batch is not None:
            results.extend(pred_batch)
    
    df_output = pd.DataFrame(results)
    df_output.to_csv(output_csv, index=False)
    
    print(f"Results saved to {output_csv}")
    print("\n" + "=" * 80)
    print("Prediction completed successfully!")
    print("=" * 80)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="FrustrAISeq - Predict local energetic frustration for residues from sequences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Basic prediction
    frustraiseq predict -i input.fasta -o output.csv
    
    # Load custom checkpoints and config with different batch size and accelerator
    frustraiseq predict -i input.fasta -o output.csv --config config.yml --checkpoint model.ckpt --plm-path /path/to/plm \
    --batch-size 32 --accelerator cuda --verbose True

    input_fasta: str,
    output_csv: str,
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    pLM_path: Optional[str] = None,
    batch_size: Optional[int] = 1,
    accelerator: Optional[str] = "auto",
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
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint file (.ckpt). If not provided, download from HuggingFace: leuschj/FrustrAI-Seq HF repository"
    )

    predict_parser.add_argument(
        "--plm-path",
        type=str,
        default=None,
        help="Path to pretrained language model directory. If not provided, downloads Rostlab/prot_t5_xl_half_uniref50-enc from HuggingFace"
    )
    
    predict_parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)"
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
        action="store_true",
        default=True,
        help="Print verbose output (default: True)"
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
            checkpoint_path=args.checkpoint,
            pLM_path=args.plm_path,
            batch_size=args.batch_size,
            accelerator=args.accelerator,
            verbose=args.verbose
        )


if __name__ == "__main__":
    main()
