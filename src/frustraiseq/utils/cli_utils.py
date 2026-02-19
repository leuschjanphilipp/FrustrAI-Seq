"""
Utility functions for FrustrAISeq CLI operations
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any


def format_predictions_to_dataframe(
    predictions: List[Any],
    sequence_ids: List[str]
) -> pd.DataFrame:
    """
    Format raw model predictions into a structured DataFrame.
    
    Args:
        predictions: List of prediction batches from the model
        sequence_ids: List of sequence IDs corresponding to predictions
        
    Returns:
        DataFrame with formatted predictions
    """
    results = []
    
    for pred_batch in predictions:
        if pred_batch is None:
            continue
            
        # Extract predictions from batch
        # This structure depends on your model's predict_step output
        # Adapt this based on your actual output format
        
        for item in pred_batch:
            result_dict = {
                'id': item.get('id', 'unknown'),
                'sequence': item.get('sequence', ''),
                'frustration_index': item.get('frustration_index', []),
                'frustration_class': item.get('frustration_class', []),
                'classification_logits': item.get('classification_logits', []),
            }
            results.append(result_dict)
    
    return pd.DataFrame(results)


def expand_predictions_per_residue(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Expand predictions to have one row per residue.
    
    Args:
        df: DataFrame with per-sequence predictions
        
    Returns:
        DataFrame with one row per residue
    """
    expanded_rows = []
    
    for idx, row in df.iterrows():
        sequence = row['sequence']
        frst_idx = row.get('frustration_index', [])
        frst_class = row.get('frustration_class', [])
        
        for pos, (residue, f_idx, f_class) in enumerate(zip(sequence, frst_idx, frst_class)):
            expanded_rows.append({
                'protein_id': row['id'],
                'position': pos + 1,  # 1-indexed
                'residue': residue,
                'frustration_index': f_idx,
                'frustration_class': f_class,
                'frustration_label': get_frustration_label(f_class)
            })
    
    return pd.DataFrame(expanded_rows)


def get_frustration_label(class_idx: int) -> str:
    """
    Convert frustration class index to human-readable label.
    
    Args:
        class_idx: Integer class index (0, 1, or 2)
        
    Returns:
        String label for the frustration class
    """
    labels = {
        0: "highly_frustrated",
        1: "neutral",
        2: "minimally_frustrated"
    }
    return labels.get(class_idx, "unknown")


def validate_fasta(fasta_path: str) -> bool:
    """
    Validate that a FASTA file exists and has valid content.
    
    Args:
        fasta_path: Path to FASTA file
        
    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(fasta_path):
        print(f"Error: File does not exist: {fasta_path}")
        return False
    
    try:
        from Bio import SeqIO
        with open(fasta_path, 'r') as handle:
            sequences = list(SeqIO.parse(handle, "fasta"))
            if len(sequences) == 0:
                print(f"Error: No sequences found in {fasta_path}")
                return False
        return True
    except Exception as e:
        print(f"Error validating FASTA file: {e}")
        return False


def check_device_availability(device: str) -> str:
    """
    Check if the requested device is available and return the appropriate device string.
    
    Args:
        device: Requested device ('cpu', 'cuda', 'mps')
        
    Returns:
        Validated device string
    """
    import torch
    
    if device == "cuda":
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            return "cpu"
        return "cuda"
    elif device == "mps":
        if not torch.backends.mps.is_available():
            print("Warning: MPS not available, falling back to CPU")
            return "cpu"
        return "mps"
    else:
        return "cpu"


def estimate_memory_requirements(
    num_sequences: int,
    avg_length: int,
    batch_size: int,
    model_size_gb: float = 2.0
) -> Dict[str, float]:
    """
    Estimate memory requirements for inference.
    
    Args:
        num_sequences: Number of sequences to process
        avg_length: Average sequence length
        batch_size: Batch size
        model_size_gb: Model size in GB
        
    Returns:
        Dictionary with memory estimates
    """
    # Rough estimates
    embedding_size = 1024  # ProtT5 embedding dimension
    bytes_per_float = 4
    
    # Memory for one sequence
    seq_memory = avg_length * embedding_size * bytes_per_float / (1024**3)  # GB
    
    # Memory for one batch
    batch_memory = seq_memory * batch_size
    
    # Total memory estimate
    total_memory = model_size_gb + batch_memory * 1.5  # 1.5x for overhead
    
    return {
        'model_size_gb': model_size_gb,
        'batch_memory_gb': batch_memory,
        'total_estimated_gb': total_memory,
        'sequences_per_gb': 1.0 / seq_memory if seq_memory > 0 else 0
    }


def print_memory_info():
    """Print current memory usage information."""
    import torch
    
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    else:
        print("GPU not available")


def create_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create summary statistics for predictions.
    
    Args:
        df: DataFrame with predictions
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'total_proteins': df['protein_id'].nunique() if 'protein_id' in df.columns else 0,
        'total_residues': len(df),
        'avg_sequence_length': 0,
        'frustration_distribution': {}
    }
    
    if 'frustration_label' in df.columns:
        summary['frustration_distribution'] = df['frustration_label'].value_counts().to_dict()
    
    if 'protein_id' in df.columns:
        summary['avg_sequence_length'] = df.groupby('protein_id').size().mean()
    
    return summary
