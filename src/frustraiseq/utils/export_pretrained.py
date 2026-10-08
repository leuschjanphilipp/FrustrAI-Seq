"""
Export a trained FrustrAI-Seq Lightning checkpoint to the `from_pretrained` format
(LoRA merged into the encoder, encoder in fp16, single safetensors file + config + tokenizer),
optionally pushing it to the HuggingFace Hub.

Example usage:
    python src/frustraiseq/utils/export_pretrained.py --checkpoint my_run/best_val_model.ckpt \
        --plm-path ./prot_t5_xl_half_uniref50-enc --out ./FrustrAI-Seq-export [--push-to-hub leuschj/FrustrAI-Seq]
"""

import os
import copy
import argparse
from huggingface_hub import HfApi

from frustraiseq.cli import load_config_from_yaml
from frustraiseq.config.default_config import DEFAULT_CONFIG
from frustraiseq.model.frustraiseq import FrustrAISeq

MODEL_CARD = """---
license: cc-by-4.0
library_name: frustraiseq
tags:
- biology
- protein
- protein-language-model
- local-energetic-frustration
datasets:
- leuschj/Funstration
---

# FrustrAI-Seq

FrustrAI-Seq predicts per-residue local energetic frustration directly from protein sequences.
It combines the ProtT5 protein language model (LoRA fine-tuned, merged into the encoder weights, stored in fp16)
with a small CNN and two output heads (frustration index regression and 3-class classification).

Code: https://github.com/leuschjanphilipp/FrustrAI-Seq

## Usage

```bash
pip install git+https://github.com/leuschjanphilipp/FrustrAI-Seq.git
frustraiseq predict -i input.fasta -o output.csv
```

```python
from frustraiseq import FrustrAISeq

model = FrustrAISeq.from_pretrained("leuschj/FrustrAI-Seq")
df = model.predict({"protein1": "MKTAYIAKQRQISFVKSHFSRQ"})
```

## Output

One row per residue with `id, residue, frustration_index, frustration_class, surprisal`.
`frustration_class` (0 = highly frustrated, 1 = neutral, 2 = minimally frustrated) is obtained by binning
`frustration_index` at -1 and 0.55. With `use_cls_heads_output_for_class_pred=True` the classification head is
used instead and an additional `entropy` column is reported.

## Citation

```bibtex
@article{leusch_frustrai-seq_2026,
    title = {{FrustrAI}-Seq: Scaling Local Energetic Frustration to the Protein Sequence Space},
    doi = {10.64898/2026.02.03.703498},
    author = {Leusch, Jan-Philipp and Poley-Gil, Miriam and Fernandez-Martin, Miguel and Bordin, Nicola and Rost, Burkhard and Parra, R. Gonzalo and Heinzinger, Michael},
    date = {2026-02-05},
    publisher = {{bioRxiv}},
}
```
"""


def main():
    parser = argparse.ArgumentParser(description="Export a FrustrAI-Seq checkpoint for FrustrAISeq.from_pretrained")
    parser.add_argument("--checkpoint", type=str, required=True, help="Lightning checkpoint (.ckpt)")
    parser.add_argument("--plm-path", type=str, required=True, help="Base ProtT5 encoder directory used for training")
    parser.add_argument("--config", type=str, default=None, help="Training config YAML (default: DEFAULT_CONFIG)")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument("--push-to-hub", type=str, default=None, help="Optional HF repo id to upload the export to")
    args = parser.parse_args()

    config = load_config_from_yaml(args.config) if args.config else copy.deepcopy(DEFAULT_CONFIG)
    config["pLM_model"] = args.plm_path
    config["precision"] = "full"  # merge LoRA in fp32, cast to fp16 afterwards
    config["verbose"] = False

    model = FrustrAISeq.load_from_checkpoint(checkpoint_path=args.checkpoint, config=config, map_location="cpu")
    model.encoder = model.encoder.merge_and_unload().half()
    model.config["precision"] = "half"
    model.config["verbose"] = True
    model.save_pretrained(args.out)
    with open(os.path.join(args.out, "README.md"), "w") as f:
        f.write(MODEL_CARD)
    print(f"Exported model to {args.out}")

    if args.push_to_hub is not None:
        HfApi().upload_folder(folder_path=args.out, repo_id=args.push_to_hub, repo_type="model",
                              commit_message="Upload FrustrAI-Seq in from_pretrained format")
        print(f"Uploaded to https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    main()
