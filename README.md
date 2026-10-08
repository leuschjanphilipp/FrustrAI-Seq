# FrustrAI-Seq

FrustrAI-Seq is a deep learning tool for predicting per-residue local energetic frustration from amino acid sequences using protein language models.

### Installation

```bash
conda create -n frustraiseq python=3.12 -y
conda activate frustraiseq
git clone https://github.com/leuschjanphilipp/FrustrAI-Seq.git
cd FrustrAI-Seq
```
Proceed by installing pytorch depending on your system. Look at their installation guides [here](https://pytorch.org/get-started/locally/).
```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

### Basic Usage

```bash
# Create input FASTA file
cat > data/example_seqs.fasta << 'EOF'
>protein1
SEQVENCE
>protein2
SEVENCE
EOF

# Run prediction
frustraiseq predict -i data/example_seqs.fasta -o data/output.csv --config src/frustraiseq/config/default_config.yml --checkpoint path/to/model.ckpt --plm-path /path/to/user-plm --batch-size 16 --accelerator cuda

# Or use the short version in which pLM and model checkpoint will be downloaded from HuggingFace.
frustraiseq predict -i data/example_seqs.fasta -o data/output.csv
```

Sequences are uppercased and non-standard residues (e.g. U, Z, O, B) are mapped to `X`; a warning is printed for every affected sequence.

The pLM and checkpoint paths can also be set in the config instead of on the command line. Use `--no-verbose` to hide the progress bar and model logs.

See `frustraiseq predict --help` for all options, or find a tutorial notebook in `notebooks/`.


## Output Format

The tool outputs a CSV file with one row per residue:

```csv
id,residue,frustration_index,frustration_class,surprisal
protein1,S,0.13,1,1.11
protein1,E,-0.29,1,0.43
...
```

- `frustration_index`: predicted local frustration index (regression head)
- `frustration_class`: 0 = highly frustrated (index ≤ -1), 1 = neutral (-1 < index ≤ 0.55), 2 = minimally frustrated (index > 0.55). By default the class is obtained by binning `frustration_index`; set `use_cls_heads_output_for_class_pred: true` in the config to use the classification head instead.
- `entropy` (only with `use_cls_heads_output_for_class_pred: true`): normalized entropy of the classification head's class probabilities (0 = confident, 1 = uniform). It is omitted by default because it describes the classification head, not the binned class.
- `surprisal`: z-score of the predicted frustration index relative to the training-set distribution of that amino acid

## Training a new model

Training and evaluation scripts live in `src/frustraiseq/train/` and `src/frustraiseq/eval/`. Training uses multi-GPU DDP with bf16-mixed precision and logs to [Weights & Biases](https://wandb.ai) (run `wandb login` first).

```bash
python src/frustraiseq/train/train.py \
    --experiment_name my_run \
    --fit_dataset leuschj/Funstration \
    --plm_model ./prot_t5_xl_half_uniref50-enc \
    --split_key split_0 \
    --batch_size 16 \
    --num_workers 10 \
    --devices 2
```

- `--fit_dataset` accepts a HuggingFace dataset id or a local Parquet file.
- `--plm_model` is a local ProtT5 encoder directory (e.g. downloaded with `T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").save_pretrained(...)`) or a HuggingFace model id.
- `--split_key` selects the train/val/test split column (`split_0` ... `split_4`).
- `--cath_sampling_n N` subsamples N proteins per CATH topology for quick debugging runs.

All outputs are written to `./my_run/`: the best checkpoint (`best_val_model.ckpt`), the run config (`config.yaml`) and validation-set predictions (`val_preds.npz`). To evaluate on the test split:

```bash
python src/frustraiseq/eval/test.py --config my_run/config.yaml --checkpoint my_run/best_val_model.ckpt
```

The resulting checkpoint and config can be used directly with `frustraiseq predict --config my_run/config.yaml --checkpoint my_run/best_val_model.ckpt`.

### Contributing

This is an Apache 2.0 licensed research repository. Contributions and suggestions are very welcome :)

## Citation

If you use FrustrAI-Seq in your research, please cite:

```bibtex
@article{leusch_frustrai-seq_2026,
	title = {{FrustrAI}-Seq: Scaling Local Energetic Frustration to the Protein Sequence Space},
	doi = {10.64898/2026.02.03.703498},
    author = {Leusch, Jan-Philipp and Poley-Gil, Miriam and Fernandez-Martin, Miguel and Bordin, Nicola and Rost, Burkhard and Parra, R. Gonzalo and Heinzinger, Michael},
	date = {2026-02-05},
    publisher = {{bioRxiv}},
	abstract = {Proteins fold into their native three-dimensional (3D) structures by navigating complex energy landscapes shaped by the biophysical and biochemical properties of their sequence. Once folded, some sequence positions (dubbed residues) remain locally frustrated, reflecting functional constraints incompatible with optimal packing. This local energetic frustration provides important insights into protein function and dynamics, but its analysis typically relies on structure-based energy calculations and remains energetically costly at scale. Here, we introduce an ultra-fast sequence-based prediction of local energetic frustration directly from protein sequences using embeddings from protein language models ({pLMs}). Our method, coined {FrustrAI}-Seq, enables proteome-wide frustration profiling in minutes (∼ 17 minutes for the entire human proteome on a single Nvidia H100 {GPU}) while retaining biologically relevant performance as shown for the α-globin and β-lactamase family. By eliminating the need for explicit structural or evolutionary information, this approach expands frustration analysis to protein regions and classes that were previously inaccessible, including intrinsically disordered regions and high-throughput de novo designed protein datasets. To support reproducibility and large-scale applications, we provide the largest freely available resource of precomputed local frustration scores to date (∼106 proteins), along with model weights and complete training and inference code at: github.com/leuschjanphilipp/{FrustrAI}-Seq.}
    }
```

## Contact

For questions and issues, open an issue on GitHub or contact the corresponding and jointly last authors: gonzalo.parra@bsc.es and ga32bav@mytum.de
