# FrustrAI-Seq

FrustrAI-Seq is a deep learning tool for predicting per-residue local energetic frustration from amino acid sequences using protein language models.

### Installation

```bash
mkdir FrustrAISeq
cd FrustrAISeq
conda create -n frustraiseq -y 
activate frustraiseq
git clone git@github.com:leuschjanphilipp/FrustrAI-Seq.git
```
Proceed by installing pytorch depending on your system. Look at their installation guides [here](https://pytorch.org/get-started/locally/).
```bash
pip install requirements.txt
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
frustraiseq predict -i data/example_seqs.fasta -o data/output.csv
```

Additionally config, pLM and checkpoint can be added as CLI arguments or in the config.

See `frustraiseq --help` for all options.

Or find a tutorial notebook in /notebooks.


## Output Format

The tool outputs a CSV file with predictions for each residue:

```csv
id,residue,frustration_index,frustration_class,entropy,surprisal
protein1,[S,E,Q,...],[0.13, -0.29, -0.14,...], [1,0,1,...], [0.79, 0.80, 0.83, ...], [1.11, 0.43, 1.09]
...
```

### Contributing

This is an APACHE2.0 LICENSE research repository. Contributions and suggestions are very welcome :)

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

For questions and issues:
- Open an issue on GitHub or contact the
- corresponding and jointly last authors: gonzalo.parra@bsc.es and ga32bav@mytum.de




