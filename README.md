# ImmunoSeq

Methods to predict antibody immunogenicity

> ImmunoSeq -- an interpretable and applicable method for immunogenicity prediction rooted in the biological principle of immune tolerance

## Installation Guide

1. Clone this repository

```bash
gh repo clone partrita/ImmunoSeq
cd ImmunoSeq
```

2. Install python packages required

```bash
uv sync
```

3. (Optional) Download OAS paired antibody sequences

```bash
cd data/OAS_PAIRED_HUMAN
bash bulk_download.
```

```bash
cd data/OAS_PAIRED_MOUSE
bash data/OAS_PAIRED_MOUSE/bulk_download.sh 
```

4. generate k-mer peptide library

```bash
uv run python src/prepare.py
```

This will generate `dump` files in `data/` folder. Be patient as it take a while.

## Benchmark

1. To run ADA correlation benchmark, use `uv run python  src/eval_ada_correlation.py`
2. To run humanness classification benchmark, use `uv run python  src/eval_humanness_classification.py`
3. To benchmark humanness classification on anbativ dataset, run `eval_abnativ.ipynb`
4. To analyze Hu-mAb 25 antibody pairs, use `uv run python  src/eval_humab25.py`
5. To perform sequence immunogenicity optimization, use `uv run python  src/infer.py`

## Usage

```bash
uv run python src/predict.py --input-file ./data/humab25_sequences.csv --output-file ./data/humab25_immunogenicity_predictions.csv
```

## Citation

```
@article{bytedance2025ImmunoSeq,
  title={Antibody immunogenicity prediction and optimization with ImmunoSeq},
  author={Huang, Qiaojing and He, Yi and Liu, Kai},
  year={2025},
  journal={bioRxiv},
  publisher={Cold Spring Harbor Laboratory},
  doi={10.1101/2025.08.14.670305},
  URL={https://www.biorxiv.org/content/10.1101/2025.08.14.670305v1},
  elocation-id={2025.08.14.670305},
  eprint={https://www.biorxiv.org/content/10.1101/2025.08.14.670305v1.full.pdf},
}
```

Please address all questions to `huangqiaojing@bytedance.com`