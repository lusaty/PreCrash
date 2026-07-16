# PreCrash

**PreCrash** (*Predicting the Reproducibility of Crash Reports*) is an automated tool for predicting whether a crash report is **reproducible** or **non-reproducible**.
PreCrash is a multi-feature fusion approach for predicting the reproducibility of a crash report. 
This approach helps identify reproducible crash reports before the manual effort on trying to reproduce the crashes. 

## Project Structure

```text
PreCrash/
├── datasets/      # Datasets and processed crash report.
├── figures/       # Results of experimental analysis.
├── saved_models/  # Trained models and serialized artifacts.
├── sourcecode/    # Sourcecode for preprocessing, training, and evaluation.
└── README.md        # Project description.
```


## Requirements

The exact package versions should be listed in a `requirements.txt` or environment configuration file. A typical environment may include:

- Python 3.8 or later;
- PyTorch;
- Transformers;
- Gensim;
- scikit-learn;
- pandas;
- NumPy;
- Matplotlib.

To create an isolated Python environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Activate the environment on Linux or macOS:

```bash
source .venv/bin/activate
```

Install the dependencies after adding a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Usage

Because script names may differ across releases, replace the example filenames below with the corresponding files in `sourcecode/`.

### 1. Data preprocessing

```bash
python sourcecode/preprocess.py
```

### 2. Feature extraction

```bash
python sourcecode/extract_features.py
```

### 3. Model training

```bash
python sourcecode/train.py
```

### 4. Model evaluation

```bash
python sourcecode/evaluate.py
```

### 5. Reproducibility prediction

```bash
python sourcecode/predict.py --input <path-to-crash-report>
```

Before running the project, check the configuration and path variables in the corresponding scripts.

## Reproducibility Labels

PreCrash formulates crash-report reproducibility prediction as a binary classification task. Each crash report is assigned one of the following labels:

- `reproducible`: the reported crash can be successfully reproduced;
- `non-reproducible`: the reported crash cannot be reproduced under the available information and experimental settings.

## Experimental Evaluation

PreCrash is evaluated against:

- traditional machine-learning models;
- deep-learning baselines;
- individual text-representation baselines;
- feature-ablation variants.

The main evaluation metrics include:

- Accuracy;
- Precision;
- Recall;
- F1-score;
- ROC-AUC.


