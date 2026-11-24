# LANCA: Latent Additive Noise Model Causal Autoencoder

Official implementation for the manuscript:
**"Towards Unsupervised Causal Representation Learning via Latent Additive Noise Model Causal Autoencoders"**.

---

## Repository Structure

```text
LANCA/
├── configs/
│   └── stage3_final/                  # Winning Stage-3 YAML configurations & random seeds
│       ├── pendulum_stage3.yaml
│       ├── flow_stage3.yaml
│       └── candle_stage3.yaml
├── scripts/
│   ├── train.py                       # Main training entry point
│   ├── evaluate.py                    # Checkpoint evaluation & metric computation
│   └── data_generation/               # Dataset generation & preprocessing scripts
│       ├── pendulum.py                # Pendulum (Gaussian ANM, eta=0.1)
│       ├── flow.py                    # Water Flow (Gaussian ANM, eta=0.1)
│       ├── pendulum_uniform.py        # Pendulum (Uniform additive noise)
│       ├── flow_uniform.py            # Water Flow (Uniform additive noise)
│       ├── generate_non_anm_datasets.py  # Heteroscedastic & multiplicative noise variants
│       └── preprocess_candle.py       # CANDLE dataset preprocessing
├── src/
│   ├── models/                        # LANCA and baselines (BetaVAE, FactorVAE, CausalVAE)
│   ├── training/                      # Trainer and loss functions (MMD, DAG, SCM, reconstruction)
│   ├── evaluation/                    # Metric computation (MIG, MMI, DCI, SAP, SHD, SID, IRS, UC, CG)
│   └── utils/                         # Dataset loaders and helpers
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── LICENSE
```

---

## Environment Setup

### Option 1: Docker (Recommended)
```bash
docker compose up -d --build
docker exec -e PYTHONPATH=/app lanca python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Option 2: Local Python Environment
Requires Python 3.10+ and a PyTorch CUDA installation:
```bash
pip install -r requirements.txt
```

---

## Data Generation & Preprocessing

### 1. Synthetic Physics Benchmarks (Pendulum & Water Flow)
```bash
# Standard Gaussian ANM benchmarks (eta=0.1)
PYTHONPATH=. python3 scripts/data_generation/pendulum.py
PYTHONPATH=. python3 scripts/data_generation/flow.py

# Uniform additive noise variants
PYTHONPATH=. python3 scripts/data_generation/pendulum_uniform.py
PYTHONPATH=. python3 scripts/data_generation/flow_uniform.py

# Non-ANM noise robustness variants (heteroscedastic & multiplicative)
PYTHONPATH=. python3 scripts/data_generation/generate_non_anm_datasets.py
```

### 2. CANDLE Benchmark
Download the original IITH CANDLE dataset into `./data/IITH_CANDLE` and run:
```bash
PYTHONPATH=. python3 scripts/data_generation/preprocess_candle.py
```

---

## Training

Train LANCA using the exact Stage-3 hyperparameter configurations and random seeds reported in the paper (`2025`–`2034` for Pendulum and Flow; `42`–`51` for CANDLE):

```bash
# Pendulum (seeds 2025-2034)
PYTHONPATH=. python3 scripts/train.py \
    --config configs/stage3_final/pendulum_stage3.yaml \
    --seed 2025

# Water Flow (seeds 2025-2034)
PYTHONPATH=. python3 scripts/train.py \
    --config configs/stage3_final/flow_stage3.yaml \
    --seed 2025

# CANDLE (seeds 42-51)
PYTHONPATH=. python3 scripts/train.py \
    --config configs/stage3_final/candle_stage3.yaml \
    --seed 42
```

Baseline models (`beta_vae`, `factor_vae`, `causal_vae`) are supported through the same entry point via `--model <model_name>`.

---

## Evaluation & Metrics

To evaluate a trained run directory (`best_model.pth` and `config.json`) and compute disentanglement (`MIG`, `MMI`, `DCI`, `SAP`), causal graph discovery (`SHD`, `SID`), and interventional robustness (`IRS`, `UC`, `CG`) metrics:

```bash
PYTHONPATH=. python3 scripts/evaluate.py \
    --run_dir ./experiments/<run_directory> \
    --model lanca \
    --dataset pendulum_anm_nr10 \
    --data_root ./data
```

Metric implementations are located in `src/evaluation/` (`evaluator.py`, `metrics.py`, `causal.py`, and `causal_metrics.py`).

---

## Note on Pretrained Checkpoints & Datasets

Due to GitHub repository size constraints, raw generated image datasets and multi-seed trained checkpoint binaries (~2.0 GB across 140 runs) are not committed directly to this repository. All datasets, models, and evaluation metrics can be reproduced directly using the provided configurations, random seeds, and scripts above. Pretrained weights are also available from the authors upon reasonable request.

---

## Citation

Citation details will be added upon publication.

## License

This project is released under the [MIT License](LICENSE).
