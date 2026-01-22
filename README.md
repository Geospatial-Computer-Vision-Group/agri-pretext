# Time2Agri: Temporal Pretext Tasks for Agricultural Monitoring

[![arXiv](https://img.shields.io/badge/arXiv-2507.04366-b31b1b.svg)](https://arxiv.org/abs/2507.04366)
[![AAAI 2026](https://img.shields.io/badge/AAAI%202026-Social%20Impact%20Track-blue.svg)](https://aaai.org/conference/aaai/aaai-26/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official PyTorch implementation of **Time2Agri**, a self-supervised learning framework for agricultural representation learning using satellite imagery. This work has been accepted at **AAAI 2026 Social Impact Track**.

## Overview

Time2Agri introduces agriculture-focused temporal pretext tasks that capture seasonal cycles and temporal patterns unique to agricultural landscapes, addressing the limitations of existing remote sensing foundation models that neglect agricultural temporal dynamics.

### Key Contributions

We propose three novel temporal pretext tasks specifically designed for agricultural monitoring:

1. **Time-Difference Prediction (TD)** - Captures temporal changes between observations to model agricultural dynamics
2. **Temporal Frequency Prediction (FP)** - Analyzes cyclical patterns in agricultural data using frequency-domain representations
3. **Future-Frame Prediction (FF)** - Forecasts upcoming satellite imagery to learn causal temporal dependencies

### Performance Highlights

- **Crop Mapping:** 69.6% IoU on crop mapping benchmarks
- **Yield Prediction:** 30.7% MAPE, outperforming baseline approaches
- **Field Delineation:** 54.2% IoU on FTW India dataset for field boundary delineation

## Installation

### Requirements

- Python 3.11+
- PyTorch 2.5+
- PyTorch Lightning
- timm (PyTorch Image Models)
- einops
- zarr (for data loading)
- tensorboard
- tqdm
- matplotlib

### Setup

```bash
git clone https://github.com/Geospatial-Computer-Vision-Group/agri-pretext.git
cd agri-pretext

# Install dependencies
pip install torch torchvision lightning timm einops zarr tensorboard tqdm matplotlib
```

## Repository Status

### Currently Available
The code for training regional models is contained in the [regional_ssl](regional_ssl/) folder. Navigate to this folder for reproducing the regional pretraining experiments.

```bash
cd regional_ssl
```

### Coming Soon
We are actively working on releasing the following components:
- **National-scale pretraining code** - Training pipeline for larger geographic coverage
- **Datasets** - Preprocessed satellite imagery datasets used in our experiments
- **Evaluation code** - Downstream task evaluation scripts for crop mapping, yield prediction, and field delineation

Stay tuned for updates!

## Data

The code expects satellite imagery data in Zarr format. With each zarr group representing a chip, and containing two arrays: data, containing the TxCxHxW tensor, and timestamps, containing the time stamp corresponding to a given instance.
The dataset will be released soon.
Update the `data_dir` path in the configuration files to point to your dataset:

```yaml
data:
  data_dir: /path/to/your/dataset.zarr
  batch_size: 512
  num_workers: 40
  split_ratio: 0.8
```

### Computing Dataset Statistics

Before training, compute normalization statistics for your dataset.
Note: you need to update the path to the zarr in this `calc_stats.py` file.

```bash
python regional_ssl/calc_stats.py
```

This will generate a `stats.pth` file containing mean and standard deviation values used for normalization.

## Training

### Pre-training with Temporal Pretext Tasks

We provide configuration files for all three pretext tasks in the [`regional_ssl/configs/`](regional_ssl/configs/) directory.

#### Future-Frame Prediction (FF)

```bash
python regional_ssl/train_ff.py fit --config regional_ssl/configs/vits_ff.yaml
```

#### Temporal Frequency Prediction (FP)

```bash
python regional_ssl/train_fp.py fit --config regional_ssl/configs/vits_fp.yaml
```

#### Time-Difference Prediction (TD)

```bash
python regional_ssl/train_td.py fit --config regional_ssl/configs/vits_td.yaml
```

#### Masked Autoencoder (MAE) Baseline

```bash
python regional_ssl/train_mae.py fit --config regional_ssl/configs/vits_mae.yaml
```

### Configuration

All configuration files use PyTorch Lightning CLI format. Key parameters you can adjust:

- `trainer.max_epochs`: Number of training epochs (default: 100)
- `trainer.devices`: Number of GPUs to use
- `model.learning_rate`: Learning rate for optimization
- `model.warmup_epochs`: Number of warmup epochs for learning rate schedule
- `data.batch_size`: Training batch size
- `data.num_workers`: Number of data loading workers

Example configuration structure:

```yaml
seed_everything: true
trainer:
  max_epochs: 100
  accelerator: gpu
  devices: 1
  default_root_dir: logs/ff_vits
model:
  model_name: "vit_small_patch16_224"
  learning_rate: 0.0009
  img_size: 224
  patch_size: 16
data:
  data_dir: /path/to/data.zarr
  batch_size: 512
  num_workers: 40
```

## Model Architecture

Time2Agri uses a Vision Transformer (ViT) backbone with task-specific components:

- **Encoder:** ViT-Small (patch size 16, 384 dimensions)
- **Time Translator:** A two-layer Transformer for predicting the future latent
- **FreqDecoder/Decoder:** Task-specific reconstruction/prediction heads

### Available Pretraining Variants

- `vits_ff.yaml` - Future-Frame prediction with ViT-Small
- `vits_fp.yaml` - Frequency prediction with ViT-Small
- `vits_td.yaml` - Time-difference prediction with ViT-Small
- `vits_mae.yaml` - Standard MAE baseline
- `vits_mae_300.yaml` - Standard MAE baseline, trained for 300 epochs

## Checkpoints and Logs

Training logs and checkpoints are saved in the directory specified by `trainer.default_root_dir`:

```
logs/
├── ff_vits/          # Future-Frame logs
├── fp_vits/          # Frequency Prediction logs
├── td_vits/          # Time-Difference logs
└── mae_vits/         # MAE baseline logs
```

Each run saves:
- `best.ckpt` - Best model based on validation loss
- `last.ckpt` - Last checkpoint
- TensorBoard logs for visualization

We use `last.ckpt` during our evaluation.

### Monitoring Training

```bash
tensorboard --logdir logs/
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{gupta2025time2agritemporalpretexttasks,
      title={Time2Agri: Temporal Pretext Tasks for Agricultural Monitoring}, 
      author={Moti Rattan Gupta and Anupam Sobti},
      year={2025},
      eprint={2507.04366},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2507.04366}, 
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This research addresses critical challenges in agricultural monitoring using self-supervised learning on satellite imagery, with applications in crop mapping, yield prediction, and field delineation.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.

---

**AAAI 2026 Social Impact Track** | [Paper](https://arxiv.org/abs/2507.04366)
