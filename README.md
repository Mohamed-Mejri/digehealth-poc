# DigeHealth PoC

## Overview

DigeHealth PoC is a proof-of-concept project for digital health applications, focusing on leveraging machine learning and AI techniques. This repository contains scripts, configuration files, and source code to quickly prototype and test health-related ML solutions.

## Features

- Rapid prototyping of AI models for healthcare tasks.

- Configurable scripts for data processing and model experimentation.

- Modular source code structure for easy extension.

## Repo Structure
```bash
digehealth-poc/
├── conf/                       # Configuration files
├── scripts/                    # Utility and automation scripts
├── src/                        # Source code for models and data processing
├── .gitignore
├── pyproject.toml              # Python project configuration
├──.pre-commit-config.yaml      # Precommit hooks configuration for linting
├──uv.lock                      # Project Dependencies
└── README.md

```
## Installation
1. Clone the repository:
```bash
git clone https://github.com/Mohamed-Mejri/digehealth-poc.git
cd digehealth-poc
```

2. Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Create & activate virtual enviroment
```bash
uv sync
source venv/bin/activate
```

## Usage
To run the training for the classification head only you can do the following:
``` bash
python /.../digehealth-poc/scripts/train.py
```

To run LoRA finetuning on the model, run the following

``` bash
python /.../digehealth-poc/scripts/lora_finetune.py
```

All of these scripts are configurable via the `digehealth-poc/conf/config.yaml`

```yaml
wandb:
  project: digehealth
  run_name: lora_10_epochs_f1

data:
  annotation_paths:
    - /path/to/23M74M.txt # respect the same order here
    - /path/to/AS_1.txt
  audio_paths:
    - /path/to/23M74M.wav # respect the same order here
    - /path/to/AS_1.wav
train:
  hf_model_id: mharvill23/hubert-xlarge-bowel-sound-detector
  num_labels: 4
  batch_size: 16
  learning_rate: 1e-5
  num_epochs: 10
  num_workers: 4
  train_split_size: 0.8
  random_seed: 42
  device: cuda

checkpoint:
  load_checkpoint: True
  checkpoint_path: /path/to/outputs/checkpoints/your_model.pt

output_dir: /path/to/outputs


lora_config:
  r: 8
  lora_alpha: 32
  lora_dropout: 0.1
  target_modules:
    - q_proj
    - v_proj
    - k_proj
    - out_proj
    - intermediate_dense
    - output_dense
  bias: none

```
