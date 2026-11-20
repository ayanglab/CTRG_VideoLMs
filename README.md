# HRCT Report Generation with Video VLM

This is the official code repository for the paper **"Unleashing Video Language Models for Fine-Grained HRCT Report Generation"**.

We propose a two-stage training pipeline — **CoT (Chain-of-Thought) SFT followed by DPO (Direct Preference Optimization)** — to fine-tune video-based vision-language models for automated HRCT report generation. The model first learns to detect abnormalities and generate complete reports via chain-of-thought reasoning, then is further refined with DPO to improve report quality through preference alignment.

This project fine-tunes [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

> **Note:** This repository currently contains only example data samples for demonstration purposes. The full dataset and model weights will be released after the anonymous review period.

## Project Structure

```
hrct_report_generation_video_vlm/
├── data/                       # Datasets for LLaMA-Factory
│   ├── dataset_info.json       # Dataset registry for LLaMA-Factory
│   ├── sft_ab_train.json       # SFT abnormality training data
│   ├── sft_ab_valid.json       # SFT abnormality validation data
│   ├── sft_raw_train.json      # SFT raw report training data
│   ├── sft_raw_valid.json      # SFT raw report validation data
│   ├── base_cot_train.json     # Chain-of-thought training data
│   ├── base_cot_valid.json     # Chain-of-thought validation data
│   └── dpo_train.json          # DPO preference data
├── example_data/               # Pre-processed CT scan video samples (mp4)
├── yaml/                       # Training configuration files
│   ├── qwen_ab_train.yaml      # SFT / DPO training config
│   ├── qwen_dpo_train.yaml     # DPO training config
│   ├── qwen_merge.yaml         # LoRA merge config
│   └── qwen_predict.yaml       # Inference config
├── .gitignore
└── README.md
```

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/WangYijia1209/hrct-report-generation-video-vlm.git
cd hrct-report-generation-video-vlm
```

### 2. Install LLaMA-Factory

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
cd ..
```

For more details, see the [LLaMA-Factory installation guide](https://github.com/hiyouga/LLaMA-Factory#getting-started).

### 3. Download the base model

Download [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) into the `model/` directory:

```bash
mkdir -p model
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir model/Qwen2.5-VL-7B-Instruct
```

### 4. Prepare your data

Place your pre-processed CT scan videos (mp4) in the `example_data/` directory. Update the paths in the YAML configuration files under `yaml/` to match your local setup.

## Training

Our proposed pipeline consists of two stages: **CoT SFT → DPO**. We also provide a raw SFT baseline for comparison.

| Training Mode | Dataset Name | Description |
|---------------|-------------|-------------|
| CoT SFT (Ours - Stage 1) | `base_cot_train` | Chain-of-thought supervised fine-tuning: the model learns to first detect abnormalities, then generate complete reports |
| DPO (Ours - Stage 2) | `dpo_ab_train` | Direct preference optimization on the CoT-SFT checkpoint to further improve report quality |
| SFT Abnormality | `sft_ab_train` | Supervised fine-tuning on abnormality reports |
| SFT Raw (Baseline) | `sft_raw_train` | Supervised fine-tuning on raw reports, used as a comparison baseline |

Switch between training modes by changing the `dataset` field in the YAML config files.

### Stage 1: CoT SFT

```bash
# Train with chain-of-thought data
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train yaml/qwen_ab_train.yaml
# Set dataset: base_cot_train in the YAML file
```

### Stage 2: DPO

```bash
# Run DPO on the CoT-SFT checkpoint
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train yaml/qwen_dpo_train.yaml
```

### Baseline: Raw SFT

```bash
# For comparison, train with raw report data
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train yaml/qwen_ab_train.yaml
# Set dataset: sft_raw_train in the YAML file
```

You can customize the training configuration by modifying the YAML files according to your specific requirements.

## Model Test

To test the model, run the following commands to merge LoRA adapters and run inference:

```bash
# Merge LoRA adapters into the base model
CUDA_VISIBLE_DEVICES=0 llamafactory-cli export yaml/qwen_merge.yaml

# Run inference
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train yaml/qwen_predict.yaml
```


## Acknowledgments

- Our code leverages the [Qwen](https://github.com/QwenLM/Qwen) series model for its powerful vision-language capabilities
- We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for efficient fine-tuning of the Qwen series model
