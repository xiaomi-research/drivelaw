# DriveLaW: Unifying Planning and Video Generation in a Latent Driving World

DriveLaW is a comprehensive framework for autonomous driving that combines video generation and trajectory planning using latent diffusion models. The project consists of two main components:

- **DriveLaW-Video**: Video generation module based on Video diffusion models
- **DriveLaW-Act**: Action planning module that integrates video generation with trajectory planning

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
  - [Video Generation (Inference)](#video-generation-inference)
  - [Video Generation (Training)](#video-generation-training)
  - [Planning Module (Training)](#planning-module-training)
  - [Planning Module (Evaluation)](#planning-module-evaluation)
- [Configuration](#configuration)
- [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

## Overview

DriveLaW integrates:
1. **Video Generation**: Uses LTX-Video based diffusion models to generate future driving scenarios
2. **Trajectory Planning**: Combines video generation with action prediction for autonomous driving planning

## Installation

### Prerequisites

- Python >= 3.10
- CUDA-compatible GPU (recommended)
- PyTorch >= 2.1.0
- Linux (recommended)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd DriveLaW
```

### Step 2: Install Dependencies

Install the package with all dependencies:

```bash
pip install -e .
```


### Step 3: Install Additional Dependencies (if needed)

Some dependencies may require additional setup:

- **NUPlan Devkit**: Installed automatically via `setup.py`, but you may need to configure NuPlan dataset paths
- **FFmpeg**: Required for video processing. Install via:
  ```bash
  sudo apt-get install ffmpeg  # Ubuntu/Debian
  ```

## Project Structure

```
DriveLaW/
├── DriveLaW-Video/          # Video generation module
│   ├── Infer/               # Inference scripts
│   │   ├── infer.py         # Main inference script
│   │   ├── infer.sh         # Inference shell script
│   │   └── diffusers/       # Modified diffusers library
│   └── Train/               # Training scripts
│       ├── scripts/         # Training utilities
│       ├── configs/         # Training configurations
│       └── src/ltxv_trainer/ # Training framework
├── DriveLaW-Act/            # Planning module
│   ├── navsim/              # NavSim integration
│   │   ├── agents/          # Agent implementations
│   │   ├── planning/        # Planning scripts
│   │   └── ...
│   ├── train_planner.sh     # Training script
│   ├── evaluate_planner.sh  # Evaluation script
│   └── run_caching_drivelaw.sh # Data caching script
├── setup.py                 # Package setup
└── README.md               # This file
```

## Quick Start

### Video Generation (Inference)

Generate driving videos from conditioning frames and text prompts.

#### Basic Usage

```bash
cd DriveLaW-Video/Infer
python infer.py \
  --condition_video demo/condition_videos/scene_0001_conditioning.mp4 \
  --prompt demo/prompts/scene_0001_prompt.txt \
  --output_path ./demo/results/output_0000.mp4 \
  --model_path ./models/LTX-Video-0.9.5-finetune-final \
  --height 704 \
  --width 1280 \
  --num_frames 33 \
  --condition_frames 9 \
  --frame_rate 8 \
  --seed 42 \
  --num_inference_steps 30 \
  --device cuda
```

#### Using Shell Script

Edit `DriveLaW-Video/Infer/infer.sh` to set paths, then run:

```bash
cd DriveLaW-Video/Infer
bash infer.sh
```

**Key Parameters:**
- `--condition_video`: Input conditioning video file
- `--prompt`: Text prompt (file path or string)
- `--model_path`: Path to model checkpoint or HuggingFace model ID
- `--num_frames`: Total frames to generate
- `--condition_frames`: Number of conditioning frames to remove from output

### Video Generation (Training)

Train Video models on custom driving datasets. **Note: You must preprocess your dataset before training.**

#### Step 0: Install the package for video training 


```bash
cd DriveLaW-Video/Train
pip install -e .
```

#### Step 1: Prepare Dataset

Organize your dataset following the structure. The dataset must be in JSON, JSONL, or CSV format with caption and video path columns:

**JSON format example:**
```json
[
  {
    "caption": "A high-quality dashboard camera view of autonomous driving",
    "media_path": "videos/scene_0001.mp4"
  },
  {
    "caption": "Driving on a highway with smooth lane keeping",
    "media_path": "videos/scene_0002.mp4"
  }
]
```

**CSV format example:**
```csv
caption,media_path
"A high-quality dashboard camera view","videos/scene_0001.mp4"
"Driving on a highway","videos/scene_0002.mp4"
```

#### Step 2: (Optional) Split Long Videos into Scenes

If you have long-form videos, split them into shorter scenes:

```bash
python scripts/split_scenes.py input.mp4 scenes_output_dir/ \
    --filter-shorter-than 5s
```

#### Step 3: (Optional) Generate Captions

If your dataset doesn't include captions, generate them automatically:

```bash
cd DriveLaW-Video/Train
python scripts/caption_videos.py scenes_output_dir/ \
    --output scenes_output_dir/dataset.json
```

#### Step 4: Preprocess Dataset (REQUIRED)

**This step is mandatory before training.** Preprocessing computes and caches video latents and text embeddings, significantly accelerating training:

```bash
cd DriveLaW-Video/Train
python scripts/preprocess_dataset.py dataset.json \
    --resolution-buckets "768x768x25" \
    --caption-column "caption" \
    --video-column "media_path" \
    --model-source "./models/LTX-Video-0.9.5"
```

**Important Notes on Resolution Buckets:**
- Spatial dimensions (width × height) must be multiples of 32
- Number of frames must be a multiple of 8 plus 1 (e.g., 25, 33, 41, 49, 89)
- Example: `"768x768x25"`, `"1024x576x65"`, `"768x448x89"`

The preprocessed data will be saved in `.precomputed/` directory with:
- `latents/`: Cached video latents
- `conditions/`: Cached text embeddings

#### Step 5: Configure Training

Edit `DriveLaW-Video/Train/configs/drivelaw_video.yaml`:

```yaml
model:
  model_source: "./models/LTX-Video-0.9.5"
  training_mode: "full" 

data:
  preprocessed_data_root: "./data/.precomputed"  # Path to .precomputed directory

# ... other configurations
```

#### Step 6: Run Training

```bash
cd DriveLaW-Video/Train
python scripts/train.py configs/drivelaw_video.yaml
```

Or use the shell script:

```bash
cd DriveLaW-Video/Train
bash train.sh
```

### Planning Module (Training)

Train the planning agent with video generation capabilities.

#### Step 1: Download NAVSIM datasets
- [Download NAVSIM datasets following official instruction](https://github.com/autonomousvision/navsim/blob/main/docs/install.md)


#### Step 2: Prepare Dataset Cache

Cache the dataset for faster training:

```bash
cd DriveLaW-Act
bash run_caching_drivelaw.sh
```

Edit `run_caching_drivelaw.sh` to set:
- `CACHE_PATH`: Path to cache directory
- `NAVSIM_*` environment variables

#### Step 3: Train Planning Agent

```bash
cd DriveLaW-Act
bash train_planner.sh
```

Edit `train_planner.sh` to configure:
- `GPUS`: Number of GPUs
- `cache_path`: Path to cached dataset
- Training split and other parameters

### Planning Module (Evaluation)

Evaluate the trained planning agent on test scenarios.

```bash
cd DriveLaW-Act
bash evaluate_planner.sh
```

Edit `evaluate_planner.sh` to set:
- `CHECKPOINT`: Path to trained model checkpoint
- Evaluation split (`TRAIN_TEST_SPLIT=navtest`)


## Citation

If you use DriveLaW in your research, please cite:

```bibtex
@article{xia2025drivelaw,
  title={DriveLaW: Unifying Planning and Video Generation in a Latent Driving World},
  author={Xia, Tianze and Li, Yongkang and Zhou, Lijun and Yao, Jingfeng and Xiong, Kaixin and Sun, Haiyang and Wang, Bing and Ma, Kun and Ye, Hangjun and Liu, Wenyu and others},
  journal={arXiv preprint arXiv:2512.23421},
  year={2025}
}
```

## Acknowledgments

- Built on [LTX-Video](https://github.com/Lightricks/LTX-Video)
- Built on [recogdrive](https://github.com/xiaomi-research/recogdrive/tree/main)
- Uses [Diffusers](https://github.com/huggingface/diffusers) library

