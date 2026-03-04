## DriveLaW-Video Training 

This repository contains the code and documentation for training a video generation model using the DriveLaW framework. 

### 1. Installation 

To install the video training module (Must-Do):

```bash
cd DriveLaW-Video/Train
pip install -e .
```
### 2. Data Preparation 

The download and preliminary organization methods for the NuPlan and NuScenes datasets are as follows:
- [Data preparation](docs/Data-Preparation.md)

Before model training, data preprocessing is required. The preprocessed data consists of videos and captions. For videos, you can process the data according to your desired length and resolution. For captions, you can use the vlm annotation method or organize the data in ego-pose into natural language for processing (as described in the paper appendix).

### 3. Dataset preprocessing 

The trainer supports either videos or single images. Note that your dataset must be homogeneous - either all videos or all images, mixing is not supported. When using images, follow the same preprocessing steps and format requirements as with videos, simply provide image files instead of video files.

The dataset must be a CSV, JSON, or JSONL metadata file with columns for captions and video paths:

**JSON format example:**
```json
[
  {
    "caption": "A cat playing with a ball of yarn",
    "media_path": "videos/cat_playing.mp4"
  },
  {
    "caption": "A dog running in the park",
    "media_path": "videos/dog_running.mp4"
  }
]
```

**JSONL format example:**
```jsonl
{"caption": "A cat playing with a ball of yarn", "media_path": "videos/cat_playing.mp4"}
{"caption": "A dog running in the park", "media_path": "videos/dog_running.mp4"}
```

**CSV format example:**
```csv
caption,media_path
"A cat playing with a ball of yarn","videos/cat_playing.mp4"
"A dog running in the park","videos/dog_running.mp4"
```

Videos are organized into "buckets" of specific dimensions (width × height × frames). Each video is assigned to the nearest matching bucket.

**Currently, the trainer only supports using a single resolution bucket.**

The dimensions of each bucket must follow these constraints due to  VAE architecture:

- Spatial dimensions (width and height) must be multiples of 32
- Number of frames must be a multiple of 8 plus 1 (e.g., 25, 33, 41, 49, etc.)

**Guidelines for choosing training resolution:**

- For high-quality, detailed videos: use larger spatial dimensions (e.g. 1280x704) with fewer frames (e.g. 33)
- For longer, motion-focused videos: use smaller spatial dimensions (512×512) with more frames (121)
- Memory usage increases with both spatial and temporal dimensions

**Example usage:**

```bash
python scripts/preprocess_dataset.py /path/to/dataset \
    --resolution-buckets "1280x704x33"
```
or you can use:

```bash
bash preprocess.sh
```

**Video processing workflow:**
1. Videos are **resized** maintaining aspect ratio until either width or height matches the target 
2. The larger dimension is **center cropped** to match the bucket's dimensions
3. Only the **first X frames are taken** to match the bucket's frame count (25 in this example), remaining frames are ignored.

> [!NOTE]
> The sequence length processed by the transformer model can be calculated as:
>
> ```
> sequence_length = (H/32) * (W/32) * ((F-1)/8 + 1)
> ```
>
> Where:
>
> - H = Height of video's latent
> - W = Width of video's latent
> - F = Number of latent frames
> - 32 = VAE's spatial downsampling factor
> - 8 = VAE's temporal downsampling factor
>
> For example, a 768×448×89 video would have sequence length:
>
> ```
> (768/32) * (448/32) * ((89-1)/8 + 1) = 24 * 14 * 12 = 4,032
> ```
>
> Keep this in mind when choosing video dimensions, as longer sequences require more GPU memory and computation power.

> [!WARNING]
> While the preprocessing script supports multiple buckets, the trainer currently only works with a single resolution bucket.
> Please ensure you specify just one bucket in your preprocessing command.

The preprocessed data is saved in a `.precomputed` directory:

```
dataset/
└── .precomputed/
    ├── latents/           # Cached video latents
    ├── conditions/        # Cached text embeddings
```


### 4. Configuring training

The main DriveLaW config example is `configs/drivelaw_video.yaml`. Typical fields to modify:

- `model.model_source`: your local LTX-Video-0.9.5 model weights, you can download from [Hugging Face](https://huggingface.co/Lightricks/LTX-Video-0.9.5).
- `data.preprocessed_data_root`: path to the `.precomputed` directory produced by preprocessing.
- `output_dir`: where checkpoints will be written.
- Training mode:
  - `model.training_mode: "full"` for full fine-tuning.
 
### 5. Launching training

Single-GPU / single-node example:

```bash
cd DriveLaW-Video/Train
python scripts/train.py configs/drivelaw_video.yaml
```

Multi-GPUS / Multi-Nodes example:

```bash
cd DriveLaW-Video/Train
bash train.sh
```

This will produce checkpoints under your configured `output_dir`:

- The **transformer weights** (e.g. `diffusion_pytorch_model.safetensors`).
- The associated **config** file.
- Sample Videos.



