## DriveLaW-Video Training (LTX-Video-Trainer)

This directory wraps the official **LTX-Video-Trainer** to train the video world model used in DriveLaW.
If you are already familiar with LTX-Video, you can use it in the same way; this README only highlights
the driving-specific recommendations and how it connects to DriveLaW-Act.

For full trainer documentation, please refer to:

- `docs/quick-start.md`
- `docs/Data-preparation.md`        # Dataset download and organization (NuPlan, nuScenes, etc.)
- `docs/dataset-preparation.md`
- `docs/training-guide.md`
- `docs/training-modes.md`

### 1. Installation (module-only)

If you have installed the root `drivelaw` package already, you can skip this section.
To install just the video training module:

```bash
cd DriveLaW-Video/Train
pip install -e .
```

This installs the same dependencies as the upstream LTX-Video-Trainer (`accelerate`, `diffusers`, `torch`, etc.).

### 2. Data sources and download (NuPlan / nuScenes, etc.)

If you want to **reproduce the experiments in the DriveLaW paper**, please first prepare
datasets such as NuPlan / nuScenes following:

- `docs/Data-preparation.md` – step-by-step instructions on downloading and organizing
  the required driving datasets and creating the metadata JSON files.

You can also use your **own driving video datasets**; in that case you only need to
ensure they are organized into a simple metadata file as described below.

### 3. Dataset format and driving-specific notes

The trainer expects a **metadata file** (CSV / JSON / JSONL) describing your dataset, for example:

```json
[
  {
    "caption": "A high-quality dashboard camera view of autonomous driving",
    "media_path": "videos/scene_0001.mp4"
  },
  {
    "caption": "Driving on a curved highway with lane keeping and moderate traffic",
    "media_path": "videos/scene_0002.mp4"
  }
]
```

Key recommendations for driving data:

- **Viewpoint**:
  - Prefer **forward-facing dashcam** or similar ego-view trajectories.
  - Avoid mixing strongly different camera geometries (e.g., third-person, side-only) in the same bucket when possible.
- **Resolution / aspect ratio**:
  - Common choices for driving are:
    - `768x448x89` (wider aspect, long temporal horizon)
    - `768x768x25` (square-ish, shorter horizon with higher spatial detail)
- **Frame count**:
  - Must follow the LTX-Video constraint: number of frames $F$ satisfies $F = 8n + 1$ (e.g. 25, 33, 41, 49, 89, 121).
  - For DriveLaW, we typically use:
    - A **long-horizon, lower-res** stage (e.g. `740x352x121`) for motion diversity.
    - A **high-res, shorter** stage (e.g. `1280x704x25`) for visual quality.

See `docs/dataset-preparation.md` for more details about buckets and sequence length.

### 4. Preprocessing (mandatory before training)

Preprocessing computes and caches:

- Resized / bucketized video clips
- VAE **latents**
- Text encoder **embeddings** for all captions

Example command:

```bash
cd DriveLaW-Video/Train
python scripts/preprocess_dataset.py path/to/dataset.json \
  --resolution-buckets "768x448x89" \
  --caption-column caption \
  --video-column media_path \
  --model-source PATH_TO_LTXV_BASE_MODEL
```

This will create a `.precomputed/` directory next to your dataset, with:

- `latents/`: cached video latents
- `conditions/`: cached text embeddings

Make sure to use **a single bucket** per run; the current trainer only supports one bucket at a time.

### 5. Configuring training

The main DriveLaW config example is `configs/drivelaw_video.yaml`. Typical fields to modify:

- `model.model_source`: path or HF ID of your base LTX-Video model (e.g. `./models/LTX-Video-0.9.5`).
- `data.preprocessed_data_root`: path to the `.precomputed` directory produced by preprocessing.
- `output_dir`: where logs and checkpoints will be written.
- Training mode: for reproducing DriveLaW we use full fine-tuning
  (`model.training_mode: "full"`). Other modes are optional and not required
  for the paper.

### 6. Launching training

Single-GPU / single-node example:

```bash
cd DriveLaW-Video/Train
python scripts/train.py configs/drivelaw_video.yaml
```

Or via the helper script:

```bash
cd DriveLaW-Video/Train
bash train.sh
```

This will produce checkpoints under your configured `output_dir`. The important artifacts for DriveLaW-Act are:

- The **transformer weights** (e.g. `diffusion_pytorch_model.safetensors`).
- The associated **config** file.

These are later referenced in the planner configs under
`DriveLaW-Act/navsim/agents/videodrive/configs/ltx_model/`, e.g.:

- `pretrained_model_name_or_path`
- `diffusion_model.model_path`

### 7. Using trained models in DriveLaW-Act

Once training has finished:

1. Copy or symlink the LTX-Video checkpoint directory to a location visible to your training cluster.
2. Edit `DriveLaW-Act/navsim/agents/videodrive/configs/ltx_model/video_model_infer_navsim.yaml` (and `*_eval*.yaml`) to point to:
   - `pretrained_model_name_or_path: /path/to/LTX-Video-0.9.5-finetune`
   - `diffusion_model.model_path: /path/to/.../transformer/diffusion_pytorch_model.safetensors`
3. Follow the main `README.md` for planner caching, training, and evaluation.

This completes the video side of the DriveLaW pipeline; the planner then reuses your LTX-Video model
to obtain consistent latent representations for both video prediction and action planning.