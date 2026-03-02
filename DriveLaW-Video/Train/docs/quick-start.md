# DriveLaW-Video Training: Quick Start

This guide gives you a minimal, DriveLaW-focused path to train the video world model.

The full documentation with more details can be found in:

- `dataset-preparation.md`
- `Data-preparation.md`
- `configuration-reference.md`
- `training-guide.md`

---

## 1. Install the training module

From the root of the DriveLaW repository:

```bash
pip install -e .
```

Or, if you only want the video training code:

```bash
cd DriveLaW-Video/Train
pip install -e .
```

This installs all required Python dependencies (PyTorch, diffusers, accelerate, etc.).

---

## 2. Prepare your data

There are two parts to data preparation:

1. **Download and organize driving datasets** (e.g. NuPlan, nuScenes, your own logs)
   - See `Data-preparation.md` for how to download and structure the datasets
     used in the DriveLaW paper.

2. **Build the training metadata and precompute latents**
   - See `dataset-preparation.md` for:
     - splitting long videos into shorter scenes (optional)
     - generating captions (optional)
     - running `scripts/preprocess_dataset.py` to compute video latents and
       text embeddings into a `.precomputed/` directory.

At the end of this step you should have:

- a metadata file (CSV / JSON / JSONL) with `caption` and `media_path` columns, and
- a `.precomputed/` directory with `latents/` and `conditions/`.

---

## 3. Configure training

Use `configs/drivelaw_video.yaml` as a starting point and edit:

- `model.model_source`: base LTX-Video checkpoint (HF ID or local path),
- `data.preprocessed_data_root`: path to your preprocessed dataset,
- `output_dir`: where to save logs and checkpoints,
- `model.training_mode: "full"` for full fine-tuning (the mode used in DriveLaW).

See `configuration-reference.md` for a complete description of the available options.

---

## 4. Launch training

Basic single-node training:

```bash
cd DriveLaW-Video/Train
python scripts/train.py configs/drivelaw_video.yaml
```

For larger jobs or multiple GPUs, see `training-guide.md` for examples using
the distributed training script.

---

## 5. Troubleshooting

If you encounter issues (out-of-memory, dataset errors, etc.), consult:

- `troubleshooting.md`

for common problems and suggested fixes.
