# Training Guide

This guide covers how to run training jobs, from basic single-GPU training to advanced distributed setups and automatic model uploads.

## ⚡ Basic Training

After preprocessing your dataset and preparing a configuration file, you can start training using the trainer script:

```bash
python scripts/train.py <PATH_TO_CONFIG_YAML_FILE>
```

The trainer will:
1. **Load your configuration** and validate all parameters
2. **Initialize models** and apply optimizations
3. **Run the training loop** with progress tracking
4. **Generate validation videos** (if configured)
5. **Save the trained weights** in your output directory

### Output Files

**For LoRA training:**
- `lora_weights.safetensors` - Main LoRA weights file
- `training_config.yaml` - Copy of training configuration
- `validation_samples/` - Generated validation videos (if enabled)

**For full model fine-tuning:**
- `model_weights.safetensors` - Full model weights
- `training_config.yaml` - Copy of training configuration
- `validation_samples/` - Generated validation videos (if enabled)

## 🖥️ Distributed / Multi-GPU Training

For larger training jobs, you can run the trainer across multiple GPUs on a single machine using our
distributed training script, which leverages [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/index).

### Basic Multi-GPU Setup

Use the provided script:

```bash
python scripts/train_distributed.py CONFIG_PATH [OPTIONS]
```

### Examples

```bash
# Launch distributed training on all available GPUs
python scripts/train_distributed.py configs/ltxv_2b_full.yaml

# Specify the number of processes/GPUs explicitly
CUDA_VISIBLE_DEVICES=0,1 python scripts/train_distributed.py configs/ltxv_2b_lora.yaml --num_processes 2
```

### Available Options

- `--num_processes`: Number of GPUs/processes to use (overrides auto-detection)
- `--disable_progress_bars`: Disables rich progress bars (recommended for multi-GPU runs)

### Benefits of Distributed Training

- **Faster training**: Distribute workload across multiple GPUs
- **Larger effective batch sizes**: Combine gradients from multiple GPUs
- **Memory efficiency**: Each GPU handles a portion of the batch

> [!NOTE]
> Distributed training requires that all GPUs have sufficient memory for the model and batch size. The effective batch size becomes `batch_size × num_processes`.

## 🤗 Pushing Models to Hugging Face Hub

You can automatically push your trained models to the Hugging Face Hub by adding the following to your configuration YAML:

```yaml
hub:
  push_to_hub: true
  hub_model_id: "your-username/your-model-name"  # Your HF username and desired repo name
```

### Prerequisites

Before pushing, make sure you:

1. **Have a Hugging Face account** - Sign up at [huggingface.co](https://huggingface.co)
2. **Are logged in** via `huggingface-cli login` or have set the `HUGGING_FACE_HUB_TOKEN` environment variable
3. **Have write access** to the specified repository (it will be created if it doesn't exist)

### Login Options

**Option 1: Interactive login**
```bash
huggingface-cli login
```

**Option 2: Environment variable**
```bash
export HUGGING_FACE_HUB_TOKEN="your_token_here"
```

### What Gets Uploaded

The trainer will automatically:

- **Create a model card** with training details and sample outputs
- **Upload model weights** (both original and ComfyUI-compatible versions)
- **Push sample videos as GIFs** in the model card
- **Include training configuration and prompts**

### Repository Structure

Your Hub repository will contain:
```
your-repo/
├── README.md                    # Auto-generated model card
├── lora_weights.safetensors     # Main weights file
├── lora_weights_comfy.safetensors  # ComfyUI-compatible version
├── training_config.yaml        # Training configuration
└── sample_videos/              # Validation samples as GIFs
    ├── sample_001.gif
    └── sample_002.gif
```

## 🔄 Complete Automated Pipeline

For a streamlined experience that combines all steps, you can use `run_pipeline.py` which automates the entire training workflow:

```bash
python scripts/run_pipeline.py [LORA_BASE_NAME] \
    --resolution-buckets "768x768x49" \
    --config-template configs/ltxv_2b_lora_template.yaml \
    --rank 32
```

### What the Pipeline Does

1. **Process raw videos** in `[basename]_raw/` directory (if they exist):
   - Split long videos into scenes
   - Save scenes to `[basename]_scenes/`

2. **Generate captions** for the scenes (if scenes exist):
   - Uses Qwen-2.5-VL for captioning
   - Saves captions to `[basename]_scenes/captions.json`

3. **Preprocess the dataset**:
   - Computes and caches video latents
   - Computes and caches text embeddings
   - Decodes videos for verification

4. **Run the training**:
   - Uses the provided config template
   - Automatically extracts validation prompts from captions
   - Saves the final model weights

5. **Convert LoRA to ComfyUI format**:
   - Automatically converts the trained LoRA weights to ComfyUI format
   - Saves the converted weights with "_comfy" suffix

### Required Arguments

- `basename`: Base name for your project (e.g., "slime")
- `--resolution-buckets`: Video resolution in format "WxHxF" (e.g., "768x768x49")
- `--config-template`: Path to your configuration template file
- `--rank`: LoRA rank (1-128) for training

### Directory Structure Created

```
[basename]_raw/          # Place your raw videos here
[basename]_scenes/       # Split scenes and captions
└── .precomputed/       # Preprocessed data
    ├── latents/       # Cached video latents
    ├── conditions/    # Cached text embeddings
    └── decoded_videos/ # Decoded videos for verification
outputs/                # Training outputs and checkpoints
    └── lora_weights_comfy.safetensors  # ComfyUI-compatible LoRA weights
```

## 🚀 Next Steps

After training completes:

- **Test your model** with validation prompts
- **Convert for ComfyUI** using [utility scripts](utility-scripts.md)
- **Share your results** by pushing to Hugging Face Hub
- **Iterate and improve** based on validation results

## 💡 Tips for Successful Training

- **Start small**: Begin a small dataset and with a few hundred steps to verify everything works
- **Monitor validation**: Keep an eye on validation samples to catch overfitting
- **Adjust learning rate**: Lower learning rates often produce better results
- **Use gradient checkpointing**: Essential for LTXV 13B training on consumer GPUs
- **Save checkpoints**: Regular checkpoints help recover from interruptions

## Need Help?

If you encounter issues during training, see the [Troubleshooting Guide](troubleshooting.md).
