# Training Modes (DriveLaW Focus)

The trainer technically supports multiple training modes, but for **reproducing
the DriveLaW paper**, we use **full model fine-tuning** only.

## 🔥 Full Model Fine-tuning (recommended for DriveLaW)

Full model fine-tuning updates all parameters of the base model and is the mode
used in our experiments:

- **Highest potential quality and capability improvements**
- **Requires more GPU memory** (use smaller models / resolutions if resources are limited)
- **Produces full checkpoints** that can be plugged into DriveLaW-Act

Configure full fine-tuning with:

```yaml
model:
  training_mode: "full"
conditioning:
  mode: "none"
```

When to use full fine-tuning:

- Training the world model used in the DriveLaW paper
- You want a single, strong model for driving video prediction

> **Note:** LoRA / IC-LoRA training modes are supported by the underlying
> trainer but are **not required** for reproducing DriveLaW, so they are not
> documented here to keep the workflow simple.

## 🚀 Next Steps

- Set up your dataset using [Dataset Preparation](dataset-preparation.md)
- Configure your training parameters in [Configuration Reference](configuration-reference.md)
- Start training with the [Training Guide](training-guide.md)
