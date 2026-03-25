## DriveLaW-Video Inference

This directory contains the inference entrypoint for the **DriveLaW-Video** world model.

If you want to train or fine-tune the underlying LTX-Video model:

- `DriveLaW-Video/Train/README.md`


---

### 1. Basic usage

The main script is `infer.py`. Minimal example:

```bash
cd DriveLaW-Video/Infer

python infer.py \
  --condition_video demo/condition_videos/scene_0001_conditioning.mp4 \
  --prompt demo/prompts/scene_0001_prompt.txt \
  --output_path ./demo/results/output_0000.mp4 \ # without condition_frames
  --model_path PATH_TO_DriveLaW-Video \  # use our pretrained DriveLaW-Video or model trained by yourself
  --height 704 \
  --width 1280 \
  --num_frames 33 \ # contains condition_frames
  --condition_frames 9 \
  --frame_rate 8 \
  --num_inference_steps 30 \
  --device cuda
```

You can also treat `--prompt` as a **raw string** instead of a file; `infer.py` will automatically detect whether
the argument is a path or plain text.

or you can edit the `infer.sh` script that calls `infer.py` with your preferred arguments and then run:

```bash
cd DriveLaW-Video/Infer
bash infer.sh
```


---

### 2. Tips for driving scenarios

- Use **forward-facing ego-view** clips as `--condition_video` for best alignment with the trained model.
- Prefer **multi-frame video conditioning** over a single image.
- For high-speed or complex motion scenes, you can experiment with **noise reinjection**:
  - `--noise_reinjection_enabled`
  - `--noise_reinjection_beta`: controls the adaptive threshold for the high-frequency mask
    (τ = β · std(H_f)). Larger values make the mask more conservative (fewer areas will be
    modified); smaller values expand the masked regions.
  - `--noise_reinjection_sigma`: controls the reinjected noise strength σ′ₜ in the masked
    regions. Higher values encourage stronger re-synthesis of details but may introduce
    instability if set too large .
  - `--noise_reinjection_steps`: limits noise reinjection to the **first N denoising steps**.
- Limitation 1: Under the current architecture, the model can still exhibit noticeable motion artifacts and ghosting in challenging scenarios. Noise reinjection does not fundamentally solve the problem or significantly improve general open-domain video generation; it mainly alleviates certain artifacts caused by very fast motion in driving videos.
- Limitation 2: Due to architectural limitations, the performance of the model on long-term video generation has declined to some extent. If you experience not good enough results when using this model for long-term generation, you may try another excellent work [Epona](https://github.com/Kevin-thu/Epona/tree/main).
- We will fundamentally solve the problem in our upcoming work (e.g., via architecture and training changes), stay tuned!

