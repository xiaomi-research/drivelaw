## DriveLaW-Video Inference

This directory contains the inference entrypoint for the **DriveLaW-Video** world model.
It wraps the LTX-Video conditional pipeline to generate future driving videos given:

- a **conditioning video** clip (e.g., a short dashcam history), and  
- a **text prompt** describing the desired future.

If you want to train or fine-tune the underlying LTX-Video model, see:

- `DriveLaW-Video/Train/README.md`

---

### 1. Requirements

You should already have:

- Installed the root `drivelaw` package (or at least the video training/inference dependencies), e.g.:

```bash
pip install -e .
```

- A CUDA-capable GPU (recommended) and a compatible PyTorch / CUDA installation.
- A **pretrained or fine-tuned LTX-Video checkpoint**, either:
  - a Hugging Face model ID (e.g. `Lightricks/LTX-Video-0.9.7-dev`), or
  - a local directory produced by `DriveLaW-Video/Train`.

---

### 2. Basic usage

The main script is `infer.py`. Minimal example:

```bash
cd DriveLaW-Video/Infer

python infer.py \
  --condition_video demo/condition_videos/scene_0001_conditioning.mp4 \
  --prompt demo/prompts/scene_0001_prompt.txt \
  --output_path ./demo/results/output_0000.mp4 \
  --model_path PATH_TO_LTX_VIDEO_MODEL \
  --height 704 \
  --width 1280 \
  --num_frames 33 \
  --condition_frames 9 \
  --frame_rate 8 \
  --num_inference_steps 30 \
  --device cuda
```

You can also treat `--prompt` as a **raw string** instead of a file; `infer.py` will automatically detect whether
the argument is a path or plain text.

If you prefer a shell wrapper, you can create or edit an `infer.sh` script that calls `infer.py` with your
preferred arguments and then run:

```bash
cd DriveLaW-Video/Infer
bash infer.sh
```

---

### 3. Key arguments

Most commonly used CLI flags:

- **Inputs**
  - `--condition_video` (str, required): path to the conditioning video file (e.g. a short history clip).
  - `--prompt` (str, required): prompt text or path to a `.txt` file containing the prompt.
  - `--output_path` (str, default: `./output.mp4`): where to save the generated video.

- **Model**
  - `--model_path` (str, default: `Lightricks/LTX-Video-0.9.7-dev`): Hugging Face model ID or local checkpoint path.

- **Generation**
  - `--height` / `--width` (int): target resolution for generation.  
    The script will internally round these to the nearest multiples of the VAE spatial compression ratio.
  - `--num_frames` (int, default: `33`): total number of frames to generate (including conditioning).
  - `--condition_frames` (int, default: `0`): number of initial frames to **remove** from the generated video
    when saving (useful if you want only future frames in the output).
  - `--frame_rate` (int, default: `8`): FPS for the output video.
  - `--num_inference_steps` (int, default: `30`): diffusion steps (higher = slower but usually better quality).
  - `--seed` (int, default: `42`): random seed for reproducibility.

- **Quality / safety**
  - `--negative_prompt` (str): default negative prompt to suppress common artifacts
    (blur, jitter, temporal inconsistency, etc.).

- **Noise reinjection (optional, for fast/complex motion)**
  - `--noise_reinjection_enabled`: enable high-frequency noise reinjection to better preserve details,
    especially in high-speed driving scenes.
  - `--noise_reinjection_beta` (float): threshold coefficient β (default `1.0`).
  - `--noise_reinjection_sigma` (float): noise strength σ′ₜ (default `0.1`).
  - `--noise_reinjection_steps` (int or `None`): how many initial steps should use reinjection
    (`None` = all steps).

- **Device**
  - `--device` (str, default: `cuda`): `cuda` or `cpu`. GPU is strongly recommended.

For a complete list, run:

```bash
cd DriveLaW-Video/Infer
python infer.py --help
```

---

### 4. Tips for driving scenarios

- Use **forward-facing ego-view** clips as `--condition_video` for best alignment with the trained model.
- Prefer **multi-frame video conditioning** over a single image: the model was trained to use a short
  temporal history, and providing several frames (e.g. 8–16) gives much more stable motion and
  reduces ambiguity compared to a single still frame.
- Keep resolutions consistent with how the model was trained (see `DriveLaW-Video/Train/README.md`
  for recommended buckets like `768x448x89` or `768x768x25`).
- For high-speed or complex motion scenes, you can experiment with **noise reinjection**:
  - `--noise_reinjection_enabled`: turn on the mechanism described in the paper, which
    (at each denoising step) detects high-frequency regions in pixel space and selectively
    adds noise back to the corresponding latent regions, forcing the model to regenerate
    fine details instead of over-smoothing them.
  - `--noise_reinjection_beta`: controls the adaptive threshold for the high-frequency mask
    (τ = β · std(H_f)). Larger values make the mask more conservative (fewer areas will be
    modified); smaller values expand the masked regions.
  - `--noise_reinjection_sigma`: controls the reinjected noise strength σ′ₜ in the masked
    regions. Higher values encourage stronger re-synthesis of details but may introduce
    instability if set too large (typical values are around `0.1–0.2`).
  - `--noise_reinjection_steps`: limits noise reinjection to the **first N denoising steps**.
    If left as `None`, reinjection is applied to all steps; in practice, restricting it to
    early steps often matches the behavior discussed in the paper.
- Limitations: under the current architecture, the model can still exhibit noticeable motion
  artifacts and ghosting in challenging scenarios. Noise reinjection **does not** fundamentally
  solve the problem or significantly improve general open-domain video generation; it mainly
  alleviates certain artifacts caused by very fast motion in driving videos. Addressing motion
  artifacts from the root (e.g., via architecture and training changes) is left to future work.

