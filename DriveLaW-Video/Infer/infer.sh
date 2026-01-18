#Future Frames = num_frames - condition_frames
python infer.py \
  --condition_video demo/condition_videos/scene_0000_conditioning.mp4 \
  --prompt demo/prompts/scene_0000_prompt.txt \
  --output_path ./demo/results/output_0000.mp4 \
  --model_path path/to/DriveLaw-Video \
  --height 704 \
  --width 1280 \
  --num_frames 33 \
  --condition_frames 9 \
  --frame_rate 8 \
  --seed 42 \
  --num_inference_steps 30 \
  --device cuda \
  --noise_reinjection_enabled \
  --noise_reinjection_beta 1.0 \
  --noise_reinjection_sigma 0.3\
  --noise_reinjection_steps 3

