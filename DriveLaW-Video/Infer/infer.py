"""
Generate videos from conditioning video and prompt (text or file).
"""
import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image

# Add local diffusers library to path (use modified diffusers in video_generation folder)
_script_dir = Path(__file__).parent.absolute()
_diffusers_path = _script_dir / "diffusers" / "src"
if str(_diffusers_path) not in sys.path:
    sys.path.insert(0, str(_diffusers_path))
    print(f"Using local diffusers library from: {_diffusers_path}")

from diffusers import LTXConditionPipeline
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video, load_video


class LTXInferenceEngine:
    
    def __init__(self, model_path="", device="cuda"):
        
        # Load main pipeline
        self.pipe = LTXConditionPipeline.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16
        )
        self.pipe.to(device)
        self.pipe.vae.enable_tiling()
        
        self.device = device
        self.vae_compression = self.pipe.vae_spatial_compression_ratio
        
    
    
    def round_to_vae_resolution(self, height, width):
        """Round resolution to VAE-acceptable values"""
        height = height - (height % self.vae_compression)
        width = width - (width % self.vae_compression)
        return height, width
    
    def generate_video(self, conditioning_frames, prompt, height, width, num_frames, 
                  negative_prompt="worst quality, low quality, blurry, jittery, distorted, motion blur, ghosting, flickering, stuttering, camera shake, unstable footage, warping, trailing artifacts, temporal inconsistency, jerky motion, choppy framerate",
                  num_inference_steps=30, seed=42,
                  noise_reinjection_enabled=False, noise_reinjection_beta=1.0, noise_reinjection_sigma=0.1,
                  noise_reinjection_steps=None):
        """
        Generate video
        
        Args:
            conditioning_frames: List of conditioning frames (PIL Images or numpy arrays)
            prompt: Text prompt
            height: Target height
            width: Target width
            num_frames: Number of frames to generate
            negative_prompt: Negative prompt
            num_inference_steps: Number of inference steps
            seed: Random seed
            noise_reinjection_enabled: Enable noise reinjection for high-frequency detail preservation
            noise_reinjection_beta: Threshold coefficient for adaptive high-frequency mask computation
            noise_reinjection_sigma: Noise strength for reinjection step
            noise_reinjection_steps: Number of initial steps to apply noise reinjection (None = all steps)
            
        Returns:
            Generated video frames (list of PIL Images)
        """
        use_transition_anchor = True
        transition_anchor_strength = 1.0
        
        # Build conditions
        conditions = []
        
        # Condition 1: First N frames as starting condition
        conditions.append(LTXVideoCondition(
            video=conditioning_frames, 
            frame_index=0,
            strength=1.0  # Strong constraint on starting frames
        ))
        
        # Condition 2: Last frame as transition anchor
        if use_transition_anchor and len(conditioning_frames) > 0:
            last_frame = conditioning_frames[-1]
            anchor_frame_index = len(conditioning_frames)
            
            conditions.append(LTXVideoCondition(
                image=last_frame,
                frame_index=anchor_frame_index,
                strength=transition_anchor_strength
            ))
            
            print(f"  Added transition anchor: frame {len(conditioning_frames)-1} -> frame {anchor_frame_index} (strength={transition_anchor_strength})")
        
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Generate directly at target resolution
        height, width = self.round_to_vae_resolution(height, width)
        print(f"  Generating directly at {width}x{height}...")
        
        video = self.pipe(
            conditions=conditions,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            generator=generator,
            output_type="pil",
            noise_reinjection_enabled=noise_reinjection_enabled,
            noise_reinjection_beta=noise_reinjection_beta,
            noise_reinjection_sigma=noise_reinjection_sigma,
            noise_reinjection_steps=noise_reinjection_steps,
        ).frames[0]
        
        return video


def load_conditioning_video(video_path):
    """
    Load conditioning video and extract frames
    
    Args:
        video_path: Path to input video file
        
    Returns:
        List of PIL Images
    """
    print(f"Loading conditioning video: {video_path}")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Load video using diffusers utility
    video_frames = load_video(video_path)
    
    # Convert to PIL Images if needed
    pil_frames = []
    for frame in video_frames:
        if isinstance(frame, Image.Image):
            pil_frames.append(frame)
        else:
            # Convert numpy array to PIL Image
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
            pil_frames.append(Image.fromarray(frame))
    
    print(f"Loaded {len(pil_frames)} frames from conditioning video")
    return pil_frames


def load_prompt(prompt_input):
    """
    Load prompt from string or file
    
    Args:
        prompt_input: Prompt string or path to prompt file
        
    Returns:
        Prompt string
    """
    prompt_path = Path(prompt_input)
    
    # Check if it's a file path
    if prompt_path.exists() and prompt_path.is_file():
        print(f"Loading prompt from file: {prompt_input}")
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        print(f"Prompt loaded ({len(prompt)} characters)")
        return prompt
    else:
        # Treat as direct prompt string
        print(f"Using prompt from command line ({len(prompt_input)} characters)")
        return prompt_input


def save_video(frames, output_path, fps=8):
    """Save PIL Image frames to video file"""
    try:
        export_to_video(frames, output_path, fps=fps)
        print(f"Video saved: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving video {output_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="LTX Video Generation - Generate videos from conditioning video and prompt"
    )
    
    # Input parameters
    parser.add_argument(
        "--condition_video", 
        type=str, 
        required=True,
        help="Path to conditioning video file"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        required=True,
        help="Text prompt (direct string) or path to prompt text file (.txt)"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="./output.mp4",
        help="Output video path"
    )
    
    # Model parameters
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="Lightricks/LTX-Video-0.9.7-dev",
        help="HuggingFace model ID or local path"
    )
    
    # Generation parameters
    parser.add_argument(
        "--height", 
        type=int, 
        default=704, 
        help="Output video height"
    )
    parser.add_argument(
        "--width", 
        type=int, 
        default=1280, 
        help="Output video width"
    )
    parser.add_argument(
        "--num_frames", 
        type=int, 
        default=33, 
        help="Total number of frames to generate (including conditioning frames)"
    )
    parser.add_argument(
        "--condition_frames", 
        type=int, 
        default=0,
        help="Number of conditioning frames to remove from the beginning of generated video (0 = keep all frames)"
    )
    parser.add_argument(
        "--frame_rate", 
        type=int, 
        default=8, 
        help="Frame rate for output video"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed"
    )
    parser.add_argument(
        "--num_inference_steps", 
        type=int, 
        default=30, 
        help="Number of diffusion inference steps"
    )
    parser.add_argument(
        "--negative_prompt", 
        type=str, 
        default="worst quality, low quality, blurry, jittery, distorted, motion blur, ghosting, flickering, stuttering, camera shake, unstable footage, warping, trailing artifacts, temporal inconsistency, jerky motion, choppy framerate",
        help="Negative prompt"
    )
    parser.add_argument(
        "--noise_reinjection_enabled",
        action="store_true",
        help="Enable noise reinjection for high-frequency detail preservation (useful for high-speed driving videos)"
    )
    parser.add_argument(
        "--noise_reinjection_beta",
        type=float,
        default=1.0,
        help="Threshold coefficient β for adaptive high-frequency mask computation (default: 1.0)"
    )
    parser.add_argument(
        "--noise_reinjection_sigma",
        type=float,
        default=0.1,
        help="Noise strength σ'ₜ for the reinjection step (default: 0.1)"
    )
    parser.add_argument(
        "--noise_reinjection_steps",
        type=int,
        default=None,
        help="Number of initial steps to apply noise reinjection (None = all steps, default: None)"
    )
    
    
    # Device
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda", 
        help="Device to run inference on (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("LTX VIDEO GENERATION")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Conditioning video: {args.condition_video}")
    print(f"Prompt: {args.prompt[:100]}..." if len(args.prompt) > 100 else f"Prompt: {args.prompt}")
    print(f"Output: {args.output_path}")
    print(f"Resolution: {args.width}x{args.height} @ {args.frame_rate}fps")
    print(f"Frames to generate: {args.num_frames}")
    print(f"Condition frames to remove: {args.condition_frames}")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"Noise reinjection enabled: {args.noise_reinjection_enabled}")
    if args.noise_reinjection_enabled:
        print(f"  Beta: {args.noise_reinjection_beta}")
        print(f"  Sigma: {args.noise_reinjection_sigma}")
        if args.noise_reinjection_steps is not None:
            print(f"  Steps: first {args.noise_reinjection_steps} steps only")
        else:
            print(f"  Steps: all steps")
    print("=" * 80)
    
    try:
        # Load prompt (from string or file)
        prompt = load_prompt(args.prompt)
        
        # Load conditioning video
        conditioning_frames = load_conditioning_video(args.condition_video)
        
        if len(conditioning_frames) == 0:
            raise ValueError("Conditioning video contains no frames")
        
        print(f"Conditioning frames: {len(conditioning_frames)}")

        ltx_engine = LTXInferenceEngine(
            model_path=args.model_path,
            device=args.device
        )
        
        # Generate video
        print("\nGenerating video...")
        video_frames = ltx_engine.generate_video(
            conditioning_frames=conditioning_frames,
            prompt=prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            noise_reinjection_enabled=args.noise_reinjection_enabled,
            noise_reinjection_beta=args.noise_reinjection_beta,
            noise_reinjection_sigma=args.noise_reinjection_sigma,
            noise_reinjection_steps=args.noise_reinjection_steps,
        )
        
        # Remove conditioning frames from the beginning if specified
        if args.condition_frames > 0:
            if len(video_frames) <= args.condition_frames:
                raise ValueError(f"Cannot remove {args.condition_frames} frames: video only has {len(video_frames)} frames")
            print(f"\nRemoving first {args.condition_frames} conditioning frames...")
            video_frames = video_frames[args.condition_frames:]
            print(f"Remaining frames: {len(video_frames)}")
        
        # Save video
        print(f"\nSaving video to {args.output_path}...")
        os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else ".", exist_ok=True)
        
        if save_video(video_frames, args.output_path, args.frame_rate):
            print("\n" + "=" * 80)
            print("SUCCESS")
            print("=" * 80)
            print(f"Generated video saved to: {args.output_path}")
            print(f"Total frames: {len(video_frames)}")
            print(f"Duration: {len(video_frames) / args.frame_rate:.2f} seconds")
        else:
            print("\nFailed to save video")
            return 1
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
