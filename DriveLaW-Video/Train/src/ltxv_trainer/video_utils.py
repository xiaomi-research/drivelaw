"""Video processing utilities for LTX Video training and inference."""

from pathlib import Path  # noqa: I001

from fractions import Fraction
import torch
from torch import Tensor
import torchvision.transforms as T  # noqa: N812
import torchvision.io
import decord  # Note: Decord must be imported after torch

# Configure decord to use PyTorch tensors
decord.bridge.set_bridge("torch")


def read_video(video_path: str | Path, target_frames: int | None = None) -> tuple[Tensor, float]:
    """Load and sample frames from a video file.

    Args:
        video_path: Path to the video file
        target_frames: Target number of frames to sample. If None, loads all frames.

    Returns:
        Video tensor with shape [F, C, H, W] in range [0, 1] and frames per second (fps).

    Raises:
        ValueError: If video has fewer frames than target_frames
    """
    # Load video using decord
    video_reader = decord.VideoReader(str(video_path))
    fps = video_reader.get_avg_fps()

    total_frames = len(video_reader)

    if target_frames is None:
        # Load all frames
        indices = list(range(total_frames))
        frames = video_reader.get_batch(indices).float() / 255.0  # [F, H, W, C]
    else:
        # Sample frames uniformly to match target frame count
        if total_frames < target_frames:
            raise ValueError(f"Video has {total_frames} frames, but {target_frames} frames are required")

        # Calculate frame indices to sample
        indices = torch.linspace(0, total_frames - 1, target_frames).long()
        frames = video_reader.get_batch(indices.tolist()).float() / 255.0  # [F, H, W, C]

    frames = frames.permute(0, 3, 1, 2)  # [F, H, W, C] -> [F, C, H, W]

    return frames, fps


def resize_video(frames: Tensor, target_width: int, target_height: int) -> Tensor:
    """Resize video frames while maintaining aspect ratio.

    Args:
        frames: Video tensor with shape [F, C, H, W]
        target_width: Target width for resizing
        target_height: Target height for resizing

    Returns:
        Resized video tensor with shape [F, C, H', W'] where H' >= target_height and W' >= target_width
    """
    # Resize maintaining aspect ratio
    current_height, current_width = frames.shape[2:]
    aspect_ratio = current_width / current_height
    target_aspect_ratio = target_width / target_height

    if aspect_ratio > target_aspect_ratio:
        # Width is relatively larger, resize based on height
        resize_height = target_height
        resize_width = int(resize_height * aspect_ratio)
    else:
        # Height is relatively larger, resize based on width
        resize_width = target_width
        resize_height = int(resize_width / aspect_ratio)

    frames = T.functional.resize(
        frames,
        size=[resize_height, resize_width],
        interpolation=T.InterpolationMode.BICUBIC,
        antialias=True,
    )

    return frames


def crop_video(video: Tensor, target_width: int, target_height: int) -> Tensor:
    """Center crop video frames to target dimensions.

    Args:
        video: Video tensor with shape [F, C, H, W]
        target_width: Target width for cropping
        target_height: Target height for cropping

    Returns:
        Cropped video tensor with shape [F, C, target_height, target_width]
    """
    current_height, current_width = video.shape[2:]

    if current_height < target_height or current_width < target_width:
        raise ValueError(
            "Video dimensions are too small for the target dimensions: "
            f"{current_height}x{current_width} -> {target_height}x{target_width}"
        )

    # Center crop to target dimensions
    crop_top = (current_height - target_height) // 2
    crop_left = (current_width - target_width) // 2

    video = T.functional.crop(
        video,
        top=crop_top,
        left=crop_left,
        height=target_height,
        width=target_width,
    )

    return video


def save_video(video_tensor: torch.Tensor, output_path: Path, fps: float = 24.0) -> None:
    """Save a video tensor to a file.

    Args:
        video_tensor: Video tensor of shape [F, C, H, W] in range [0, 255]
        output_path: Path to save the video
        fps: Frames per second for the output video
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to uint8 and correct format for torchvision.io.write_video
    if video_tensor.max() <= 1:
        video_tensor = video_tensor * 255
    video_tensor = video_tensor.to(torch.uint8)
    video_tensor = video_tensor.permute(0, 2, 3, 1)  # [F, C, H, W] -> [F, H, W, C]

    # Save video
    torchvision.io.write_video(
        str(output_path),
        video_tensor.cpu(),
        fps=Fraction(fps).limit_denominator(1000),
        video_codec="h264",
        options={"crf": "18"},
    )
