import io
import subprocess
import sys
from pathlib import Path

import rich
import torch
from PIL import ExifTags, Image, ImageCms, ImageOps
from PIL.Image import Image as PilImage
from safetensors.torch import load_file, save_file

from ltxv_trainer import logger


def get_gpu_memory_gb(device: torch.device) -> float:
    """Get current GPU memory usage in GB using nvidia-smi"""
    try:
        device_id = device.index if device.index is not None else 0
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,nounits,noheader",
                "-i",
                str(device_id),
            ],
            encoding="utf-8",
        )
        return float(result.strip()) / 1024  # Convert MB to GB
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to get GPU memory from nvidia-smi: {e}")
        # Fallback to torch
        return torch.cuda.memory_allocated(device) / 1024**3


def open_image_as_srgb(image_path: str | Path | io.BytesIO) -> PilImage:
    """
    Opens an image file, applies rotation (if it's set in metadata) and converts it
    to the sRGB color space respecting the original image color space .
    """
    exif_colorspace_srgb = 1

    with Image.open(image_path) as img_raw:
        img = ImageOps.exif_transpose(img_raw)

    input_icc_profile = img.info.get("icc_profile")

    # Try to convert to sRGB if the image has ICC profile metadata
    srgb_profile = ImageCms.createProfile(colorSpace="sRGB")
    if input_icc_profile is not None:
        input_profile = ImageCms.ImageCmsProfile(io.BytesIO(input_icc_profile))
        srgb_img = ImageCms.profileToProfile(img, input_profile, srgb_profile, outputMode="RGB")
    else:
        # Try fall back to checking EXIF
        exif_data = img.getexif()
        if exif_data is not None:
            # Assume sRGB if no ICC profile and EXIF has no ColorSpace tag
            color_space_value = exif_data.get(ExifTags.Base.ColorSpace.value)
            if color_space_value is not None and color_space_value != exif_colorspace_srgb:
                raise ValueError(
                    "Image has colorspace tag in EXIF but it isn't set to sRGB,"
                    " conversion is not supported."
                    f" EXIF ColorSpace tag value is {color_space_value}",
                )

        srgb_img = img.convert("RGB")

        # Set sRGB profile in metadata since now the image is assumed to be in sRGB.
        srgb_profile_data = ImageCms.ImageCmsProfile(srgb_profile).tobytes()
        srgb_img.info["icc_profile"] = srgb_profile_data

    return srgb_img


def convert_checkpoint(input_path: str, output_path: str, to_comfy: bool = True) -> None:
    """
    Convert checkpoint format between Diffusers and ComfyUI formats

    Args:
        input_path: Path to input safetensors file
        output_path: Path to save converted safetensors file
        to_comfy: If True, converts "transformer." to "diffusion_model."
                 If False, converts "diffusion_model." to "transformer."
    """
    # Load the state dict
    state_dict = load_file(input_path)

    # Convert the keys
    source_prefix = "transformer." if to_comfy else "diffusion_model."
    target_prefix = "diffusion_model." if to_comfy else "transformer."
    format_name = "ComfyUI" if to_comfy else "Diffusers"

    converted_state_dict = {}
    replaced_count = 0
    for k, v in state_dict.items():
        new_key = k.replace(source_prefix, target_prefix)
        converted_state_dict[new_key] = v
        if new_key != k:
            replaced_count += 1

    if replaced_count == 0:
        rich.print(
            f"No keys were converted. The checkpoint may already be in {format_name} format or "
            f"doesn't contain '{source_prefix}' keys."
        )
        rich.print("[red]Aborting[/red]")
        sys.exit(1)

    # Save the converted state dict
    save_file(converted_state_dict, output_path)
