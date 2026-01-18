import torch
from diffusers import AutoencoderKLLTXVideo
from torch import Tensor
from transformers import T5EncoderModel, T5Tokenizer


def encode_prompt(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: str | list[str],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    max_sequence_length: int = 256,
) -> dict[str, Tensor]:
    """Prepares text prompt embeddings and attention mask for model input.

    Args:
        tokenizer: T5 tokenizer for encoding text
        text_encoder: T5 encoder model for generating embeddings
        prompt: Text prompt or list of prompts
        device: Target device for tensors
        dtype: Target dtype for tensors
        max_sequence_length: Maximum sequence length for tokenization

    Returns:
        Dict containing prompt embeddings and attention mask
    """
    device = device or text_encoder.device
    dtype = dtype or text_encoder.dtype

    if isinstance(prompt, str):
        prompt = [prompt]

    return _encode_prompt_t5(tokenizer, text_encoder, prompt, device, dtype, max_sequence_length)


def pack_latents(
    latents: Tensor,
    spatial_patch_size: int = 1,
    temporal_patch_size: int = 1,
) -> Tensor:
    """Reshapes latents [B,C,F,H,W] into patches and flattens to sequence form [B,L,D].

    Args:
        latents: Input latent tensor
        spatial_patch_size: Size of spatial patches
        temporal_patch_size: Size of temporal patches

    Returns:
        Flattened sequence of patches
    """
    b, c, f, h, w = latents.shape
    latents = latents.reshape(
        b,
        -1,
        f // temporal_patch_size,
        temporal_patch_size,
        h // spatial_patch_size,
        spatial_patch_size,
        w // spatial_patch_size,
        spatial_patch_size,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
    return latents


def encode_video(
    vae: AutoencoderKLLTXVideo,
    image_or_video: Tensor,
    patch_size: int = 1,
    patch_size_t: int = 1,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, Tensor | int]:
    """Encodes input images/videos into latent representations.

    Args:
        vae: VAE model for encoding
        image_or_video: Input tensor of shape [B,C,F,H,W] or [B,C,1,H,W]
        patch_size: Spatial patch size
        patch_size_t: Temporal patch size
        device: Target device for tensors
        dtype: Target dtype for tensors
        generator: Random number generator

    Returns:
        Dict containing latents and shape information
    """
    device = device or vae.device

    if image_or_video.ndim == 4:
        image_or_video = image_or_video.unsqueeze(2)
    assert image_or_video.ndim == 5, f"Expected 5D tensor, got {image_or_video.ndim}D tensor"

    image_or_video = image_or_video.to(device=device, dtype=vae.dtype)
    image_or_video = image_or_video.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, F, H, W] -> [B, F, C, H, W]

    # Encode image/video.
    latents = vae.encode(image_or_video).latent_dist.sample(generator=generator)
    latents = latents.to(dtype=dtype)
    _, _, num_frames, height, width = latents.shape

    # Normalize to zero mean and unit variance.
    latents = _normalize_latents(latents, vae.latents_mean, vae.latents_std)

    # Patchify and pack latents to a sequence expected by the transformer.
    latents = pack_latents(latents, patch_size, patch_size_t)
    return {"latents": latents, "num_frames": num_frames, "height": height, "width": width}


def decode_video(  # noqa: PLR0913
    vae: AutoencoderKLLTXVideo,
    latents: Tensor,
    num_frames: int,
    height: int,
    width: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    patch_size: int = 1,
    patch_size_t: int = 1,
    decode_timestep: float = 0.0,
    decode_noise_scale: float | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Decodes latent representations back into videos.

    This function reverses the encoding process performed by encode_video().
    It takes the packed latents and shape information and reconstructs the original video.

    Args:
        vae: VAE model for decoding
        latents: Latent tensor as saved by encode_video()
        num_frames: Number of latents frames in the latent tensor
        height: Height of the latent representation
        width: Width of the latent representation
        device: Target device for tensors
        dtype: Target dtype for tensors
        patch_size: Spatial patch size for unpacking
        patch_size_t: Temporal patch size for unpacking
        decode_timestep: The timestep to use for decoding (default: 0.0)
        decode_noise_scale: Scale factor for noise to add before decoding (default: same as decode_timestep)
        generator: Random number generator for noise generation

    Returns:
        Decoded video tensor of shape [C,F,H,W]
    """
    device = device or vae.device
    latents = latents.to(device=device, dtype=vae.dtype)

    # Add batch dimension if not present
    if latents.dim() == 1:
        latents = latents.unsqueeze(0)

    # Unpack the latents from [B,L,D] to [B,C,F,H,W]
    latents = latents.reshape(
        1,  # batch size
        num_frames // patch_size_t,
        height // patch_size,
        width // patch_size,
        -1,  # channels * patch sizes
        patch_size_t,
        patch_size,
        patch_size,
    )
    latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7)
    latents = latents.reshape(1, -1, num_frames, height, width)
    # Denormalize latents exactly as in pipeline's _denormalize_latents
    latents_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = vae.latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents = latents * latents_std / vae.config.scaling_factor + latents_mean

    # Add noise before decoding if specified
    if decode_noise_scale is None:
        decode_noise_scale = decode_timestep

    # Generate random noise
    noise = torch.randn(latents.shape, generator=generator, device=device, dtype=latents.dtype)

    decode_noise_scale = torch.tensor([decode_noise_scale], device=device, dtype=latents.dtype).view(1, 1, 1, 1, 1)
    latents = (1 - decode_noise_scale) * latents + decode_noise_scale * noise

    # Decode the latents with timestep
    timestep = torch.tensor([decode_timestep], device=device, dtype=latents.dtype)
    video = vae.decode(latents, timestep, return_dict=False)[0]
    video *= 0.5
    video += 0.5
    video = video.to(dtype=dtype) if dtype is not None else video

    return video


def get_rope_scale_factors(fps: float) -> list[float]:
    """Get ROPE interpolation scale factors for video transformers.

    Args:
        fps: Frames per second

    Returns:
        List of scale factors [temporal_scale, spatial_scale, spatial_scale]
    """
    if fps <= 0:
        raise ValueError("FPS must be a positive number.")

    temporal_compression_ratio = 8.0
    spatial_compression_ratio = 32.0

    return [
        temporal_compression_ratio / fps,
        spatial_compression_ratio,
        spatial_compression_ratio,
    ]


def prepare_video_coordinates(
    num_frames: int,
    height: int,
    width: int,
    batch_size: int,
    sequence_multiplier: int = 1,
    device: torch.device | None = None,
) -> Tensor:
    """Prepare video coordinates for positional embeddings.

    Args:
        num_frames: Number of frames
        height: Height in latent space
        width: Width in latent space
        batch_size: Batch size
        sequence_multiplier: Multiplier for sequence length (2 for IC-LoRA)
        device: Target device for tensors

    Returns:
        Video coordinates tensor of shape [batch_size, 3, sequence_length * sequence_multiplier]
    """
    if device is None:
        device = torch.device("cpu")

    # Create base coordinate tensors
    raw_frame_indices = torch.arange(num_frames, device=device, dtype=torch.float32)
    raw_height_indices = torch.arange(height, device=device, dtype=torch.float32)
    raw_width_indices = torch.arange(width, device=device, dtype=torch.float32)

    # Create meshgrid for one video part
    grid_f, grid_h, grid_w = torch.meshgrid(
        raw_frame_indices,
        raw_height_indices,
        raw_width_indices,
        indexing="ij",
    )

    # Flatten to (F*H*W, 3) for one video part
    raw_coords_single_video = torch.stack(
        [
            grid_f.flatten(),
            grid_h.flatten(),
            grid_w.flatten(),
        ],
        dim=-1,
    )

    # Repeat for sequence multiplier (e.g., for IC-LoRA with reference + target)
    if sequence_multiplier > 1:
        coords_list = [raw_coords_single_video for _ in range(sequence_multiplier)]
        raw_coords_combined = torch.cat(coords_list, dim=0)
    else:
        raw_coords_combined = raw_coords_single_video

    # Expand for batch: (B, sequence_length * multiplier, 3)
    raw_video_coords_batched = raw_coords_combined.unsqueeze(0).expand(batch_size, -1, -1)

    return raw_video_coords_batched


def _normalize_latents(
    latents: Tensor,
    mean: Tensor,
    std: Tensor,
) -> Tensor:
    """Normalizes latents using mean and standard deviation across the channel dimension."""
    mean = mean.view(1, -1, 1, 1, 1).repeat(latents.shape[0], 1, 1, 1, 1).to(latents.device, latents.dtype)
    std = std.view(1, -1, 1, 1, 1).repeat(latents.shape[0], 1, 1, 1, 1).to(latents.device, latents.dtype)
    latents = (latents - mean) / std
    return latents


def _encode_prompt_t5(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: list[str],
    device: torch.device,
    dtype: torch.dtype,
    max_sequence_length: int,
) -> dict[str, Tensor]:
    """Encodes text prompts using T5 tokenizer and encoder.

    Args:
        tokenizer: T5 tokenizer
        text_encoder: T5 encoder model
        prompt: List of text prompts
        device: Target device
        dtype: Target dtype
        max_sequence_length: Maximum sequence length

    Returns:
        Dict containing prompt embeddings and attention mask
    """
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_attention_mask = text_inputs.attention_mask
    prompt_attention_mask = prompt_attention_mask.bool().to(device)

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)

    return {"prompt_embeds": prompt_embeds, "prompt_attention_mask": prompt_attention_mask}
