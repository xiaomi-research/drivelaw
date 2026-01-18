# ruff: noqa

# Copyright 2024 Lightricks and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import PIL.Image
import torch
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.loaders import FromSingleFileMixin, LTXVideoLoraLoaderMixin
from diffusers.models.autoencoders import AutoencoderKLLTXVideo
from diffusers.models.transformers import LTXVideoTransformer3DModel
from diffusers.pipelines.ltx.pipeline_output import LTXPipelineOutput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from transformers import T5EncoderModel, T5TokenizerFast
from torchvision.transforms.functional import center_crop, resize

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXConditionPipeline, LTXVideoCondition
        >>> from diffusers.utils import export_to_video, load_video, load_image

        >>> pipe = LTXConditionPipeline.from_pretrained("Lightricks/LTX-Video-0.9.5", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")

        >>> # Load input image and video
        >>> video = load_video(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cosmos/cosmos-video2world-input-vid.mp4"
        ... )
        >>> image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cosmos/cosmos-video2world-input.jpg"
        ... )

        >>> # Create conditioning objects
        >>> condition1 = LTXVideoCondition(
        ...     image=image,
        ...     frame_index=0,
        ... )
        >>> condition2 = LTXVideoCondition(
        ...     video=video,
        ...     frame_index=80,
        ... )

        >>> prompt = "The video depicts a long, straight highway stretching into the distance, flanked by metal guardrails. The road is divided into multiple lanes, with a few vehicles visible in the far distance. The surrounding landscape features dry, grassy fields on one side and rolling hills on the other. The sky is mostly clear with a few scattered clouds, suggesting a bright, sunny day. And then the camera switch to a winding mountain road covered in snow, with a single vehicle traveling along it. The road is flanked by steep, rocky cliffs and sparse vegetation. The landscape is characterized by rugged terrain and a river visible in the distance. The scene captures the solitude and beauty of a winter drive through a mountainous region."
        >>> negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

        >>> # Generate video
        >>> generator = torch.Generator("cuda").manual_seed(0)
        >>> # Text-only conditioning is also supported without the need to pass `conditions`
        >>> video = pipe(
        ...     conditions=[condition1, condition2],
        ...     prompt=prompt,
        ...     negative_prompt=negative_prompt,
        ...     width=768,
        ...     height=512,
        ...     num_frames=161,
        ...     num_inference_steps=40,
        ...     generator=generator,
        ... ).frames[0]

        >>> export_to_video(video, "output.mp4", fps=24)
        ```
"""


@dataclass
class LTXVideoCondition:
    """
    Defines a single frame-conditioning item for LTX Video - a single frame or a sequence of frames.

    Attributes:
        image (`PIL.Image.Image`):
            The image to condition the video on.
        video (`List[PIL.Image.Image]`):
            The video to condition the video on.
        frame_index (`int`):
            The frame index at which the image or video will conditionally effect the video generation.
        strength (`float`, defaults to `1.0`):
            The strength of the conditioning effect. A value of `1.0` means the conditioning effect is fully applied.
    """

    image: Optional[PIL.Image.Image] = None
    video: Optional[List[PIL.Image.Image]] = None
    frame_index: int = 0
    strength: float = 1.0


# from LTX-Video/ltx_video/schedulers/rf.py
def linear_quadratic_schedule(num_steps, threshold_noise=0.025, linear_steps=None):
    if linear_steps is None:
        linear_steps = num_steps // 2
    if num_steps < 2:
        return torch.tensor([1.0])
    linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
    threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
    quadratic_steps = num_steps - linear_steps
    quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps**2)
    linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps**2)
    const = quadratic_coef * (linear_steps**2)
    quadratic_sigma_schedule = [
        quadratic_coef * (i**2) + linear_coef * i + const for i in range(linear_steps, num_steps)
    ]
    sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
    sigma_schedule = [1.0 - x for x in sigma_schedule]
    return torch.tensor(sigma_schedule[:-1])


# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class LTXConditionPipeline(DiffusionPipeline, FromSingleFileMixin, LTXVideoLoraLoaderMixin):
    r"""
    Pipeline for text/image/video-to-video generation.

    Reference: https://github.com/Lightricks/LTX-Video

    Args:
        transformer ([`LTXVideoTransformer3DModel`]):
            Conditional Transformer architecture to denoise the encoded video latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLLTXVideo`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer (`T5TokenizerFast`):
            Second Tokenizer of class
            [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLLTXVideo,
        text_encoder: T5EncoderModel,
        tokenizer: T5TokenizerFast,
        transformer: LTXVideoTransformer3DModel,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.vae_spatial_compression_ratio = (
            self.vae.spatial_compression_ratio if getattr(self, "vae", None) is not None else 32
        )
        self.vae_temporal_compression_ratio = (
            self.vae.temporal_compression_ratio if getattr(self, "vae", None) is not None else 8
        )
        self.transformer_spatial_patch_size = (
            self.transformer.config.patch_size if getattr(self, "transformer", None) is not None else 1
        )
        self.transformer_temporal_patch_size = (
            self.transformer.config.patch_size_t if self.transformer is not None else 1
        )

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_compression_ratio)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if getattr(self, "tokenizer", None) is not None else 128
        )

        self.default_height = 512
        self.default_width = 704
        self.default_frames = 121

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
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

        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask)[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_videos_per_prompt, 1)

        return prompt_embeds, prompt_attention_mask

    # Copied from diffusers.pipelines.mochi.pipeline_mochi.MochiPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds, negative_prompt_attention_mask = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

    def check_inputs(
        self,
        prompt,
        conditions,
        image,
        video,
        frame_index,
        strength,
        height,
        width,
        callback_on_step_end_tensor_inputs=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
        reference_video=None,
    ):
        if height % 32 != 0 or width % 32 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 32 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt_embeds is not None and prompt_attention_mask is None:
            raise ValueError("Must provide `prompt_attention_mask` when specifying `prompt_embeds`.")

        if negative_prompt_embeds is not None and negative_prompt_attention_mask is None:
            raise ValueError("Must provide `negative_prompt_attention_mask` when specifying `negative_prompt_embeds`.")

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
            if prompt_attention_mask.shape != negative_prompt_attention_mask.shape:
                raise ValueError(
                    "`prompt_attention_mask` and `negative_prompt_attention_mask` must have the same shape when passed directly, but"
                    f" got: `prompt_attention_mask` {prompt_attention_mask.shape} != `negative_prompt_attention_mask`"
                    f" {negative_prompt_attention_mask.shape}."
                )

        if conditions is not None and (image is not None or video is not None):
            raise ValueError("If `conditions` is provided, `image` and `video` must not be provided.")

        if conditions is None:
            if isinstance(image, list) and isinstance(frame_index, list) and len(image) != len(frame_index):
                raise ValueError(
                    "If `conditions` is not provided, `image` and `frame_index` must be of the same length."
                )
            elif isinstance(image, list) and isinstance(strength, list) and len(image) != len(strength):
                raise ValueError("If `conditions` is not provided, `image` and `strength` must be of the same length.")
            elif isinstance(video, list) and isinstance(frame_index, list) and len(video) != len(frame_index):
                raise ValueError(
                    "If `conditions` is not provided, `video` and `frame_index` must be of the same length."
                )
            elif isinstance(video, list) and isinstance(strength, list) and len(video) != len(strength):
                raise ValueError("If `conditions` is not provided, `video` and `strength` must be of the same length.")

        if reference_video is not None:
            if not isinstance(reference_video, torch.Tensor):
                raise ValueError(
                    "`reference_video` must be a torch.Tensor with shape [F, C, H, W] as returned by read_video()."
                )
            if reference_video.ndim != 4:
                raise ValueError(
                    f"`reference_video` must be a 4D tensor with shape [F, C, H, W], but got shape {reference_video.shape}."
                )

    @staticmethod
    def _prepare_video_ids(
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        patch_size: int = 1,
        patch_size_t: int = 1,
        device: torch.device = None,
    ) -> torch.Tensor:
        latent_sample_coords = torch.meshgrid(
            torch.arange(0, num_frames, patch_size_t, device=device),
            torch.arange(0, height, patch_size, device=device),
            torch.arange(0, width, patch_size, device=device),
            indexing="ij",
        )
        latent_sample_coords = torch.stack(latent_sample_coords, dim=0)
        latent_coords = latent_sample_coords.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        latent_coords = latent_coords.reshape(batch_size, -1, num_frames * height * width)

        return latent_coords

    @staticmethod
    def _scale_video_ids(
        video_ids: torch.Tensor,
        scale_factor: int = 32,
        scale_factor_t: int = 8,
        frame_index: int = 0,
        device: torch.device = None,
    ) -> torch.Tensor:
        scaled_latent_coords = (
            video_ids
            * torch.tensor([scale_factor_t, scale_factor, scale_factor], device=video_ids.device)[None, :, None]
        )
        scaled_latent_coords[:, 0] = (scaled_latent_coords[:, 0] + 1 - scale_factor_t).clamp(min=0)
        scaled_latent_coords[:, 0] += frame_index

        return scaled_latent_coords

    @staticmethod
    # Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline._pack_latents
    def _pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
        # Unpacked latents of shape are [B, C, F, H, W] are patched into tokens of shape [B, C, F // p_t, p_t, H // p, p, W // p, p].
        # The patch dimensions are then permuted and collapsed into the channel dimension of shape:
        # [B, F // p_t * H // p * W // p, C * p_t * p * p] (an ndim=3 tensor).
        # dim=0 is the batch size, dim=1 is the effective video sequence length, dim=2 is the effective number of input features
        batch_size, num_channels, num_frames, height, width = latents.shape
        post_patch_num_frames = num_frames // patch_size_t
        post_patch_height = height // patch_size
        post_patch_width = width // patch_size
        latents = latents.reshape(
            batch_size,
            -1,
            post_patch_num_frames,
            patch_size_t,
            post_patch_height,
            patch_size,
            post_patch_width,
            patch_size,
        )
        latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline._unpack_latents
    def _unpack_latents(
        latents: torch.Tensor, num_frames: int, height: int, width: int, patch_size: int = 1, patch_size_t: int = 1
    ) -> torch.Tensor:
        # Packed latents of shape [B, S, D] (S is the effective video sequence length, D is the effective feature dimensions)
        # are unpacked and reshaped into a video tensor of shape [B, C, F, H, W]. This is the inverse operation of
        # what happens in the `_pack_latents` method.
        batch_size = latents.size(0)
        latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
        latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline._normalize_latents
    def _normalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
    ) -> torch.Tensor:
        # Normalize latents across the channel dimension [B, C, F, H, W]
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = (latents - latents_mean) * scaling_factor / latents_std
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline._denormalize_latents
    def _denormalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
    ) -> torch.Tensor:
        # Denormalize latents across the channel dimension [B, C, F, H, W]
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents * latents_std / scaling_factor + latents_mean
        return latents

    def trim_conditioning_sequence(self, start_frame: int, sequence_num_frames: int, target_num_frames: int):
        """
        Trim a conditioning sequence to the allowed number of frames.

        Args:
            start_frame (int): The target frame number of the first frame in the sequence.
            sequence_num_frames (int): The number of frames in the sequence.
            target_num_frames (int): The target number of frames in the generated video.
        Returns:
            int: updated sequence length
        """
        scale_factor = self.vae_temporal_compression_ratio
        num_frames = min(sequence_num_frames, target_num_frames - start_frame)
        # Trim down to a multiple of temporal_scale_factor frames plus 1
        num_frames = (num_frames - 1) // scale_factor * scale_factor + 1
        return num_frames

    @staticmethod
    def add_noise_to_image_conditioning_latents(
        t: float,
        init_latents: torch.Tensor,
        latents: torch.Tensor,
        noise_scale: float,
        conditioning_mask: torch.Tensor,
        generator,
        eps=1e-6,
    ):
        """
        Add timestep-dependent noise to the hard-conditioning latents. This helps with motion continuity, especially
        when conditioned on a single frame.
        """
        noise = randn_tensor(
            latents.shape,
            generator=generator,
            device=latents.device,
            dtype=latents.dtype,
        )
        # Add noise only to hard-conditioning latents (conditioning_mask = 1.0)
        need_to_noise = (conditioning_mask > 1.0 - eps).unsqueeze(-1)
        noised_latents = init_latents + noise_scale * noise * (t**2)
        latents = torch.where(need_to_noise, noised_latents, latents)
        return latents

    def prepare_latents(
        self,
        conditions: Optional[List[torch.Tensor]] = None,
        condition_strength: Optional[List[float]] = None,
        condition_frame_index: Optional[List[int]] = None,
        batch_size: int = 1,
        num_channels_latents: int = 128,
        height: int = 512,
        width: int = 704,
        num_frames: int = 161,
        num_prefix_latent_frames: int = 2,
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        num_latent_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        if len(conditions) > 0:
            condition_latent_frames_mask = torch.zeros(
                (batch_size, num_latent_frames), device=device, dtype=torch.float32
            )

            extra_conditioning_latents = []
            extra_conditioning_video_ids = []
            extra_conditioning_mask = []
            extra_conditioning_num_latents = 0
            for data, strength, frame_index in zip(conditions, condition_strength, condition_frame_index, strict=False):
                condition_latents = retrieve_latents(self.vae.encode(data), generator=generator)
                condition_latents = self._normalize_latents(
                    condition_latents, self.vae.latents_mean, self.vae.latents_std
                ).to(device, dtype=dtype)

                num_data_frames = data.size(2)
                num_cond_frames = condition_latents.size(2)

                if frame_index == 0:
                    latents[:, :, :num_cond_frames] = torch.lerp(
                        latents[:, :, :num_cond_frames], condition_latents, strength
                    )
                    condition_latent_frames_mask[:, :num_cond_frames] = strength

                else:
                    if num_data_frames > 1:
                        if num_cond_frames < num_prefix_latent_frames:
                            raise ValueError(
                                f"Number of latent frames must be at least {num_prefix_latent_frames} but got {num_data_frames}."
                            )

                        if num_cond_frames > num_prefix_latent_frames:
                            start_frame = frame_index // self.vae_temporal_compression_ratio + num_prefix_latent_frames
                            end_frame = start_frame + num_cond_frames - num_prefix_latent_frames
                            latents[:, :, start_frame:end_frame] = torch.lerp(
                                latents[:, :, start_frame:end_frame],
                                condition_latents[:, :, num_prefix_latent_frames:],
                                strength,
                            )
                            condition_latent_frames_mask[:, start_frame:end_frame] = strength
                            condition_latents = condition_latents[:, :, :num_prefix_latent_frames]

                    noise = randn_tensor(condition_latents.shape, generator=generator, device=device, dtype=dtype)
                    condition_latents = torch.lerp(noise, condition_latents, strength)

                    condition_video_ids = self._prepare_video_ids(
                        batch_size,
                        condition_latents.size(2),
                        latent_height,
                        latent_width,
                        patch_size=self.transformer_spatial_patch_size,
                        patch_size_t=self.transformer_temporal_patch_size,
                        device=device,
                    )
                    condition_video_ids = self._scale_video_ids(
                        condition_video_ids,
                        scale_factor=self.vae_spatial_compression_ratio,
                        scale_factor_t=self.vae_temporal_compression_ratio,
                        frame_index=frame_index,
                        device=device,
                    )
                    condition_latents = self._pack_latents(
                        condition_latents,
                        self.transformer_spatial_patch_size,
                        self.transformer_temporal_patch_size,
                    )
                    condition_conditioning_mask = torch.full(
                        condition_latents.shape[:2], strength, device=device, dtype=dtype
                    )

                    extra_conditioning_latents.append(condition_latents)
                    extra_conditioning_video_ids.append(condition_video_ids)
                    extra_conditioning_mask.append(condition_conditioning_mask)
                    extra_conditioning_num_latents += condition_latents.size(1)

        video_ids = self._prepare_video_ids(
            batch_size,
            num_latent_frames,
            latent_height,
            latent_width,
            patch_size_t=self.transformer_temporal_patch_size,
            patch_size=self.transformer_spatial_patch_size,
            device=device,
        )
        if len(conditions) > 0:
            conditioning_mask = condition_latent_frames_mask.gather(1, video_ids[:, 0])
        else:
            conditioning_mask, extra_conditioning_num_latents = None, 0
        video_ids = self._scale_video_ids(
            video_ids,
            scale_factor=self.vae_spatial_compression_ratio,
            scale_factor_t=self.vae_temporal_compression_ratio,
            frame_index=0,
            device=device,
        )
        latents = self._pack_latents(latents, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size)

        if len(conditions) > 0 and len(extra_conditioning_latents) > 0:
            latents = torch.cat([*extra_conditioning_latents, latents], dim=1)
            video_ids = torch.cat([*extra_conditioning_video_ids, video_ids], dim=2)
            conditioning_mask = torch.cat([*extra_conditioning_mask, conditioning_mask], dim=1)

        return latents, conditioning_mask, video_ids, extra_conditioning_num_latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        conditions: Union[LTXVideoCondition, List[LTXVideoCondition]] = None,
        image: Union[PipelineImageInput, List[PipelineImageInput]] = None,
        video: List[PipelineImageInput] = None,
        frame_index: Union[int, List[int]] = 0,
        strength: Union[float, List[float]] = 1.0,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 512,
        width: int = 704,
        num_frames: int = 161,
        frame_rate: int = 25,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 3,
        image_cond_noise_scale: float = 0.15,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        reference_video: Optional[torch.Tensor] = None,
        output_reference_comparison: bool = False,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        decode_timestep: Union[float, List[float]] = 0.0,
        decode_noise_scale: Optional[Union[float, List[float]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            conditions (`List[LTXVideoCondition], *optional*`):
                The list of frame-conditioning items for the video generation.If not provided, conditions will be
                created using `image`, `video`, `frame_index` and `strength`.
            image (`PipelineImageInput` or `List[PipelineImageInput]`, *optional*):
                The image or images to condition the video generation. If not provided, one has to pass `video` or
                `conditions`.
            video (`List[PipelineImageInput]`, *optional*):
                The video to condition the video generation. If not provided, one has to pass `image` or `conditions`.
            frame_index (`int` or `List[int]`, *optional*):
                The frame index or frame indices at which the image or video will conditionally effect the video
                generation. If not provided, one has to pass `conditions`.
            strength (`float` or `List[float]`, *optional*):
                The strength or strengths of the conditioning effect. If not provided, one has to pass `conditions`.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, defaults to `512`):
                The height in pixels of the generated image. This is set to 480 by default for the best results.
            width (`int`, defaults to `704`):
                The width in pixels of the generated image. This is set to 848 by default for the best results.
            num_frames (`int`, defaults to `161`):
                The number of video frames to generate
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, defaults to `3 `):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            reference_video (`torch.Tensor`, *optional*):
                An optional reference video to guide the generation process. Should be a tensor with shape
                [F, C, H, W] in range [0, 1] as returned by `read_video()` from video_utils. The reference video
                will be encoded and concatenated to the latent sequence, providing global guidance while remaining
                unchanged during denoising. The reference video can be of any size and will be automatically
                resized and cropped to match the target dimensions.
            output_reference_comparison (`bool`, defaults to `False`):
                Whether to output a side-by-side comparison showing both the reference video (if provided) and the
                generated video. If `False`, only the generated video is returned. Only applies when `reference_video`
                is provided.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_attention_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask for text embeddings.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            negative_prompt_attention_mask (`torch.FloatTensor`, *optional*):
                Pre-generated attention mask for negative text embeddings.
            decode_timestep (`float`, defaults to `0.0`):
                The timestep at which generated video is decoded.
            decode_noise_scale (`float`, defaults to `None`):
                The interpolation factor between random noise and denoised latents at the decode timestep.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ltx.LTXPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to `128 `):
                Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.ltx.LTXPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ltx.LTXPipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        if latents is not None:
            raise ValueError("Passing latents is not yet supported.")

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt=prompt,
            conditions=conditions,
            image=image,
            video=video,
            frame_index=frame_index,
            strength=strength,
            height=height,
            width=width,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            reference_video=reference_video,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False
        self._current_timestep = None

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if conditions is not None:
            if not isinstance(conditions, list):
                conditions = [conditions]

            strength = [condition.strength for condition in conditions]
            frame_index = [condition.frame_index for condition in conditions]
            image = [condition.image for condition in conditions]
            video = [condition.video for condition in conditions]
        elif image is not None or video is not None:
            if not isinstance(image, list):
                image = [image]
                num_conditions = 1
            elif isinstance(image, list):
                num_conditions = len(image)
            if not isinstance(video, list):
                video = [video]
                num_conditions = 1
            elif isinstance(video, list):
                num_conditions = len(video)

            if not isinstance(frame_index, list):
                frame_index = [frame_index] * num_conditions
            if not isinstance(strength, list):
                strength = [strength] * num_conditions

        device = self._execution_device

        # 3. Prepare text embeddings
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        vae_dtype = self.vae.dtype

        conditioning_tensors = []
        is_conditioning_image_or_video = image is not None or video is not None
        if is_conditioning_image_or_video:
            for condition_image, condition_video, condition_frame_index, condition_strength in zip(
                image, video, frame_index, strength, strict=False
            ):
                if condition_image is not None:
                    condition_tensor = (
                        self.video_processor.preprocess(condition_image, height, width)
                        .unsqueeze(2)
                        .to(device, dtype=vae_dtype)
                    )
                elif condition_video is not None:
                    condition_tensor = self.video_processor.preprocess_video(condition_video, height, width)
                    num_frames_input = condition_tensor.size(2)
                    num_frames_output = self.trim_conditioning_sequence(
                        condition_frame_index, num_frames_input, num_frames
                    )
                    condition_tensor = condition_tensor[:, :, :num_frames_output]
                    condition_tensor = condition_tensor.to(device, dtype=vae_dtype)
                else:
                    raise ValueError("Either `image` or `video` must be provided for conditioning.")

                if condition_tensor.size(2) % self.vae_temporal_compression_ratio != 1:
                    raise ValueError(
                        f"Number of frames in the video must be of the form (k * {self.vae_temporal_compression_ratio} + 1) "
                        f"but got {condition_tensor.size(2)} frames."
                    )
                conditioning_tensors.append(condition_tensor)

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents, conditioning_mask, video_coords, extra_conditioning_num_latents = self.prepare_latents(
            conditioning_tensors,
            strength,
            frame_index,
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )

        # 4.5. Process reference video (if provided) and concatenate at the beginning
        reference_latents = None
        reference_num_latents = 0
        if reference_video is not None:
            # Work with the original tensor format [F, C, H, W]
            ref_frames = reference_video  # [F, C, H, W]

            # Resize maintaining aspect ratio (resize all frames)
            current_height, current_width = ref_frames.shape[2:]
            aspect_ratio = current_width / current_height
            target_aspect_ratio = width / height

            if aspect_ratio > target_aspect_ratio:
                # Width is relatively larger, resize based on height
                resize_height = height
                resize_width = int(resize_height * aspect_ratio)
            else:
                # Height is relatively larger, resize based on width
                resize_width = width
                resize_height = int(resize_width / aspect_ratio)

            ref_frames = resize(ref_frames, [resize_height, resize_width], antialias=True)

            # Center crop to target dimensions
            ref_frames = center_crop(ref_frames, [height, width])

            # Convert to VAE input format: [1, C, F, H, W] and proper range [-1, 1]
            reference_tensor = ref_frames.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, F, C, H, W] -> [1, C, F, H, W]
            reference_tensor = reference_tensor * 2.0 - 1.0  # [0, 1] -> [-1, 1]

            # Trim reference video to proper frame count for temporal compression
            ref_num_frames_input = reference_tensor.size(2)
            ref_num_frames_output = self.trim_conditioning_sequence(0, ref_num_frames_input, num_frames)
            reference_tensor = reference_tensor[:, :, :ref_num_frames_output]
            reference_tensor = reference_tensor.to(device, dtype=vae_dtype)

            # Ensure proper frame count for VAE temporal compression
            if reference_tensor.size(2) % self.vae_temporal_compression_ratio != 1:
                # Trim to make it compatible with temporal compression
                ref_frames_to_keep = (
                    (reference_tensor.size(2) - 1) // self.vae_temporal_compression_ratio
                ) * self.vae_temporal_compression_ratio + 1
                reference_tensor = reference_tensor[:, :, :ref_frames_to_keep]

            # Expand reference tensor for batch and num_videos_per_prompt
            reference_tensor = reference_tensor.repeat(batch_size * num_videos_per_prompt, 1, 1, 1, 1)

            # Encode reference video to latents
            reference_latents = retrieve_latents(self.vae.encode(reference_tensor), generator=generator)
            reference_latents = self._normalize_latents(
                reference_latents, self.vae.latents_mean, self.vae.latents_std
            ).to(device, dtype=torch.float32)

            # Create "clean" coordinates for reference video (as if no frame conditioning applied)
            ref_latent_frames = reference_latents.size(2)
            ref_latent_height = reference_latents.size(3)
            ref_latent_width = reference_latents.size(4)

            reference_video_coords = self._prepare_video_ids(
                batch_size * num_videos_per_prompt,
                ref_latent_frames,
                ref_latent_height,
                ref_latent_width,
                patch_size_t=self.transformer_temporal_patch_size,
                patch_size=self.transformer_spatial_patch_size,
                device=device,
            )
            reference_video_coords = self._scale_video_ids(
                reference_video_coords,
                scale_factor=self.vae_spatial_compression_ratio,
                scale_factor_t=self.vae_temporal_compression_ratio,
                frame_index=0,  # Reference video starts at frame 0
                device=device,
            )

            # Pack reference latents
            reference_latents = self._pack_latents(
                reference_latents,
                self.transformer_spatial_patch_size,
                self.transformer_temporal_patch_size,
            )
            reference_num_latents = reference_latents.size(1)

            # Concatenate reference latents at the beginning: [reference_latents, frame_conditions, target_latents]
            latents = torch.cat([reference_latents, latents], dim=1)

            # Update video coordinates: [reference_coords, existing_coords]
            reference_video_coords = reference_video_coords.float()
            video_coords = torch.cat([reference_video_coords, video_coords], dim=2)
            video_coords[:, 0] = video_coords[:, 0] * (1.0 / frame_rate)

            # Update conditioning mask to include reference (frozen = strength 1.0)
            if conditioning_mask is not None:
                reference_conditioning_mask = torch.ones(
                    (batch_size * num_videos_per_prompt, reference_num_latents), device=device, dtype=torch.float32
                )
                conditioning_mask = torch.cat([reference_conditioning_mask, conditioning_mask], dim=1)
            else:
                # If no frame conditioning, still create mask for reference
                conditioning_mask = torch.ones(
                    (batch_size * num_videos_per_prompt, reference_num_latents), device=device, dtype=torch.float32
                )
                # Add zeros for target latents
                target_conditioning_mask = torch.zeros(
                    (batch_size * num_videos_per_prompt, latents.size(1) - reference_num_latents),
                    device=device,
                    dtype=torch.float32,
                )
                conditioning_mask = torch.cat([conditioning_mask, target_conditioning_mask], dim=1)

        video_coords = video_coords.float()
        if reference_video is None:
            video_coords[:, 0] = video_coords[:, 0] * (1.0 / frame_rate)

        init_latents = latents.clone() if is_conditioning_image_or_video or reference_video is not None else None

        if self.do_classifier_free_guidance:
            video_coords = torch.cat([video_coords, video_coords], dim=0)

        # 5. Prepare timesteps
        latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio
        sigmas = linear_quadratic_schedule(num_inference_steps)
        timesteps = sigmas * 1000
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps=timesteps,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                if image_cond_noise_scale > 0 and init_latents is not None:
                    # Add timestep-dependent noise to the hard-conditioning latents
                    # This helps with motion continuity, especially when conditioned on a single frame
                    latents = self.add_noise_to_image_conditioning_latents(
                        t / 1000.0,
                        init_latents,
                        latents,
                        image_cond_noise_scale,
                        conditioning_mask,
                        generator,
                    )

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                if is_conditioning_image_or_video or reference_video is not None:
                    conditioning_mask_model_input = (
                        torch.cat([conditioning_mask, conditioning_mask])
                        if self.do_classifier_free_guidance
                        else conditioning_mask
                    )
                latent_model_input = latent_model_input.to(prompt_embeds.dtype)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0]).unsqueeze(-1).float()
                if is_conditioning_image_or_video or reference_video is not None:
                    timestep = torch.min(timestep, (1 - conditioning_mask_model_input) * 1000.0)

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    encoder_attention_mask=prompt_attention_mask,
                    video_coords=video_coords,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    timestep, _ = timestep.chunk(2)

                denoised_latents = self.scheduler.step(
                    -noise_pred, t, latents, per_token_timesteps=timestep, return_dict=False
                )[0]
                if is_conditioning_image_or_video or reference_video is not None:
                    tokens_to_denoise_mask = (t / 1000 - 1e-6 < (1.0 - conditioning_mask)).unsqueeze(-1)
                    latents = torch.where(tokens_to_denoise_mask, denoised_latents, latents)
                else:
                    latents = denoised_latents

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        # Handle reference video output processing
        if reference_video is not None and output_reference_comparison:
            # Split latents: [reference_latents, frame_conditions, target_latents]
            reference_latents_out = latents[:, :reference_num_latents]
            remaining_latents = latents[:, reference_num_latents:]

            # Remove frame conditioning from remaining latents if needed
            if is_conditioning_image_or_video:
                target_latents_out = remaining_latents[:, extra_conditioning_num_latents:]
            else:
                target_latents_out = remaining_latents

            # Process both reference and target latents
            videos = []
            for curr_latents in [reference_latents_out, target_latents_out]:
                if output_type == "latent":
                    curr_video = curr_latents
                else:
                    curr_latents = self._unpack_latents(
                        curr_latents,
                        latent_num_frames,
                        latent_height,
                        latent_width,
                        self.transformer_spatial_patch_size,
                        self.transformer_temporal_patch_size,
                    )
                    curr_latents = self._denormalize_latents(
                        curr_latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
                    )
                    curr_latents = curr_latents.to(prompt_embeds.dtype)

                    if not self.vae.config.timestep_conditioning:
                        timestep = None
                    else:
                        noise = torch.randn(
                            curr_latents.shape, generator=generator, device=device, dtype=curr_latents.dtype
                        )
                        if not isinstance(decode_timestep, list):
                            decode_timestep = [decode_timestep] * batch_size
                        if decode_noise_scale is None:
                            decode_noise_scale = decode_timestep
                        elif not isinstance(decode_noise_scale, list):
                            decode_noise_scale = [decode_noise_scale] * batch_size

                        timestep = torch.tensor(decode_timestep, device=device, dtype=curr_latents.dtype)
                        decode_noise_scale = torch.tensor(decode_noise_scale, device=device, dtype=curr_latents.dtype)[
                            :, None, None, None, None
                        ]
                        curr_latents = (1 - decode_noise_scale) * curr_latents + decode_noise_scale * noise

                    curr_video = self.vae.decode(curr_latents, timestep, return_dict=False)[0]
                    curr_video = self.video_processor.postprocess_video(curr_video, output_type=output_type)
                videos.append(curr_video)

            # Concatenate videos side-by-side (along width dimension for visual output)
            if output_type == "latent":
                video = torch.cat(videos, dim=0)
            # For video tensors, shape is [B, C, F, H, W] or list of PIL images
            elif isinstance(videos[0], list):
                # Handle PIL images case - concatenate each frame side by side
                video = []
                for batch_idx in range(len(videos[0])):
                    combined_video = []
                    for frame_idx in range(len(videos[0][batch_idx])):
                        ref_frame = videos[0][batch_idx][frame_idx]
                        gen_frame = videos[1][batch_idx][frame_idx]
                        # Create side-by-side comparison
                        import PIL.Image

                        if isinstance(ref_frame, PIL.Image.Image) and isinstance(gen_frame, PIL.Image.Image):
                            combined_width = ref_frame.width + gen_frame.width
                            combined_height = max(ref_frame.height, gen_frame.height)
                            combined_frame = PIL.Image.new("RGB", (combined_width, combined_height))
                            combined_frame.paste(ref_frame, (0, 0))
                            combined_frame.paste(gen_frame, (ref_frame.width, 0))
                            combined_video.append(combined_frame)
                        else:
                            combined_video.append(gen_frame)  # Fallback to generated only
                    video.append(combined_video)
            else:
                # Handle tensor case - concatenate along width dimension (dim=4)
                video = torch.cat(videos, dim=4)
        else:
            # Regular processing - just remove conditioning parts and output generated video
            if reference_video is not None:
                # Remove reference latents
                latents = latents[:, reference_num_latents:]

            if is_conditioning_image_or_video:
                latents = latents[:, extra_conditioning_num_latents:]

            latents = self._unpack_latents(
                latents,
                latent_num_frames,
                latent_height,
                latent_width,
                self.transformer_spatial_patch_size,
                self.transformer_temporal_patch_size,
            )

            if output_type == "latent":
                video = latents
            else:
                latents = self._denormalize_latents(
                    latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
                )
                latents = latents.to(prompt_embeds.dtype)

                if not self.vae.config.timestep_conditioning:
                    timestep = None
                else:
                    noise = torch.randn(latents.shape, generator=generator, device=device, dtype=latents.dtype)
                    if not isinstance(decode_timestep, list):
                        decode_timestep = [decode_timestep] * batch_size
                    if decode_noise_scale is None:
                        decode_noise_scale = decode_timestep
                    elif not isinstance(decode_noise_scale, list):
                        decode_noise_scale = [decode_noise_scale] * batch_size

                    timestep = torch.tensor(decode_timestep, device=device, dtype=latents.dtype)
                    decode_noise_scale = torch.tensor(decode_noise_scale, device=device, dtype=latents.dtype)[
                        :, None, None, None, None
                    ]
                    latents = (1 - decode_noise_scale) * latents + decode_noise_scale * noise

                video = self.vae.decode(latents, timestep, return_dict=False)[0]
                video = self.video_processor.postprocess_video(video, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return LTXPipelineOutput(frames=video)
