#!/usr/bin/env python3

"""
Decode precomputed video latents back into videos using the VAE.

This script loads latent files saved during preprocessing and decodes them
back into video clips using the same VAE model.

Basic usage:
    decode_latents.py /path/to/latents/dir --output-dir /path/to/output
"""

from fractions import Fraction
from pathlib import Path

import torch
import torchvision
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from transformers.utils.logging import disable_progress_bar

from ltxv_trainer import logger
from ltxv_trainer.ltxv_utils import decode_video
from ltxv_trainer.model_loader import LtxvModelVersion, load_vae

disable_progress_bar()
console = Console()
app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Decode precomputed video latents back into videos using the VAE.",
)


class LatentsDecoder:
    def __init__(self, model_source: str, device: str = "cuda", vae_tiling: bool = False):
        """Initialize the decoder with model configuration.

        Args:
            model_source: Model source - can be a version string, HF repo, or local path
            device: Device to use for computation
            vae_tiling: Whether to enable VAE tiling for larger video resolutions
        """
        self.device = torch.device(device)
        self._load_model(model_source, vae_tiling)

    def _load_model(self, model_source: str, vae_tiling: bool) -> None:
        """Initialize and load the VAE model"""
        with console.status(f"[bold]Loading VAE model from {model_source}...", spinner="dots"):
            self.vae = load_vae(model_source, dtype=torch.bfloat16).to(self.device)

            if vae_tiling:
                self.vae.enable_tiling()

    @torch.inference_mode()
    def decode(self, latents_dir: Path, output_dir: Path, seed: int | None = None) -> None:
        """Decode all latent files in the directory recursively.

        Args:
            latents_dir: Directory containing latent files (.pt)
            output_dir: Directory to save decoded videos
            seed: Optional random seed for noise generation
        """
        # Find all .pt files recursively
        latent_files = list(latents_dir.rglob("*.pt"))

        if not latent_files:
            logger.warning(f"No .pt files found in {latents_dir}")
            return

        logger.info(f"Found {len(latent_files):,} latent files to decode")

        # Process files with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Decoding latents", total=len(latent_files))

            for latent_file in latent_files:
                # Calculate relative path to maintain directory structure
                rel_path = latent_file.relative_to(latents_dir)
                output_subdir = output_dir / rel_path.parent
                output_subdir.mkdir(parents=True, exist_ok=True)

                try:
                    self._process_file(latent_file, output_subdir, seed)
                except Exception as e:
                    logger.error(f"Error processing {latent_file}: {e}")
                    continue

                progress.advance(task)

        logger.info(f"Decoding complete! Videos saved to {output_dir}")

    def _process_file(self, latent_file: Path, output_dir: Path, seed: int | None) -> None:
        """Process a single latent file"""
        # Load the latent data
        data = torch.load(latent_file, map_location=self.device)

        # Create generator only if seed is provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

        # Decode the video
        video = decode_video(
            vae=self.vae,
            latents=data["latents"],
            num_frames=data["num_frames"],
            height=data["height"],
            width=data["width"],
            device=self.device,
            patch_size=1,  # These match the transformer's config in the pipeline
            patch_size_t=1,
            generator=generator,
        )

        video = video[0]  # Remove batch dimension

        # Convert to uint8 for saving
        video = (video * 255).round().clamp(0, 255).to(torch.uint8)
        video = video.permute(1, 2, 3, 0)  # [C,F,H,W] -> [F,H,W,C]

        # Determine output format and save
        is_image = video.shape[0] == 1
        if is_image:
            # Save as PNG for single frame
            output_path = output_dir / f"{latent_file.stem}.png"
            torchvision.utils.save_image(
                video[0].permute(2, 0, 1) / 255.0,  # [H,W,C] -> [C,H,W] and normalize
                str(output_path),
            )
        else:
            # Save as MP4 for video
            output_path = output_dir / f"{latent_file.stem}.mp4"
            fps = data.get("fps", 24)  # Use stored FPS or default to 24
            torchvision.io.write_video(
                str(output_path),
                video.cpu(),
                fps=Fraction(fps).limit_denominator(1000),
                video_codec="h264",
                options={"crf": "18"},
            )


@app.command()
def main(
    latents_dir: str = typer.Argument(
        ...,
        help="Directory containing the precomputed latent files (searched recursively)",
    ),
    output_dir: str = typer.Argument(
        ...,
        help="Directory to save the decoded videos (maintains same folder hierarchy as input)",
    ),
    device: str = typer.Option(
        default="cuda",
        help="Device to use for computation",
    ),
    model_source: str = typer.Option(
        default=str(LtxvModelVersion.latest()),
        help="Model source - can be a version string (e.g. 'LTXV_2B_0.9.5'), HF repo, or local path",
    ),
    vae_tiling: bool = typer.Option(
        default=False,
        help="Enable VAE tiling for larger video resolutions",
    ),
    seed: int = typer.Option(
        default=None,
        help="Random seed for noise generation during decoding",
    ),
) -> None:
    """Decode precomputed video latents back into videos using the VAE.

    This script recursively searches for .pt latent files in the input directory
    and decodes them to videos, maintaining the same folder hierarchy in the output.

    Examples:
        # Basic usage
        python decode_latents.py /path/to/latents --output-dir /path/to/videos

        # With VAE tiling for large videos
        python decode_latents.py /path/to/latents --output-dir /path/to/videos --vae-tiling

        # With specific model and seed
        python scripts/decode_latents.py /mnt/evad_fs/worldmodel/tianzexia/nuplan_ltx/.precomputed/latents  /mnt/evad_fs/worldmodel/tianzexia/nuplan_ltx/decode_videos --model-source /mnt/evad_fs/worldmodel/tianzexia/LTX-Video-0.9.7-dev --seed 42
    """
    latents_path = Path(latents_dir)
    output_path = Path(output_dir)

    if not latents_path.exists() or not latents_path.is_dir():
        raise typer.BadParameter(f"Latents directory does not exist: {latents_path}")

    decoder = LatentsDecoder(model_source=model_source, device=device, vae_tiling=vae_tiling)
    decoder.decode(latents_path, output_path, seed=seed)


if __name__ == "__main__":
    app()
