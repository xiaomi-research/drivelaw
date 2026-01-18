import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from ltxv_trainer.utils import convert_checkpoint

console = Console()

app = typer.Typer(no_args_is_help=True, help="Convert checkpoint format between Diffusers and ComfyUI formats")


@app.command()
def main(
    input_path: str = typer.Argument(..., help="Path to input safetensors file"),
    to_comfy: bool = typer.Option(
        False, "--to-comfy", help="Convert from transformer to diffusion_model prefix (ComfyUI format)"
    ),
    output_path: Optional[str] = typer.Option(
        None,
        "--output-path",
        help="Path to save converted safetensors file. If not provided, will use input filename with suffix.",
    ),
) -> None:
    input_path = Path(input_path)
    if not input_path.exists():
        console.print(f"[bold red]Error:[/bold red] Input file not found: {input_path}")
        sys.exit(1)

    if output_path:
        output_path = Path(output_path)
    else:
        # Auto-generate output path by adding suffix to input filename
        suffix = "_comfy" if to_comfy else "_diffusers"
        # Remove existing _comfy or _diffusers suffix if present
        stem = input_path.stem
        if stem.endswith(("_comfy", "_diffusers")):
            stem = stem.rsplit("_", 1)[0]
        output_path = input_path.parent / f"{stem}{suffix}{input_path.suffix}"

    console.print(f"Converting {input_path} -> {output_path}")
    console.print(f"Direction: {'Diffusers -> ComfyUI' if to_comfy else 'ComfyUI -> Diffusers'}")

    convert_checkpoint(str(input_path), str(output_path), to_comfy)
    console.print("[bold green]Conversion complete![/bold green]")


if __name__ == "__main__":
    app()
