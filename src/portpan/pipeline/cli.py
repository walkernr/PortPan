"""Command-line interface for the PortPan pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from rich.console import Console
from rich.table import Table
import typer
from portpan.geometry.transform_store import TransformStore
from portpan.localization.gemini_localizer import GeminiCoarseLocalizer
from portpan.localization.heuristic_localizer import HeuristicLocalizer
from portpan.pipeline.orchestrator import RegistrationPipeline

app = typer.Typer(help="PortPan: High-magnification digiscope registration & panoramic canvas composition.")
console = Console()


@app.command()
def register(
    location_name: str = typer.Argument(..., help="Name of location folder under data/locations/"),
    data_dir: Path = typer.Option(Path("data/locations"), help="Path to locations directory"),
    use_gemini: bool = typer.Option(False, help="Use Gemini multimodal API for coarse localization"),
    output_audit: Path = typer.Option(Path("output/audit"), help="Directory for visual audit artifacts"),
) -> None:
    """Register all digiscope assets for a location against the master base canvas."""
    loc_dir = data_dir / location_name
    if not loc_dir.exists():
        console.print(f"[red]Error:[/red] Location directory not found: {loc_dir}")
        raise typer.Exit(1)

    console.print(f"[bold cyan]Registering location:[/bold cyan] {location_name}")

    localizer = GeminiCoarseLocalizer() if use_gemini else HeuristicLocalizer()
    pipeline = RegistrationPipeline(localizer=localizer, audit_dir=output_audit)

    transforms = pipeline.process_location(loc_dir)

    # Print summary table
    table = Table(title=f"Registration Summary: {location_name}")
    table.add_column("Asset", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Type")
    table.add_column("Inliers", justify="right")
    table.add_column("Inlier %", justify="right")
    table.add_column("Confidence", justify="right")

    for name, res in transforms.transforms.items():
        status_color = "green" if res.status == "SUCCESS" else ("yellow" if res.status == "FALLBACK_AFFINE" else "red")
        table.add_row(
            name,
            f"[{status_color}]{res.status.value}[/{status_color}]",
            res.transform_type.value,
            str(res.inlier_count),
            f"{res.inlier_ratio * 100:.1f}%",
            f"{res.confidence_score * 100:.1f}%",
        )

    console.print(table)
    console.print(f"[green]Saved transforms to:[/green] {loc_dir / 'transforms.json'}")
    console.print(f"[green]Audit artifacts generated in:[/green] {output_audit}")


@app.command()
def status(
    location_name: str = typer.Argument(..., help="Name of location folder under data/locations/"),
    data_dir: Path = typer.Option(Path("data/locations"), help="Path to locations directory"),
) -> None:
    """Display registration status from existing transforms.json."""
    loc_dir = data_dir / location_name
    transforms = TransformStore.load(loc_dir)

    if transforms is None:
        console.print(f"[yellow]No transforms.json found for {location_name}.[/yellow]")
        return

    table = Table(title=f"Stored Transforms: {location_name} (Base: {transforms.base_image_name})")
    table.add_column("Asset", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Inliers", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Notes")

    for name, res in transforms.transforms.items():
        status_color = "green" if res.status == "SUCCESS" else ("yellow" if res.status == "FALLBACK_AFFINE" else "red")
        table.add_row(
            name,
            f"[{status_color}]{res.status.value}[/{status_color}]",
            str(res.inlier_count),
            f"{res.confidence_score * 100:.1f}%",
            res.notes or "",
        )

    console.print(table)


@app.command()
def animate(
    location_name: str = typer.Argument(..., help="Name of location folder under data/locations/"),
    data_dir: Path = typer.Option(Path("data/locations"), help="Path to locations directory"),
    output_video: Path | None = typer.Option(None, help="Custom output video path"),
    loop_frames: int = typer.Option(60, help="Number of frames in the seamless loop"),
    fps: float = typer.Option(30.0, help="Frames per second of the output video"),
    canvas_scale: float = typer.Option(1.0, help="Canvas super-resolution scaling factor (e.g. 1.0, 1.5, 2.0)"),
    max_dimension: int = typer.Option(3840, help="Maximum video dimension for standard 4K compatibility (0 for unconstrained)"),
) -> None:
    """Synthesize a synchronized, seamlessly looping master video with animated portholes over the base panorama."""
    from portpan.video.canvas_animator import CanvasVideoAnimator

    loc_dir = data_dir / location_name
    if not loc_dir.exists():
        console.print(f"[red]Error:[/red] Location directory not found: {loc_dir}")
        raise typer.Exit(1)

    out_path = output_video or (Path("output") / location_name / f"{location_name}_animated_loop.mp4")
    console.print(f"[bold cyan]Synthesizing animated canvas video for:[/bold cyan] {location_name}")
    console.print(f"Canvas scale: [bold]{canvas_scale}x[/bold], Loop length: [bold]{loop_frames} frames[/bold] @ [bold]{fps} fps[/bold], Max dimension: [bold]{max_dimension}[/bold]")

    animator = CanvasVideoAnimator()
    exported_path = animator.animate_location(
        location_dir=loc_dir,
        output_path=out_path,
        loop_frames=loop_frames,
        fps=fps,
        canvas_scale=canvas_scale,
        max_dimension=max_dimension if max_dimension > 0 else None,
    )

    console.print(f"[bold green]Successfully generated animated video:[/bold green] {exported_path}")


if __name__ == "__main__":
    app()
