"""CLI interface for rl-deploy-bench using Typer.

Provides commands for model export, quantization, benchmarking,
and report generation from the command line.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="rl-deploy-bench",
    help="Cross-platform RL model deployment and performance benchmarking toolkit",
    no_args_is_help=True,
)
console = Console()


@app.command()
def info():
    """Show platform information and available backends."""
    from .utils.platform import detect_platform, get_monitor_backend

    info = detect_platform()
    backend = get_monitor_backend(info)

    table = Table(title="Platform Information")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("OS", info.os)
    table.add_row("Architecture", info.arch)
    table.add_row("Python", info.python_version)
    table.add_row("CPU Cores", str(info.cpu_count))
    table.add_row("Total Memory", f"{info.total_memory_gb} GB")
    table.add_row("NVIDIA GPU", f"{info.gpu_name} (x{info.gpu_count})" if info.has_nvidia_gpu else "No")
    table.add_row("Jetson", "Yes" if info.is_jetson else "No")
    table.add_row("Monitor Backend", backend)

    # Check available ONNX Runtime providers
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        table.add_row("ONNX Providers", ", ".join(providers))
    except ImportError:
        table.add_row("ONNX Providers", "onnxruntime not installed")

    console.print(table)


@app.command()
def export(
    model_path: str = typer.Argument(..., help="Path to SB3 model zip file"),
    output: str = typer.Option(..., "--output", "-o", help="Output ONNX file path"),
    algo: Optional[str] = typer.Option(None, "--algo", "-a", help="Algorithm name (PPO, SAC, DQN, etc.)"),
    opset: int = typer.Option(17, "--opset", help="ONNX opset version"),
):
    """Export an SB3 model to ONNX format."""
    from .exporter.sb3 import export_sb3_model, load_sb3_model, verify_sb3_export

    console.print(f"[cyan]Loading model:[/cyan] {model_path}")
    model = load_sb3_model(model_path, algo=algo)
    console.print(f"[green]Model loaded successfully[/green]")

    console.print(f"[cyan]Exporting to ONNX...[/cyan]")
    from .exporter.onnx_export import ExportConfig

    config = ExportConfig(opset_version=opset)
    onnx_path = export_sb3_model(model, output, config=config)
    console.print(f"[green]Exported to:[/green] {onnx_path}")

    console.print("[cyan]Verifying export...[/cyan]")
    result = verify_sb3_export(onnx_path, model, num_samples=100)
    if result["passed"]:
        console.print(f"[green]Verification passed![/green] Max diff: {result['max_abs_diff']:.6f}")
    else:
        console.print(f"[yellow]Verification warning:[/yellow] Max diff {result['max_abs_diff']:.6f} exceeds tolerance")


@app.command()
def quantize(
    onnx_path: str = typer.Argument(..., help="Path to input ONNX model"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output quantized model path"),
    mode: str = typer.Option("dynamic", "--mode", "-m", help="Quantization mode: dynamic or static"),
    obs_shape: Optional[str] = typer.Option(None, "--obs-shape", help="Observation shape (e.g., '4' or '3,84,84')"),
    calibration_samples: int = typer.Option(100, "--calibration-samples", help="Number of calibration samples for static mode"),
    calibration_file: Optional[str] = typer.Option(None, "--calibration-file", help="Path to calibration dataset .npz file (from 'calibrate' command)"),
):
    """Quantize an ONNX model to INT8."""
    from .quantizer.int8 import dynamic_quantize, static_quantize, static_quantize_with_dataset, get_model_size_mb

    original_size = get_model_size_mb(onnx_path)
    console.print(f"[cyan]Original model size:[/cyan] {original_size:.2f} MB")

    if mode == "dynamic":
        console.print("[cyan]Applying dynamic INT8 quantization...[/cyan]")
        quantized_path = dynamic_quantize(onnx_path, output)
    elif mode == "static":
        if obs_shape is None and calibration_file is None:
            console.print("[red]Error:[/red] --obs-shape or --calibration-file is required for static quantization")
            raise typer.Exit(1)

        if calibration_file:
            from .benchmark.calibration import CalibrationDataset
            console.print(f"[cyan]Loading calibration dataset:[/cyan] {calibration_file}")
            dataset = CalibrationDataset.load(calibration_file)
            console.print(f"[green]Loaded {len(dataset)} calibration samples from {dataset.env_name}[/green]")
            console.print("[cyan]Applying static INT8 quantization with environment calibration...[/cyan]")
            quantized_path = static_quantize_with_dataset(onnx_path, dataset, output, input_name="observation")
        else:
            shape = tuple(int(x) for x in obs_shape.split(","))
            console.print(f"[cyan]Applying static INT8 quantization (calibration: {calibration_samples} random samples)...[/cyan]")
            console.print("[yellow]Tip: Use 'rl-deploy-bench calibrate' to generate environment-based calibration data for better results[/yellow]")
            quantized_path = static_quantize(onnx_path, shape, output, calibration_samples=calibration_samples)
    else:
        console.print(f"[red]Error:[/red] Unknown mode '{mode}'. Use 'dynamic' or 'static'.")
        raise typer.Exit(1)

    quantized_size = get_model_size_mb(quantized_path)
    reduction = original_size - quantized_size
    reduction_pct = (reduction / original_size * 100) if original_size > 0 else 0

    console.print(f"[green]Quantized model:[/green] {quantized_path}")
    console.print(f"[green]Size:[/green] {quantized_size:.2f} MB (reduced {reduction:.2f} MB, {reduction_pct:.1f}%)")


@app.command()
def calibrate(
    env_name: str = typer.Argument(..., help="Gymnasium environment name (e.g., 'Pendulum-v1', 'CartPole-v1')"),
    output: str = typer.Option(..., "--output", "-o", help="Output calibration dataset .npz file path"),
    num_samples: int = typer.Option(500, "--num-samples", "-n", help="Number of calibration samples to collect"),
    strategy: str = typer.Option("random", "--strategy", "-s", help="Collection strategy: random, policy, or mixed"),
    sb3_model: Optional[str] = typer.Option(None, "--sb3-model", help="Path to SB3 model zip for policy-guided collection"),
    algo: Optional[str] = typer.Option(None, "--algo", help="SB3 algorithm name (PPO, SAC, etc.)"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
):
    """Generate calibration data from a Gymnasium environment for static quantization.

    Environment-based calibration produces much better quantization results than
    random noise because the calibration data matches the actual observation
    distribution the model will see during deployment.
    """
    from .benchmark.calibration import (
        EnvironmentCalibrationGenerator,
        SB3PolicyCalibrationGenerator,
        CalibrationConfig,
    )

    config = CalibrationConfig(
        num_samples=num_samples,
        collection_strategy=strategy,
        seed=seed,
    )

    if sb3_model:
        from .exporter.sb3 import load_sb3_model
        console.print(f"[cyan]Loading SB3 model:[/cyan] {sb3_model}")
        model = load_sb3_model(sb3_model, algo=algo)
        console.print("[green]SB3 model loaded, using policy-guided collection[/green]")
        generator = SB3PolicyCalibrationGenerator(env_name, model, config=config)
    else:
        if strategy != "random":
            console.print(f"[yellow]Warning:[/yellow] No SB3 model provided, falling back to random strategy")
            config.collection_strategy = "random"
        generator = EnvironmentCalibrationGenerator(env_name, config=config)

    console.print(f"[cyan]Collecting {num_samples} samples from {env_name} (strategy: {config.collection_strategy})...[/cyan]")
    dataset = generator.generate()

    stats = dataset.get_statistics()
    console.print(f"[green]Collected {len(dataset)} samples in {dataset.collection_stats['episodes_completed']} episodes[/green]")
    console.print(f"  Observation mean: {stats['mean']:.4f}, std: {stats['std']:.4f}")
    console.print(f"  Observation range: [{stats['min']:.4f}, {stats['max']:.4f}]")

    saved_path = dataset.save(output)
    console.print(f"[green]Calibration dataset saved:[/green] {saved_path}")
    console.print(f"\n[cyan]Next steps:[/cyan]")
    console.print(f"  rl-deploy-bench quantize model.onnx --mode static --calibration-file {saved_path}")


@app.command()
def benchmark(
    onnx_path: str = typer.Argument(..., help="Path to ONNX model"),
    obs_shape: str = typer.Option(..., "--obs-shape", help="Observation shape (e.g., '4' or '3,84,84')"),
    num_runs: int = typer.Option(500, "--num-runs", help="Number of benchmark runs"),
    num_warmup: int = typer.Option(50, "--num-warmup", help="Number of warmup runs"),
    batch_size: int = typer.Option(1, "--batch-size", help="Batch size"),
    monitor: bool = typer.Option(True, "--monitor/--no-monitor", help="Collect system metrics during benchmark"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output report path (Markdown or HTML)"),
):
    """Run latency and throughput benchmark on an ONNX model."""
    import numpy as np

    from .benchmark.latency import benchmark_latency
    from .monitor import create_monitor
    from .reporter.markdown import generate_markdown_report
    from .runtime.onnx_runtime import OnnxRuntimeInference
    from .utils.platform import detect_platform

    shape = tuple(int(x) for x in obs_shape.split(","))
    console.print(f"[cyan]Loading model:[/cyan] {onnx_path}")

    inference = OnnxRuntimeInference(onnx_path)
    console.print(f"[green]Active providers:[/green] {', '.join(inference.session.get_providers())}")

    # Create monitor if requested
    mon = None
    if monitor:
        try:
            mon = create_monitor()
            console.print("[green]System monitor enabled[/green]")
        except Exception as e:
            console.print(f"[yellow]Monitor disabled:[/yellow] {e}")
            mon = None

    console.print(f"[cyan]Running benchmark ({num_runs} runs, batch size {batch_size})...[/cyan]")
    result = benchmark_latency(
        inference,
        observation_shape=shape,
        num_warmup=num_warmup,
        num_runs=num_runs,
        batch_size=batch_size,
        monitor=mon,
    )

    # Print results
    lat = result.latency
    table = Table(title="Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Mean Latency", f"{lat.mean_ms:.3f} ms")
    table.add_row("P50 Latency", f"{lat.p50_ms:.3f} ms")
    table.add_row("P90 Latency", f"{lat.p90_ms:.3f} ms")
    table.add_row("P95 Latency", f"{lat.p95_ms:.3f} ms")
    table.add_row("P99 Latency", f"{lat.p99_ms:.3f} ms")
    table.add_row("Min Latency", f"{lat.min_ms:.3f} ms")
    table.add_row("Max Latency", f"{lat.max_ms:.3f} ms")
    table.add_row("Throughput", f"{lat.throughput_fps:.1f} FPS")
    if result.avg_gpu_utilization is not None:
        table.add_row("Avg GPU Util", f"{result.avg_gpu_utilization:.1f}%")
    if result.avg_gpu_power_w is not None:
        table.add_row("Avg GPU Power", f"{result.avg_gpu_power_w:.2f} W")
    console.print(table)

    # Generate report if output specified
    if output:
        platform_info = detect_platform()
        if output.endswith(".html"):
            from .reporter.html import generate_html_report

            report_path = generate_html_report(output, [result], ["Model"], platform_info=platform_info)
        else:
            report_path = generate_markdown_report(output, [result], ["Model"], platform_info=platform_info)
        console.print(f"[green]Report saved:[/green] {report_path}")


@app.command()
def compare(
    original: str = typer.Argument(..., help="Path to original (FP32) ONNX model"),
    quantized: str = typer.Argument(..., help="Path to quantized ONNX model"),
    obs_shape: str = typer.Option(..., "--obs-shape", help="Observation shape"),
    num_samples: int = typer.Option(1000, "--num-samples", help="Number of test samples"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output report path"),
):
    """Compare original and quantized models: latency, accuracy, and size."""
    import numpy as np

    from .benchmark.accuracy import compare_actions, generate_test_observations
    from .benchmark.latency import benchmark_latency
    from .quantizer.int8 import compare_model_sizes
    from .reporter.markdown import generate_markdown_report
    from .runtime.onnx_runtime import OnnxRuntimeInference
    from .utils.platform import detect_platform

    shape = tuple(int(x) for x in obs_shape.split(","))

    # Load models
    console.print("[cyan]Loading models...[/cyan]")
    orig_inf = OnnxRuntimeInference(original)
    quant_inf = OnnxRuntimeInference(quantized)

    # Generate test observations
    observations = generate_test_observations(shape, num_samples)

    # Get actions from both models
    console.print("[cyan]Running inference comparison...[/cyan]")
    orig_actions = []
    quant_actions = []
    for obs in observations:
        orig_actions.append(orig_inf.infer(obs).actions[0])
        quant_actions.append(quant_inf.infer(obs).actions[0])
    orig_actions = np.array(orig_actions)
    quant_actions = np.array(quant_actions)

    # Compare actions (quantized vs original as reference)
    # For accuracy comparison, we compare quantized to original
    # The compare_actions function expects original vs deployed
    # Here original=FP32, deployed=INT8
    # We need to pass original actions as reference and quantized as "deployed"
    # But compare_actions compares actions_original vs actions_deployed
    # So we pass orig_actions as "original" and quant_actions as "deployed"
    # But the function computes diff = original - deployed, which is what we want
    acc_result = compare_actions(orig_actions, quant_actions, observations)

    # Benchmark latency for both
    console.print("[cyan]Benchmarking original model...[/cyan]")
    orig_bench = benchmark_latency(orig_inf, shape, num_runs=200, num_warmup=30)
    console.print("[cyan]Benchmarking quantized model...[/cyan]")
    quant_bench = benchmark_latency(quant_inf, shape, num_runs=200, num_warmup=30)

    # Size comparison
    size_info = compare_model_sizes(original, quantized)

    # Print summary
    table = Table(title="Comparison Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Original (FP32)", style="green")
    table.add_column("Quantized (INT8)", style="yellow")
    table.add_column("Change", style="magenta")
    table.add_row("Mean Latency", f"{orig_bench.latency.mean_ms:.3f} ms", f"{quant_bench.latency.mean_ms:.3f} ms",
                  f"{(quant_bench.latency.mean_ms / orig_bench.latency.mean_ms - 1) * 100:+.1f}%")
    table.add_row("P95 Latency", f"{orig_bench.latency.p95_ms:.3f} ms", f"{quant_bench.latency.p95_ms:.3f} ms",
                  f"{(quant_bench.latency.p95_ms / orig_bench.latency.p95_ms - 1) * 100:+.1f}%")
    table.add_row("Throughput", f"{orig_bench.latency.throughput_fps:.1f} FPS", f"{quant_bench.latency.throughput_fps:.1f} FPS",
                  f"{(quant_bench.latency.throughput_fps / orig_bench.latency.throughput_fps - 1) * 100:+.1f}%")
    table.add_row("Model Size", f"{size_info['original_size_mb']:.2f} MB", f"{size_info['quantized_size_mb']:.2f} MB",
                  f"-{size_info['size_reduction_pct']:.1f}%")
    table.add_row("Action MSE", "-", f"{acc_result.action_mse:.6f}", "-")
    table.add_row("Cosine Sim", "-", f"{acc_result.action_cosine_similarity:.6f}", "-")
    console.print(table)

    # Generate report
    if output:
        platform_info = detect_platform()
        report_path = generate_markdown_report(
            output,
            [orig_bench, quant_bench],
            ["Original (FP32)", "Quantized (INT8)"],
            accuracy_results=[None, acc_result],  # Original has no accuracy comparison
            model_paths=[original, quantized],
            platform_info=platform_info,
        )
        console.print(f"[green]Report saved:[/green] {report_path}")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
