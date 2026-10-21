# Round-1 fresh-agent audit — rl-deploy-bench (2026-10-22)

Independent, read-then-fix review from a clean-slate perspective on HEAD `c3f189d`.
Environment: Windows 11, Python 3.13.16, no C compiler; pure-wheel venv only.
Full pytest output: `docs/audit_round1_1022_pytest.txt`.

## Scope and method

- Read every executable source file (`exporter/`, `benchmark/`, `quantizer/`,
  `runtime/`, `reporter/`, `monitor/`, `utils/`, `cli.py`) plus tests, CI,
  README/API/CHANGELOG.
- Mutation testing: inject one logical reversal / swallowed-exception / boundary
  shift per key branch, run the targeted tests, record red/green, restore via
  `git checkout --`. See "Mutation experiments" below.
- Windows platform rule: any Windows-specific failure must be reproduced on the
  pristine baseline via `git stash` before attributing it to a change. No such
  failure occurred this round (see "Platform notes").

## Final test counts (local, Windows py3.13)

- Baseline (pristine `c3f189d`): **58 passed, 1 skipped**.
- After fix: **59 passed, 1 skipped** (the +1 is the new regression test below).
- The 1 skip is `test_sb3_onnx_matches_predict`, guarded by `importorskip` on the
  optional `[sb3]` extra, which was not installed locally. CI installs the extra
  on Linux, so CI expects **60 passed, 0 skipped** (baseline CI was 59 passed;
  +1 new test).
- CI-parity static gates on the final tree: `black --check` (31 files unchanged),
  `isort --check-only` (clean), `flake8` (clean), `compileall` (clean).

## Findings

### P2-1 (fixed) — Markdown report did not escape the platform GPU name

- **File:line:** `src/rl_deploy_bench/reporter/markdown.py:75-78` (GPU row).
- **Root cause:** every other metadata string in the Markdown reporter goes
  through `_escape_pipe` (model name, title, OS-derived fields), but
  `platform_info.gpu_name` was interpolated raw. The HTML reporter already
  `html.escape`s `gpu_name` (E2 defense), so this was an MD/HTML asymmetry: a
  GPU name containing `|` splits the platform table into extra columns, and a
  newline injects a new table row. GPU names come from the driver/NVML rather
  than being hardcoded, so this is the one platform string worth hardening.
- **Fix:** route `gpu_name` through `_escape_pipe` (pipe escaped, newline folded
  to a space), matching the existing name/title defense.
- **Red/green evidence:**
  - RED on pristine: `test_markdown_report_escapes_gpu_name_pipe_and_newline`
    failed — emitted `| GPU | RTX | 4090\nINJECTED (x1) |` (raw pipe splits the
    row, raw newline starts a new line).
  - GREEN after fix: same test passes; full suite 59 passed / 1 skipped.
- **Regression test:** `tests/test_core.py:1067`
  `TestReportGeneration::test_markdown_report_escapes_gpu_name_pipe_and_newline`
  (hermetic `SimpleNamespace` stub, no ONNX needed).

### P0 / P1

None found. No incorrect algorithm/numeric result, no pickle/`allow_pickle`
hole, no resource leak on the tested paths, no crash on valid input.

### P3 (recorded, deliberately not changed)

- **P3-1 `TorchScriptInference.warmup` hardcodes `observation_shape=(4,)`**
  (`exporter/torchscript_export.py:235`). The benchmark path always passes the
  real shape, so this is unreached in tested flows; a standalone
  `warmup()` on a non-4-dim model would feed a wrong-shaped dummy. Latent only.
- **P3-2 TorchScript auto-batch only handles `ndim == 1`**
  (`exporter/torchscript_export.py:204`). Unlike the ONNX backend (which derives
  the batch dim from `input_shape`), a multi-dim single observation (e.g. an
  image) is not auto-unsqueezed. Benchmark always feeds batched arrays.
- **P3-3 FP16 Method-2 output-cast producer lookup** (`quantizer/fp16.py`): if a
  graph output has no producer node, `producer_index = -1` and the cast is
  inserted at position 0, yielding a non-topological graph. Not reachable for the
  MLPs exercised here; guarded downstream by `onnx.checker`/ORT run.
- **P3-4 `generate_test_observations` docstring advertises `distribution="real"`
  but raises `ValueError`** (`benchmark/accuracy.py`). Doc/code mismatch only.
- **P3-5 HTML system-metrics GPU-Power trace sets `yaxis="y2"` but the layout
  never defines a second y-axis** (`reporter/html.py:138`). Cosmetic; both bars
  share the primary axis.
- **P3-6 E5 non-ASCII hardening lives on the load side** (`_safe_load_torchscript`);
  `torch.jit.save` writes to the requested path directly. Documented accurately
  in the study-site pitfall note; no code change made.
- **P3-7 TensorRT runtime is framework-only** (`runtime/tensorrt_runtime.py`):
  `infer()` returns a `(output, latency)` tuple rather than the
  `InferenceResult(.actions/.latency_ms)` seam, so it cannot drop into
  `benchmark_latency` as-is; untestable without a device/TensorRT install.
- **P3-8 Throughput derives from summed self-reported per-call latency, not
  wall-clock**; caliber already stated in D4, kept consistent across models.

## Mutation experiments (all CAUGHT = targeted tests went red)

| ID | Mutation (pristine source) | Targeted `-k` | Result |
|----|----------------------------|---------------|--------|
| M1 | `_accuracy_verdict` gate `mse_ok and cosine_ok` → `or` | accuracy_verdict / evaluate_* | CAUGHT (red) |
| M2 | `num_runs <= 0` guard → `< 0` (0 slips through) | rejects_nonpositive | CAUGHT (red) |
| M3 | dry-episode guard `>= max` → `> max` (fires on 4th, not 3rd) | guard_raises_exactly_on_third / productive_episode | CAUGHT (red) |
| M4 | remove `torch.clamp(action, low, high)` in export | clips_pre_unscale | CAUGHT (red) |
| M5 | Markdown `_escape_pipe` removed | escapes_pipe / folds_newlines / title_folds | CAUGHT (red) |
| M6 | HTML title `_html.escape` bypassed | escapes_model_name | CAUGHT (red) |
| M7 | calibration `np.load(allow_pickle=False)` → `True` | load_rejects_object_arrays | CAUGHT (red) |
| M8 | `monitor.stop()` removed from benchmark `finally` | monitor_snapshot_failure / stops_monitor_when_infer_raises | CAUGHT (red) |

Each mutation was applied as a single exact replacement, the targeted tests ran
and failed, then the file was restored with `git checkout --`; the working tree
returned to pristine before the next mutation.

## Platform notes (Windows py3.13, no compiler)

- Installed pure wheels into a fresh venv: torch (CPU), numpy, onnx,
  onnxruntime, gymnasium, psutil, plotly, typer, rich, pytest, black, isort,
  flake8. No C extension build attempted.
- The SB3 extra (`stable-baselines3`) was not installed locally; its single test
  skips via `importorskip`. This is dependency absence, **not** a platform
  failure, so no pristine `git stash` reproduction was required.
- No Windows-only failure appeared; nothing needed pristine-baseline
  reproduction. CI remains the Linux source of truth.

## Attack surface for the next adversarial reviewer

1. **TorchScript seam** (P3-1/P3-2): standalone `warmup` default shape and the
   ndim==1-only auto-batch — try a 3-D image observation and a non-4-dim model
   outside `benchmark_latency`.
2. **FP16 Method-2 on non-MLP graphs** (P3-3): multi-output / initializer-fed
   outputs may hit the `producer_index=-1` insert-at-front path.
3. **TensorRT backend** (P3-7): tuple-vs-`InferenceResult` seam mismatch; buffer
   caching and `close()` are reviewed but never executed on device.
4. **Weak equivalence check**: `verify_onnx_export` uses one random dummy input;
   `verify_sb3_export`/`verify_torchscript_export` use random samples out of env
   bounds — look for shape/batch edge cases they miss.
5. **Numerics coverage**: quantization/FP16 are exercised only on a tiny MLP;
   real-network INT8/FP16 action drift is not covered.
6. **CLI error paths** (`compare`, `benchmark`, `quantize` static branch) have no
   unit tests; exercise malformed `--obs-shape`, missing calibration file, and
   non-existent model paths.
7. **HTML reporter y2 axis** (P3-5) renders GPU power on the utilization axis —
   check scale distortion when both are plotted.
