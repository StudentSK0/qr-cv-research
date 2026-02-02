from __future__ import annotations

from pathlib import Path

from InquirerPy import inquirer

from .binning import build_bin_stats
from .dataset_io import list_datasets
from .engines import ENGINE_REGISTRY
from .metrics import ExperimentConfig, run_experiment, save_results_json
from .plot.plot_interactive import build_interactive_plot


def _is_project_root(path: Path) -> bool:
    return (path / "src").is_dir()


def _resolve_project_root() -> Path:
    cwd = Path.cwd()
    if _is_project_root(cwd):
        _ensure_project_dirs(cwd)
        return cwd

    fallback = Path(__file__).resolve().parents[2]
    if _is_project_root(fallback):
        _ensure_project_dirs(fallback)
        return fallback

    raise RuntimeError("Cannot locate project root. Expected src/.")


def _ensure_project_dirs(project_root: Path) -> None:
    (project_root / "datasets").mkdir(parents=True, exist_ok=True)
    (project_root / "outputs").mkdir(parents=True, exist_ok=True)


def _select_engine() -> str | None:
    choices = list(ENGINE_REGISTRY.keys())
    if not choices:
        print("No engines registered.")
        return None

    selected = inquirer.select(
        message="Select decoding engine:",
        choices=choices,
        default=choices[0],
    ).execute()
    return str(selected)


def _select_dataset(project_root: Path) -> str | None:
    datasets = list_datasets(project_root)
    if not datasets:
        print("No datasets found in ./datasets.")
        return None

    selected = inquirer.select(
        message="Select dataset:",
        choices=datasets,
        default=datasets[0],
    ).execute()
    return str(selected)


def _prompt_int(message: str, default: int) -> int:
    def _validate(value: str) -> bool | str:
        if value.isdigit() and int(value) > 0:
            return True
        return "Enter a positive integer."

    selected = inquirer.text(
        message=message,
        default=str(default),
        validate=_validate,
    ).execute()
    return int(selected)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> None:
    project_root = _resolve_project_root()

    engine_key = _select_engine()
    if not engine_key:
        return

    dataset_name = _select_dataset(project_root)
    if not dataset_name:
        return

    iterations = _prompt_int("Decode iterations (default 3):", default=3)
    bin_step_px = _prompt_int("Module size bin step in px (default 2):", default=2)

    try:
        engine = ENGINE_REGISTRY[engine_key]()
    except Exception as exc:
        print(f"Failed to initialize engine '{engine_key}': {exc}")
        return

    cfg = ExperimentConfig(
        dataset_name=dataset_name,
        decode_iterations=iterations,
        module_bin_step_px=bin_step_px,
        time_mode="time_total_min",
    )

    results, summary = run_experiment(project_root, engine, cfg)
    if not results:
        print("No results to save.")
        return

    out_json = save_results_json(project_root, engine.name, dataset_name, results)

    bin_stats = build_bin_stats(results)
    out_html = out_json.with_name(f"qr_experiment_plot_{dataset_name}_{engine.name}.html")
    build_interactive_plot(
        results,
        bin_stats,
        out_html,
        title=f"QR decoding vs module size ({dataset_name}, {engine.name})",
    )

    print("Summary")
    print(f"processed: {summary.processed}")
    print(f"skipped_no_markup: {summary.skipped_no_markup}")
    print(f"skipped_bad_module: {summary.skipped_bad_module}")
    print(f"skipped_image_read: {summary.skipped_image_read}")
    print(f"saved_json_path: {_relative(out_json, project_root)}")
    print(f"saved_html_path: {_relative(out_html, project_root)}")


if __name__ == "__main__":
    main()
