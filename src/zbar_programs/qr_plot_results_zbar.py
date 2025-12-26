import json
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from InquirerPy import inquirer

# --- настройки группировки по X ---
# 1) округляем module_size до целого числа пикселей
# 2) затем биним по шагу BIN_STEP_PX (1/2/3/5 и т.п.)
BIN_STEP_PX = 2  # поменяйте на 2/3/5, если нужно укрупнить группы


def select_json(project_root: Path):
    outputs_path = project_root / "outputs" / "zbar_json_and_graphics"
    json_files = list(outputs_path.glob("qr_experiment_data_*.json"))

    if not json_files:
        print("Нет файлов данных.")
        return None

    choices = [f.name for f in json_files]
    selected = inquirer.select(
        message="Выберите файл результатов (ZBar):",
        choices=choices,
        default=choices[0],
    ).execute()

    return outputs_path / selected


def _bin_module_sizes_px(module_sizes: np.ndarray, step_px: int) -> np.ndarray:
    """
    module_sizes: float array (как в JSON)
    step_px: шаг биннинга в пикселях (>=1)
    Возвращает int array (размер модуля после округления и биннинга).
    """
    step_px = int(step_px)
    if step_px < 1:
        step_px = 1

    module_px = np.rint(module_sizes).astype(int)
    module_bin = (np.rint(module_px / step_px) * step_px).astype(int)
    return module_bin


def plot_interactive(results, out_path: Path, dataset_name: str):
    module_sizes_raw = np.array([r["module_size"] for r in results], dtype=float)
    times = np.array([r["time"] for r in results], dtype=float)
    accuracies = np.array([r["accuracy"] for r in results], dtype=float)

    module_bins = _bin_module_sizes_px(module_sizes_raw, BIN_STEP_PX)
    uniq_m = np.sort(np.unique(module_bins))

    avg_time = []
    mean_acc = []
    n_samples = []
    n_acc_1 = []
    n_acc_0 = []

    for m in uniq_m:
        mask = module_bins == m
        t_m = times[mask]
        a_m = accuracies[mask]

        avg_time.append(float(np.mean(t_m)) if t_m.size else 0.0)
        mean_acc.append(float(np.mean(a_m)) if a_m.size else 0.0)

        n = int(a_m.size)
        n_samples.append(n)

        acc1 = int(np.sum(np.isclose(a_m, 1.0)))
        acc0 = int(np.sum(np.isclose(a_m, 0.0)))

        n_acc_1.append(acc1)
        n_acc_0.append(acc0)

    avg_time = np.array(avg_time, dtype=float)
    mean_acc = np.array(mean_acc, dtype=float)

    if len(avg_time) >= 5:
        avg_time_smooth = gaussian_filter1d(avg_time, sigma=2)
    else:
        avg_time_smooth = avg_time

    max_time = float(np.max(times)) if times.size else 1.0
    y1_top = max_time * 1.05 if max_time > 0 else 1.0
    y1_range = [0.0, y1_top]

    max_count = int(max(n_samples)) if n_samples else 1
    y_bar_top = max_count * 1.10 if max_count > 0 else 1

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.55, 0.20, 0.25],
        specs=[
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "table"}],
        ],
    )

    fig.add_trace(
        go.Scatter(
            x=module_bins,
            y=times,
            mode="markers",
            name="Время (точки)",
            marker=dict(color="blue", opacity=0.30),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=uniq_m,
            y=avg_time_smooth,
            mode="lines",
            name="Тренд времени",
            line=dict(color="navy", width=3),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=uniq_m,
            y=n_acc_1,
            name="Точность = 1 (кол-во)",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=uniq_m,
            y=n_acc_0,
            name="Точность = 0 (кол-во)",
        ),
        row=2,
        col=1,
    )

    mean_acc_pct = (mean_acc * 100.0).round(2)

    fig.add_trace(
        go.Table(
            header=dict(
                values=[
                    "Размер модуля (px)",
                    "N изображений",
                    "Средняя точность",
                    "Точность = 1",
                    "Точность = 0",
                ],
                fill_color="#F2F2F2",
                align="center",
                font=dict(size=14, color="black"),
            ),
            cells=dict(
                values=[
                    [int(x) for x in uniq_m],
                    n_samples,
                    [f"{p:.2f}%" for p in mean_acc_pct],
                    n_acc_1,
                    n_acc_0,
                ],
                align=["center"] * 5,
                font=dict(size=13),
                height=26,
            ),
        ),
        row=3,
        col=1,
    )

    step_note = f", bin step = {BIN_STEP_PX}px" if BIN_STEP_PX != 1 else ""
    fig.update_layout(
        title=f"ZBar — влияние размера модуля на скорость и точность ({dataset_name})",
        title_font_size=24,
        template="plotly_white",
        height=1150,
        width=1600,
        barmode="group",
        legend=dict(
            x=1.02,
            y=1,
            bordercolor="gray",
            borderwidth=1,
        ),
        margin=dict(l=70, r=260, t=90, b=40),
    )

    fig.update_xaxes(
        title_text=f"Размер модуля (px){step_note}",
        type="linear",
        zeroline=True,
        row=2,
        col=1,
    )
    fig.update_xaxes(
        title_text=f"Размер модуля (px){step_note}",
        type="linear",
        zeroline=True,
        row=3,
        col=1,
    )

    fig.update_yaxes(
        title_text="Время декодирования (сек)",
        range=y1_range,
        zeroline=True,
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title_text="Количество изображений",
        range=[0, y_bar_top],
        zeroline=True,
        row=2,
        col=1,
    )

    fig.write_html(str(out_path))
    print(f"\nHTML сохранён: {out_path}")
    print(f"Группировка module_size: округление до int + биннинг шагом {BIN_STEP_PX}px")
    print(f"Уникальных значений по X после биннинга: {len(uniq_m)}")


def main():
    script = Path(__file__).resolve()
    project_root = script.parent.parent.parent
    print(f"Корень проекта: {project_root}")

    json_file = select_json(project_root)
    if not json_file:
        return

    dataset_name = json_file.stem.replace("qr_experiment_data_", "")

    with open(json_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    out_path = (
        project_root
        / "outputs"
        / "zbar_json_and_graphics"
        / f"interactive_plot_{dataset_name}.html"
    )
    plot_interactive(results, out_path, dataset_name)


if __name__ == "__main__":
    main()
