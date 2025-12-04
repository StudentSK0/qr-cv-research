import json
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
import plotly.graph_objects as go
from InquirerPy import inquirer


def select_json(project_root: Path):
    outputs_path = project_root / "outputs" / "open_cv_json_and_graphics"
    json_files = list(outputs_path.glob("qr_experiment_data_*.json"))

    if not json_files:
        print("Нет файлов данных.")
        return None

    choices = [f.name for f in json_files]

    selected = inquirer.select(
        message="Выберите файл результатов:",
        choices=choices,
        default=choices[0]
    ).execute()

    return outputs_path / selected


def plot_interactive(results, out_path, dataset_name):
    module_sizes = np.array([r['module_size'] for r in results])
    times = np.array([r['time'] for r in results])
    accuracies = np.array([r['accuracy'] for r in results])

    uniq_m = np.sort(np.unique(module_sizes))
    avg_time = [np.mean(times[module_sizes == m]) for m in uniq_m]
    avg_acc = [np.mean(accuracies[module_sizes == m]) for m in uniq_m]

    avg_time_smooth = gaussian_filter1d(avg_time, sigma=2)
    avg_acc_smooth = gaussian_filter1d(avg_acc, sigma=2)

    fig = go.Figure()

    # Время - точки
    fig.add_trace(go.Scatter(
        x=module_sizes, y=times,
        mode='markers',
        name='Время (точки)',
        marker=dict(color='blue', opacity=0.3),
        yaxis='y1'
    ))

    # Время - тренд
    fig.add_trace(go.Scatter(
        x=uniq_m, y=avg_time_smooth,
        mode='lines',
        name='Тренд времени',
        line=dict(color='navy', width=3),
        yaxis='y1'
    ))

    # Точность - точки
    fig.add_trace(go.Scatter(
        x=module_sizes, y=accuracies,
        mode='markers',
        name='Точность (точки)',
        marker=dict(color='red', opacity=0.3),
        yaxis='y2'
    ))

    # Точность - тренд
    fig.add_trace(go.Scatter(
        x=uniq_m, y=avg_acc_smooth,
        mode='lines',
        name='Тренд точности',
        line=dict(color='darkred', width=3, dash="dash"),
        yaxis='y2'
    ))

    fig.update_layout(
        title=f"OpenCV — влияние размера модуля на скорость и точность ({dataset_name})",
        title_font_size=24,
        template="plotly_white",
        height=900,
        width=1600,

        xaxis=dict(
            title="Размер модуля",
            type="log"
        ),
        yaxis=dict(
            title="Время декодирования (сек)",
            titlefont=dict(color="navy"),
            tickfont=dict(color="navy"),
            side="left"
        ),
        yaxis2=dict(
            title="Точность",
            titlefont=dict(color="darkred"),
            tickfont=dict(color="darkred"),
            overlaying="y",
            side="right",
            range=[0, 1.05]
        ),
        legend=dict(
            x=1.02, y=1,
            bordercolor="gray", borderwidth=1
        )
    )

    fig.write_html(str(out_path))
    print(f"\nhtml страница сохраняется: {out_path}")


def main():
    script = Path(__file__).resolve()
    project_root = script.parent.parent.parent
    print(f"Корень проекта: {project_root}")

    json_file = select_json(project_root)
    if not json_file:
        return

    dataset_name = json_file.stem.replace("qr_experiment_data_", "")

    with open(json_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    out_path = project_root / "outputs" / "open_cv_json_and_graphics" / f"interactive_plot_{dataset_name}.html"
    plot_interactive(results, out_path, dataset_name)


if __name__ == "__main__":
    main()
