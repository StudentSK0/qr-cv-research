# src/qr_core/module_size.py
from __future__ import annotations

import numpy as np


def get_module_size_raw(markup_data: dict) -> float | None:
    """
    Извлекает средний module_size из разметки (как у вас: 2 диапазона [x_min,x_max], [y_min,y_max]).
    """
    try:
        module_size = markup_data["props"]["barcode"]["module_size"]
        all_values = []
        for pair in module_size:
            all_values.extend(pair)
        return float(np.mean(all_values))
    except Exception:
        return None


def quantize_module_size_px(module_size_px: float | None, step_px: int) -> int | None:
    """
    1) округление до int
    2) биннинг к ближайшему кратному step_px
    3) защита от нулей: минимум 1 (или step_px при step_px>1)
    """
    if module_size_px is None:
        return None

    step_px = int(step_px)
    if step_px < 1:
        step_px = 1

    rounded = int(np.rint(module_size_px))

    if step_px == 1:
        return max(1, rounded)

    q = int(np.rint(rounded / step_px) * step_px)
    return max(step_px, q)
