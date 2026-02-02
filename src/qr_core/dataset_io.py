from __future__ import annotations

from pathlib import Path
from typing import Iterator


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def list_datasets(project_root: Path) -> list[str]:
    datasets_path = project_root / "datasets"
    if not datasets_path.is_dir():
        return []
    return sorted([d.name for d in datasets_path.iterdir() if d.is_dir()])


def iter_qr_samples(project_root: Path, dataset_name: str) -> Iterator[tuple[Path, Path]]:
    base = project_root / "datasets" / dataset_name
    images_dir = base / "images" / "QR_CODE"
    markup_dir = base / "markup" / "QR_CODE"

    if not images_dir.is_dir():
        return

    for path in sorted(images_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        markup_path = markup_dir / f"{path.name}.json"
        yield path, markup_path
