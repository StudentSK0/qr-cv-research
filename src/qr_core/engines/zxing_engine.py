from __future__ import annotations

from pathlib import Path

from .base import BaseEngine, ImageReadError


class ZXingEngine(BaseEngine):
    def __init__(self) -> None:
        from pyzxing import BarCodeReader

        self._reader = BarCodeReader()

    @property
    def name(self) -> str:
        return "zxing"

    def _decode(self, image_path: Path) -> str:
        if not image_path.is_file():
            raise ImageReadError(f"Missing image file: {image_path}")

        try:
            result = self._reader.decode(str(image_path))
        except Exception as exc:
            raise ImageReadError(f"ZXing failed to read image: {image_path}") from exc

        if not result:
            return ""

        first = result[0] if isinstance(result, list) and result else result
        if not isinstance(first, dict):
            return ""

        parsed = first.get("parsed", "")
        if isinstance(parsed, bytes):
            parsed = parsed.decode("utf-8", errors="ignore")
        return str(parsed) if parsed else ""
