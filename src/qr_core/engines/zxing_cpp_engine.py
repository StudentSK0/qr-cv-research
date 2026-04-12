from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BaseEngine, EngineError, ImageReadError


class ZXingCppEngine(BaseEngine):
    def __init__(self) -> None:
        try:
            import zxingcpp
        except Exception as exc:  # pragma: no cover - import-time environment issue
            raise EngineError(
                "zxing_cpp engine requires the 'zxing-cpp' Python package. "
                "Install it with: pip install zxing-cpp"
            ) from exc

        import cv2

        self._zxingcpp = zxingcpp
        self._cv2 = cv2

    @property
    def name(self) -> str:
        return "zxing_cpp"

    def _decode(self, image_path: Path) -> str:
        if not image_path.is_file():
            raise ImageReadError(f"Missing image file: {image_path}")

        image = self._cv2.imread(str(image_path))
        if image is None:
            raise ImageReadError(f"Failed to read image: {image_path}")

        return self._decode_array(image)

    def _decode_array(self, image: Any) -> str:
        if image is None:
            raise ImageReadError("Failed to decode from empty image.")

        zxingcpp = self._zxingcpp

        try:
            reader = getattr(zxingcpp, "read_barcodes", None)
            if callable(reader):
                results = reader(image)
            else:
                single_reader = getattr(zxingcpp, "read_barcode", None)
                if callable(single_reader):
                    single = single_reader(image)
                    results = [single] if single else []
                else:
                    raise EngineError("zxing-cpp API not found (read_barcodes/read_barcode).")
        except EngineError:
            raise
        except Exception as exc:
            raise ImageReadError("ZXing-cpp failed to decode image array.") from exc

        if not results:
            return ""

        first = results[0]
        text = getattr(first, "text", "")
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="ignore")

        if text:
            return str(text)

        parsed = getattr(first, "parsed", "")
        if isinstance(parsed, bytes):
            parsed = parsed.decode("utf-8", errors="ignore")

        return str(parsed) if parsed else ""
