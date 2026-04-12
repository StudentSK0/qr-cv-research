from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

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

        return self._extract_parsed_value(result)

    def _decode_array(self, image: Any) -> str:
        if image is None:
            raise ImageReadError("Failed to decode from empty image.")

        try:
            import cv2
        except Exception as exc:
            raise ImageReadError("OpenCV is required for ZXing in-memory decode.") from exc

        try:
            encoded_ok, encoded = cv2.imencode(".png", image)
        except Exception as exc:
            raise ImageReadError("Failed to encode image array for ZXing.") from exc

        if not encoded_ok:
            raise ImageReadError("Failed to encode image array for ZXing.")

        temp_path: Path | None = None
        try:
            with NamedTemporaryFile(prefix="qr_zxing_", suffix=".png", delete=False) as tmp:
                tmp.write(encoded.tobytes())
                temp_path = Path(tmp.name)
            return self._decode(temp_path)
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    # Best-effort cleanup for temporary files.
                    pass

    @staticmethod
    def _extract_parsed_value(result: Any) -> str:
        if not result:
            return ""

        first = result[0] if isinstance(result, list) and result else result
        if not isinstance(first, dict):
            return ""

        parsed = first.get("parsed", "")
        if isinstance(parsed, bytes):
            parsed = parsed.decode("utf-8", errors="ignore")
        return str(parsed) if parsed else ""
