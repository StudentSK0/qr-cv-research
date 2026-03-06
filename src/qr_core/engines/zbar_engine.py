from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BaseEngine, ImageReadError


class ZBarEngine(BaseEngine):
    def __init__(self) -> None:
        import cv2
        from pyzbar.pyzbar import decode as zbar_decode
        from pyzbar.pyzbar import ZBarSymbol

        self._cv2 = cv2
        self._zbar_decode = zbar_decode
        self._ZBarSymbol = ZBarSymbol

    @property
    def name(self) -> str:
        return "zbar"

    def _decode(self, image_path: Path) -> str:
        cv2 = self._cv2

        image = cv2.imread(str(image_path))
        if image is None:
            raise ImageReadError(f"Failed to read image: {image_path}")

        return self._decode_array(image)

    def _decode_array(self, image: Any) -> str:
        cv2 = self._cv2

        if image is None:
            raise ImageReadError("Failed to decode from empty image.")

        if len(getattr(image, "shape", ())) == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        decoded = self._zbar_decode(gray, symbols=[self._ZBarSymbol.QRCODE])
        if not decoded:
            return ""

        data = decoded[0].data
        if isinstance(data, bytes):
            data = data.decode("utf-8", errors="ignore")
        return str(data) if data else ""
