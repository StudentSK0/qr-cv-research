from __future__ import annotations

from pathlib import Path

from .base import BaseEngine, ImageReadError


class OpenCVEngine(BaseEngine):
    def __init__(self) -> None:
        import cv2

        self._cv2 = cv2
        self._detector = cv2.QRCodeDetector()

    @property
    def name(self) -> str:
        return "opencv"

    def _decode(self, image_path: Path) -> str:
        cv2 = self._cv2

        image = cv2.imread(str(image_path))
        if image is None:
            raise ImageReadError(f"Failed to read image: {image_path}")

        try:
            retval, decoded_info, _, _ = self._detector.detectAndDecodeMulti(image)
        except Exception:
            retval, decoded_info = False, None

        if retval and decoded_info:
            if isinstance(decoded_info, (list, tuple)):
                for value in decoded_info:
                    if value:
                        return str(value)
                return ""
            if isinstance(decoded_info, str):
                return decoded_info or ""

        decoded_str, _, _ = self._detector.detectAndDecode(image)
        return str(decoded_str) if decoded_str else ""
