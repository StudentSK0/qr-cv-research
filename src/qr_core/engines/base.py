from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from time import perf_counter
from typing import Any


class EngineError(Exception):
    """Base class for engine errors."""


class ImageReadError(EngineError):
    """Raised when an image cannot be read by the engine."""


class BaseEngine(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _decode(self, image_path: Path) -> str:
        raise NotImplementedError

    def _decode_array(self, image: Any) -> str:
        """Decode from in-memory image. Engines may override for normalization mode."""
        raise EngineError(f"Engine '{self.name}' does not support in-memory decode.")

    def decode_once(self, image_path: Path) -> tuple[str, float]:
        start = perf_counter()
        decoded = self._decode(image_path)
        end = perf_counter()
        return decoded, end - start

    def decode_array_once(self, image: Any) -> tuple[str, float]:
        start = perf_counter()
        decoded = self._decode_array(image)
        end = perf_counter()
        return decoded, end - start
