from __future__ import annotations

from typing import Dict, Type

from .base import BaseEngine
from .opencv_engine import OpenCVEngine
from .zbar_engine import ZBarEngine
from .zxing_engine import ZXingEngine


ENGINE_REGISTRY: Dict[str, Type[BaseEngine]] = {
    "opencv": OpenCVEngine,
    "zxing": ZXingEngine,
    "zbar": ZBarEngine,
}
