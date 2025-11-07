import logging
from abc import ABC, abstractmethod
from typing import Self

from aidetector.config import Config, Detection, DetectorConfig


class Exporter(ABC):
    logger = logging.getLogger(__name__)
    min_confidence: float

    def __init__(self, min_confidence: float, *args):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing with args={args}")
        self.min_confidence = min_confidence

    @classmethod
    @abstractmethod
    def from_config(cls: Self, config: Config, detector: DetectorConfig, exporter: object) -> Self:
        pass

    def export(self, detections: list[Detection]):
        sorted_detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        filtered_detections = [d for d in sorted_detections if d.confidence >= self.min_confidence]
        if not filtered_detections:
            self.logger.info("No detections meet the minimum confidence threshold")
            return
        self.filtered_export(filtered_detections)

    @abstractmethod
    def filtered_export(self, sorted_detections: list[Detection]):
        pass
