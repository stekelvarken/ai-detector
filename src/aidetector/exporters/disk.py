import os
from pathlib import Path
from typing import Self

from aidetector.config import (
    Config,
    Detection,
    DetectorConfig,
    DiskConfig,
    get_date_path,
    get_timestamped_filename,
)
from aidetector.exporters.exporter import Exporter


class DiskExporter(Exporter):
    directory: Path

    def __init__(self, directory: Path, min_confidence: float):
        super().__init__(min_confidence, directory)
        self.directory = os.path.join("detections", directory)
        os.makedirs(self.directory, exist_ok=True)

    @classmethod
    def from_config(cls, config: Config, detector: DetectorConfig, exporter: DiskConfig) -> Self:
        return cls(exporter.directory, exporter.min_confidence or detector.collection.min_confidence)

    def filtered_export(self, sorted_detections: list[Detection]):
        self.logger.info(f"Saving {len(sorted_detections)} photos to disk")
        timestamp = get_date_path(sorted_detections[0], "seconds")
        timestamped_directory = os.path.join(self.directory, timestamp)
        os.makedirs(timestamped_directory, exist_ok=True)
        for result in sorted_detections:
            image_name = get_timestamped_filename(result)
            image_path = os.path.join(timestamped_directory, image_name)
            with open(image_path, "wb") as f:
                f.write(result.jpg)
