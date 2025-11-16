import logging
import tempfile
from datetime import datetime
from threading import Thread
from typing import Self

import cv2
from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.engine.results import Results

from aidetector.config import ChatConfig, Config, Detection, DetectionConfig, DetectorConfig, DiskConfig, WebhookConfig
from aidetector.exporters.disk import DiskExporter
from aidetector.exporters.exporter import Exporter
from aidetector.exporters.telegram import TelegramExporter
from aidetector.exporters.webhook import WebhookExporter


class Detector:
    logger = logging.getLogger(__name__)
    detections: list[Detection] = []

    def __init__(
        self,
        model: str,
        sources: list[str],
        config: DetectionConfig,
        exporters: list[Exporter],
    ):
        self.config = config
        self.exporters = exporters
        self.logger.info(f"Loading model from {model}")
        self.model = YOLO(model, task="detect")

        is_file = sources[0].lower().endswith(tuple(IMG_FORMATS.union(VID_FORMATS)))
        is_stream = sources[0].isnumeric() or not is_file

        self.source = tempfile.mkstemp(suffix=".streams" if is_stream else ".txt", text=True)[1]
        with open(self.source, "w", encoding="utf-8") as f:
            f.write("\n".join(sources))

    @classmethod
    def from_config(cls, config: Config, detector: DetectorConfig) -> Self:
        exporters: list[Exporter] = []
        if detector.exporters is not None:
            telegram_obj: list[ChatConfig] | ChatConfig = detector.exporters.telegram or []
            telegram_list: list[ChatConfig] = [
                x for x in (telegram_obj if isinstance(telegram_obj, list) else [telegram_obj]) if x is not None
            ]
            for telegram_exporter in telegram_list:
                exporters.append(TelegramExporter.from_config(config, detector, telegram_exporter))

            disk_obj: list[DiskConfig] | DiskConfig = detector.exporters.disk or []
            disk_list: list[DiskConfig] = [
                x for x in (disk_obj if isinstance(disk_obj, list) else [disk_obj]) if x is not None
            ]
            for disk_exporter in disk_list:
                exporters.append(DiskExporter.from_config(config, detector, disk_exporter))
            # todo: add webhook exporter

            webhook_obj: list[WebhookConfig] | WebhookConfig = detector.exporters.webhook or []
            webhook_list: list[WebhookConfig] = [
                x for x in (webhook_obj if isinstance(webhook_obj, list) else [webhook_obj]) if x is not None
            ]
            for webhook_exporter in webhook_list:
                exporters.append(WebhookExporter.from_config(config, detector, webhook_exporter))

        return cls(detector.model, detector.sources, detector.detection, exporters)

    def start(self):
        def runner():
            results = self.model.predict(
                source=self.source,
                conf=self.config.confidence,
                stream=True,
            )
            for result in results:
                self._add_detection(result)
                self._try_export()
                self._filter_detections()

        Thread(target=runner).start()

    def _filter_detections(self):
        self.detections = [
            d for d in self.detections if (datetime.now() - d.date).total_seconds() <= self.config.time_max
        ]

    def _add_detection(self, result: Results):
        if result.boxes is not None and len(result.boxes) > 0:
            confidence = max(box.conf.item() for box in result.boxes)
            success, jpg = cv2.imencode(".jpg", result.orig_img)
            if not success:
                return

            self.detections.append(Detection(date=datetime.now(), jpg=jpg.tobytes(), confidence=confidence))

    def _try_export(self):
        now: datetime = datetime.now()
        if not self.detections or len(self.detections) < self.config.frames_min:
            return

        time_collecting = (now - self.detections[0].date).total_seconds()
        timeout = (now - self.detections[-1].date).total_seconds()

        if (time_collecting < self.config.time_max) and (timeout < self.config.timeout):
            return

        self.logger.info(
            f"Exporting collection with {len(self.detections)} detections over {time_collecting} seconds with max confidence {max(d.confidence for d in self.detections)}"
        )
        sorted_detections = sorted(self.detections, key=lambda d: d.confidence, reverse=True)

        def runner():
            for exporter in self.exporters:
                try:
                    exporter.export(sorted_detections)
                except Exception:
                    self.logger.exception(f"Exporter {exporter.__class__.__name__} failed")

        Thread(target=runner, daemon=True).start()

        self.detections = []
