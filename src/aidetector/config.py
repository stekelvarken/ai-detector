import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic.dataclasses import dataclass


@dataclass
class Detection:
    date: datetime
    jpg: bytes
    confidence: float


def get_timestamped_filename(detection: Detection) -> str:
    rounded_confidence = round(detection.confidence, 3)
    timestamp = get_date_path(detection, "milliseconds")
    return f"{timestamp}_{rounded_confidence}.jpg"


def get_date_path(detection: Detection, timespec: Literal["seconds", "milliseconds"]) -> str:
    return detection.date.isoformat(timespec=timespec).replace(":", "-")


@dataclass
class DetectionConfig:
    confidence: float
    time_max: int = 60
    timeout: int = 0
    frames_min: int = 1


@dataclass
class ChatConfig:
    token: str
    chat: str
    confidence: float | None = None


@dataclass
class WebhookConfig:
    webhook_url: str
    webhook_token: str
    confidence: float | None = None


@dataclass
class DiskConfig:
    directory: Path
    confidence: float | None = None


@dataclass
class ExportersConfig:
    disk: DiskConfig | list[DiskConfig] | None = None
    telegram: ChatConfig | list[ChatConfig] | None = None
    webhook: WebhookConfig | list[WebhookConfig] | None = None


@dataclass
class DetectorConfig:
    detection: DetectionConfig
    model: str
    sources: list[str]
    exporters: ExportersConfig | None = None


@dataclass
class Config:
    detectors: list[DetectorConfig]


config_json = json.load(open("config.json"))
if config_json is None:
    raise ValueError("Config file is empty or not found.")
config = Config(**config_json)
