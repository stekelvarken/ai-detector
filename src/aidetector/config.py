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
class CollectionConfig:
    time_seconds: int
    frames_min: int
    min_confidence: float


@dataclass
class ChatConfig:
    token: str
    chat: str
    min_confidence: float | None = None


@dataclass
class DiskConfig:
    directory: Path
    min_confidence: float | None = None


@dataclass
class ExportersConfig:
    disk: DiskConfig | list[DiskConfig] | None = None
    telegram: ChatConfig | list[ChatConfig] | None = None


@dataclass
class DetectorConfig:
    collection: CollectionConfig
    model_url: str
    sources: list[str]
    exporters: ExportersConfig | None = None


@dataclass
class Config:
    detectors: list[DetectorConfig]


config_json = json.load(open("config.json"))
if config_json is None:
    raise ValueError("Config file is empty or not found.")
config = Config(**config_json)
