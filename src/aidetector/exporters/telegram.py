from typing import Self

import requests

from aidetector.config import ChatConfig, Config, Detection, DetectorConfig, get_timestamped_filename
from aidetector.exporters.exporter import Exporter


class TelegramExporter(Exporter):
    base_url: str
    chat: str

    def __init__(self, token: str, chat: str, confidence: float):
        super().__init__(confidence, token, chat)
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.chat = chat

    @classmethod
    def from_config(cls, config: Config, detector: DetectorConfig, exporter: ChatConfig) -> Self:
        return cls(
            exporter.token,
            exporter.chat,
            confidence=exporter.confidence or detector.detection.confidence,
        )

    def filtered_export(self, detections: list[Detection]):
        try:
            self.logger.info(f"Sending photo to Telegram with confidence {detections[0].confidence}")
            url = f"{self.base_url}/sendPhoto"
            files = {
                "photo": (
                    get_timestamped_filename(detections[0]),
                    detections[0].jpg,
                    "image/jpeg",
                )
            }
            payload = {
                "chat_id": self.chat,
                "caption": "👍 / 👎",
            }
            response = requests.post(url, data=payload, files=files)
            if response.status_code != 200:
                self.logger.error(f"Failed to send photo to Telegram: {response.text}")
        except Exception as e:
            self.logger.error(f"Error sending photo to Telegram: {e}")
