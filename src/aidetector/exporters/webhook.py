import base64
import io
from typing import Self

import requests
from PIL import Image

from aidetector.config import (
    Config,
    Detection,
    DetectorConfig,
    WebhookConfig,
    get_date_path,
    get_timestamped_filename,
)
from aidetector.exporters.exporter import Exporter


class WebhookExporter(Exporter):
    webhook_url: str
    webhook_token: str
    confidence: float

    def __init__(self, webhook_url: str, webhook_token: str, confidence: float):
        super().__init__(confidence, webhook_url, webhook_token)
        self.webhook_url: str = webhook_url
        self.webhook_token: str = webhook_token
        self.confidence: float = confidence

    @classmethod
    def from_config(cls, config: Config, detector: DetectorConfig, exporter: WebhookConfig) -> Self:
        return cls(
            exporter.webhook_url,
            exporter.webhook_token,
            confidence=exporter.confidence or detector.detection.confidence,
        )

    # @classmethod
    # def fromConfig(cls, config: Config, detector: DetectorConfig) -> Self | None:
    #     # If one of the config items of wehook is empty return none
    #     if config.webhook_url is None or config.webhook_token is None:
    #         return None
    #     return cls(config.webhook_url, config.webhook_token)

    def filtered_export(self, sorted_detections: list[Detection]):
        if not sorted_detections:
            return
        try:
            self.logger.info(f"Sending photo to Webhook with confidence {sorted_detections[0].confidence}")
            headers = {"Authorization": f"Bearer {self.webhook_token}"}

            files = {
                "photo": (
                    get_timestamped_filename(sorted_detections[0]),
                    sorted_detections[0].jpg,
                    "image/jpeg",
                )
            }
            photo_data = files["photo"][1]

            # Maak een kleinere thumbnail
            img = Image.open(io.BytesIO(photo_data))

            # Resize naar max 800px breed (houdt aspect ratio)
            max_width = 800
            if img.width > max_width:
                ratio = max_width / img.width
                new_size = (max_width, int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Converteer naar JPEG met lagere kwaliteit
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=60, optimize=True)
            thumbnail_data = buffer.getvalue()

            # Encodeer naar base64
            photo_base64 = base64.b64encode(thumbnail_data).decode("utf-8")

            # Genereer unieke bestandsnaam
            timestamp = get_date_path(sorted_detections[0], "seconds")
            filename = f"detection_{timestamp}.jpg"

            # Check grootte (moet < 250KB zijn voor template)
            if len(photo_base64) > 200000:
                print(f"Waarschuwing: foto nog steeds te groot ({len(photo_base64)} chars)")
                # Probeer nog kleiner
                img = img.resize((int(img.width * 0.5), int(img.height * 0.5)), Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=50, optimize=True)
                thumbnail_data = buffer.getvalue()
                photo_base64 = base64.b64encode(thumbnail_data).decode("utf-8")

            # Verstuur naar save_detection_photo webhook
            data = {
                "confidence": f"{sorted_detections[0].confidence * 100:.1f}%",
                "filename": filename,
                "image": photo_base64,
            }

            response = requests.post(
                self.webhook_url,
                headers=headers,
                json=data,
            )
            # Check of het gelukt is
            if response.status_code == 200:
                print(
                    f"Webhook succesvol verstuurd naar Home Assistant (confidence: {sorted_detections[0].confidence})"
                )
                return True
            else:
                print(f"Fout bij versturen webhook: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            self.logger.error(f"Error sending photo to Webhook: {e}")
