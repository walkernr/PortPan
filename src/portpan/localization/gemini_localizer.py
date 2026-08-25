"""Hierarchical Gemini Vision API client for semantic coarse localization."""

from __future__ import annotations

import io
import os
from typing import Any
import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
from portpan.core.exceptions import LocalizationError
from portpan.core.models import BoundingBox
from portpan.localization.base import BaseCoarseLocalizer


class GeminiCandidateBox(BaseModel):
    """Schema for structured Gemini output."""

    ymin: float = Field(description="Normalized top coordinate [0.0 - 1.0]")
    xmin: float = Field(description="Normalized left coordinate [0.0 - 1.0]")
    ymax: float = Field(description="Normalized bottom coordinate [0.0 - 1.0]")
    xmax: float = Field(description="Normalized right coordinate [0.0 - 1.0]")
    confidence: float = Field(default=0.9, description="Confidence score [0.0 - 1.0]")
    visual_clue: str = Field(default="", description="Description of landmark/features matched")


class GeminiLocalizationResponse(BaseModel):
    """Structured response list from Gemini."""

    candidates: list[GeminiCandidateBox] = Field(description="List of top candidate bounding boxes in the panorama")


class GeminiCoarseLocalizer(BaseCoarseLocalizer):
    """Uses Google Gemini multimodal models to identify candidate sub-regions in ultra-wide panoramas."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-2.5-flash",
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name

    def _encode_jpeg(self, image: np.ndarray, max_dim: int = 2048) -> bytes:
        """Resize and encode a NumPy BGR image to JPEG bytes."""
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    def localize(self, base_image: np.ndarray, target_patch: np.ndarray) -> list[BoundingBox]:
        """Call Gemini API with panorama and target crop to retrieve candidate bounding boxes."""
        if not self.api_key:
            raise LocalizationError(
                "Gemini API key is not set. Please set GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
            )

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)

            base_bytes = self._encode_jpeg(base_image, max_dim=2560)
            target_bytes = self._encode_jpeg(target_patch, max_dim=1024)

            prompt = (
                "You are an expert computer vision assistant specializing in geographic landmark matching.\n"
                "Image 1 is a wide base panoramic photo of a landscape or seascape.\n"
                "Image 2 is a high-magnification digiscope telephoto observation taken of a specific sub-region within Image 1.\n\n"
                "Task: Locate where Image 2 is located within Image 1 (the base panorama).\n"
                "Provide the normalized bounding box coordinates [ymin, xmin, ymax, xmax] in the range 0.0 to 1.0 relative to Image 1.\n"
                "Provide up to 3 candidate locations if ambiguous, ordered by confidence."
            )

            response = client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Part.from_bytes(data=base_bytes, mime_type="image/jpeg"),
                    types.Part.from_bytes(data=target_bytes, mime_type="image/jpeg"),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GeminiLocalizationResponse,
                    temperature=0.1,
                ),
            )

            if not response.parsed or not response.parsed.candidates:
                raise LocalizationError("Gemini returned no candidate bounding boxes")

            results = []
            for c in response.parsed.candidates:
                # Clamp coordinates
                box = BoundingBox(
                    ymin=max(0.0, min(1.0, c.ymin)),
                    xmin=max(0.0, min(1.0, c.xmin)),
                    ymax=max(0.0, min(1.0, c.ymax)),
                    xmax=max(0.0, min(1.0, c.xmax)),
                    confidence=max(0.1, min(1.0, c.confidence)),
                )
                results.append(box)

            return results

        except Exception as e:
            raise LocalizationError(f"Gemini coarse localization failed: {e}") from e
