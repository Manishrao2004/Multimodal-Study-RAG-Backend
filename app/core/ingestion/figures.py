"""Figure extraction, filtering and de-duplication — EduRAG Tier-1, Sec. III-A.

A textbook PDF yields far more embedded images than actual diagrams: page
rules, logos, bullet glyphs, watermarks, and the same figure repeated in a
running header. Captioning all of them with a VLM is slow and pollutes the
index with junk chunks, so images pass two filters before captioning:

  1. Geometry  — minimum width/height/area and a maximum aspect ratio, which
                 removes icons and long thin rules.
  2. Perceptual hash — a 64-bit dHash; images within a small Hamming distance
                 of one already kept are treated as the same figure.

dHash is implemented here directly rather than pulling in `imagehash`, since
it is ~15 lines and avoids a dependency for one function.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import Settings


@dataclass
class ExtractedFigure:
    """One candidate diagram, already filtered and de-duplicated."""

    index: int
    page_number: int | None
    png_bytes: bytes
    width: int
    height: int
    phash: int
    image_path: Path | None = None


def dhash(image, hash_size: int = 8) -> int:
    """64-bit difference hash: compare each pixel to its right-hand neighbour."""
    from PIL import Image

    grey = image.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = np.asarray(grey, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _passes_geometry(width: int, height: int, settings: Settings) -> bool:
    if width < settings.vlm_min_width_px or height < settings.vlm_min_height_px:
        return False
    if width * height < settings.vlm_min_area_px:
        return False
    longest, shortest = max(width, height), min(width, height)
    if shortest == 0 or longest / shortest > settings.vlm_max_aspect_ratio:
        return False
    return True


def extract_figures(docling_doc, settings: Settings, source_stem: str) -> list[ExtractedFigure]:
    """Pulls picture items out of a converted Docling document, applies the
    geometry filter and perceptual-hash de-duplication, and writes the
    survivors to disk so the API can serve them alongside their captions.

    Requires the converter to have run with `generate_picture_images=True`;
    without it `get_image` returns None and this yields nothing.
    """
    figures: list[ExtractedFigure] = []
    kept_hashes: list[int] = []

    pictures = getattr(docling_doc, "pictures", None) or []
    for i, picture in enumerate(pictures):
        if len(figures) >= settings.vlm_max_images_per_doc:
            break
        try:
            pil_image = picture.get_image(docling_doc)
        except Exception:
            continue
        if pil_image is None:
            continue

        width, height = pil_image.size
        if not _passes_geometry(width, height, settings):
            continue

        digest = dhash(pil_image)
        if any(hamming_distance(digest, seen) <= settings.vlm_phash_distance for seen in kept_hashes):
            continue
        kept_hashes.append(digest)

        page_number = None
        prov = getattr(picture, "prov", None) or []
        if prov:
            page_number = getattr(prov[0], "page_no", None)

        buffer = io.BytesIO()
        pil_image.convert("RGB").save(buffer, format="PNG")
        png_bytes = buffer.getvalue()

        image_path = settings.image_dir / f"{source_stem}_fig{len(figures):03d}.png"
        image_path.write_bytes(png_bytes)

        figures.append(
            ExtractedFigure(
                index=len(figures),
                page_number=page_number,
                png_bytes=png_bytes,
                width=width,
                height=height,
                phash=digest,
                image_path=image_path,
            )
        )
    return figures
