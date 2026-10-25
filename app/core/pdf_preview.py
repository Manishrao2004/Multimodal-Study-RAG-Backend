"""Render one bounded PDF page in an isolated process, away from Docling/PDFium."""
import io
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache


def render_page(path: str, number: int) -> tuple[bytes, int]:
    import pypdfium2 as pdfium
    with pdfium.PdfDocument(path) as document:
        count = len(document)
        if number < 1 or number > count:
            raise IndexError("Page is outside the document.")
        page = document[number - 1]
        try:
            width, height = page.get_size()
            scale = min(2.0, 1400 / max(width, 1), 2000 / max(height, 1))
            bitmap = page.render(scale=scale)
            try:
                image = bitmap.to_pil()
                output = io.BytesIO()
                image.save(output, format="PNG")
                image.close()
                return output.getvalue(), count
            finally:
                bitmap.close()
        finally:
            page.close()


def count_pages(path: str) -> int:
    import pypdfium2 as pdfium
    with pdfium.PdfDocument(path) as document:
        return len(document)


@lru_cache(maxsize=128)
def cached_page_count(path: str, modified: int, size: int) -> int:
    return renderer().submit(count_pages, path).result(timeout=15)


@lru_cache(maxsize=1)
def renderer() -> ProcessPoolExecutor:
    # PDFium is not thread-safe; the single worker also bounds rendering memory.
    return ProcessPoolExecutor(max_workers=1)
