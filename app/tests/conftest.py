import pytest

from app.config import Settings
from app.models.schemas import Chunk, ChunkType


@pytest.fixture(autouse=True)
def block_provider_network(monkeypatch):
    import httpx
    def blocked(*args, **kwargs):
        raise AssertionError("These regression tests must not make network calls")
    monkeypatch.setattr(httpx.Client, "send", blocked)
    monkeypatch.setattr(httpx.AsyncClient, "send", blocked)


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, llm_provider="none", vlm_provider="none",
                    evidence_method="adaptive",
                    hyde_provider="none", hyde_enabled=False, asr_provider="none",
                    image_dir=tmp_path / "images", kb_db_path=tmp_path / "kb.sqlite3")


def chunk(text="A transformer transfers energy through electromagnetic induction.",
          chunk_id="c1", kind=ChunkType.text, **kwargs):
    return Chunk(chunk_id=chunk_id, source_file="notes.pdf", page_number=1,
                 text=text, chunk_length=len(text), type=kind, **kwargs)
