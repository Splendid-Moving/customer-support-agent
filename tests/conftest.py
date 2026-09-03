"""
Shared fixtures.

Every test in this suite runs without a network connection and without an API
key. The nodes that call a model are tested through their pure-Python halves —
the pattern lists, the validators, the routing functions — because those are
where the rules this agent has to keep actually live. A test that needs a model
to tell you whether a regex works is a test that costs money to run and passes
for the wrong reasons.
"""

import os

import pytest

# Set before anything imports services.config, which reads the environment at
# module level via load_dotenv.
os.environ.setdefault("DRY_RUN", "true")
os.environ.setdefault("LANGSMITH_TRACING", "false")


@pytest.fixture(autouse=True)
def isolated_uploads(tmp_path, monkeypatch):
    """Photos never touch the real upload directory during tests."""
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    yield


@pytest.fixture
def knowledge_dir(tmp_path, monkeypatch):
    """
    A knowledge base the test controls.

    Returned as a factory so a test can write exactly the material it wants the
    agent to be limited to.
    """
    directory = tmp_path / "kb"
    directory.mkdir()

    def write(name: str, text: str):
        (directory / name).write_text(text, encoding="utf-8")
        monkeypatch.setenv("KNOWLEDGE_DIR", str(directory))
        # The loader caches on (path, mtimes), so a fresh file is picked up on
        # its own — but the cache is keyed per path and tests reuse tmp dirs.
        from services import knowledge
        knowledge._load.cache_clear()
        return directory

    return write


# ── Sample image bytes ─────────────────────────────────────────────────────────
# Real leading bytes, so the sniffing in services/uploads.py is exercised for
# what it actually does rather than against a stub.

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
WEBP = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 64
HEIC = b"\x00\x00\x00\x18" + b"ftyp" + b"heic" + b"\x00" * 64
NOT_AN_IMAGE = b"%PDF-1.7\n" + b"\x00" * 64
