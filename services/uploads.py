"""
Photo storage: the customer's pictures of what they're moving.

WHY PHOTOS DO NOT GO THROUGH THE GRAPH
--------------------------------------
The tempting design is to put the images straight into the conversation. It
breaks quietly and expensively. LangGraph writes the entire state into the
checkpoint database after every step, so a base64 photo in state is re-serialised
on every step of every subsequent turn. Eight phone photos is roughly 30 MB of
state, rewritten each time the customer says "thanks" — the database grows
without bound and every message gets slower.

So the browser posts photos to /api/upload BEFORE submitting the form. They land
on disk here, and the only thing that travels through the graph is a list of
short ids.

TRUST
-----
Everything arriving here came from the public internet. The filename is ignored,
the browser's declared content-type is ignored, and the file's own leading bytes
decide whether it is an image. Anything else is rejected — an endpoint that
writes attacker-named, attacker-typed files to disk and then emails them onward
is a way to deliver a payload to a manager's inbox with our return address on it.
"""

import logging
import re
import secrets
import shutil
from pathlib import Path

from services import config

logger = logging.getLogger(__name__)


class UploadRejected(ValueError):
    """The file is not something we will store. The message is shown to the user."""


# ── What an image actually looks like on disk ──────────────────────────────────
# Keyed by extension, because the extension is what the file is stored under and
# what the mime type is derived from later.

def _sniff(data: bytes) -> tuple[str, str] | None:
    """(extension, mime) from the file's own leading bytes, or None."""
    if data[:3] == b"\xff\xd8\xff":
        return "jpg", "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png", "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp", "image/webp"
    # HEIC/HEIF — what iPhones produce by default. The brand sits at offset 8,
    # just past the box size and the 'ftyp' marker.
    if data[4:8] == b"ftyp" and data[8:12] in (b"heic", b"heix", b"hevc", b"mif1", b"heim"):
        return "heic", "image/heic"
    return None


_SAFE_THREAD = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _thread_dir(thread_id: str, *, create: bool = False) -> Path:
    """
    The directory for one conversation's photos.

    The thread id comes from the request body, so it is checked against a strict
    pattern rather than sanitised. A thread id of "../../etc" would otherwise
    turn this function into an arbitrary-path writer.
    """
    if not _SAFE_THREAD.match(thread_id or ""):
        raise UploadRejected("Something went wrong with this upload — try reloading the page.")
    path = config.upload_dir() / thread_id
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def save(thread_id: str, data: bytes) -> dict[str, str]:
    """
    Store one photo. Returns {"id", "ext", "mime", "bytes"}.

    Raises UploadRejected with a message meant for the customer's eyes.
    """
    if not data:
        raise UploadRejected("That file was empty.")

    if len(data) > config.MAX_UPLOAD_BYTES:
        mb = config.MAX_UPLOAD_BYTES // (1024 * 1024)
        raise UploadRejected(f"That photo is over {mb} MB — try a smaller one.")

    sniffed = _sniff(data)
    if sniffed is None:
        raise UploadRejected("That doesn't look like a photo. JPG, PNG, WEBP or HEIC please.")
    ext, mime = sniffed

    directory = _thread_dir(thread_id, create=True)
    if len(list(directory.glob("*.*"))) >= config.MAX_UPLOADS_PER_THREAD:
        raise UploadRejected(
            f"That's the {config.MAX_UPLOADS_PER_THREAD} photo limit — plenty for "
            "an estimate. Send the rest to the office if a manager asks."
        )

    upload_id = secrets.token_urlsafe(9)
    (directory / f"{upload_id}.{ext}").write_bytes(data)

    logger.info("Upload: %s/%s.%s (%d bytes)", thread_id, upload_id, ext, len(data))
    return {"id": upload_id, "ext": ext, "mime": mime, "bytes": str(len(data))}


_MIME_BY_EXT = {
    "jpg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
    "heic": "image/heic",
}


def list_for(thread_id: str) -> list[dict[str, str]]:
    """Every stored photo for a thread, oldest first."""
    try:
        directory = _thread_dir(thread_id)
    except UploadRejected:
        return []
    if not directory.is_dir():
        return []

    out = []
    for path in sorted(directory.glob("*.*"), key=lambda p: p.stat().st_mtime):
        ext = path.suffix.lstrip(".").lower()
        if ext not in _MIME_BY_EXT:
            continue
        out.append({"id": path.stem, "ext": ext, "mime": _MIME_BY_EXT[ext], "path": str(path)})
    return out


def collect(thread_id: str, ids: list[str]) -> list[dict]:
    """
    The photos named by `ids`, with their bytes, ready to attach to an email.

    Ids that do not exist are skipped silently rather than raising. They arrive
    from the browser, and the realistic cause is a customer who removed a photo
    after it uploaded — which should not stop their estimate from being sent.
    """
    wanted = set(ids or [])
    files = []
    for index, meta in enumerate(list_for(thread_id), start=1):
        if meta["id"] not in wanted:
            continue
        files.append(
            {
                "filename": f"photo-{index}.{meta['ext']}",
                "mime": meta["mime"],
                "content": Path(meta["path"]).read_bytes(),
            }
        )
    return files


def discard(thread_id: str) -> None:
    """
    Delete a thread's photos once the lead has been emailed.

    They have reached the office inbox, which is where they are useful and where
    they will be kept. Leaving a second copy on a web server that anyone can post
    to is storing strangers' belongings for no reason.
    """
    try:
        directory = _thread_dir(thread_id)
    except UploadRejected:
        return
    if directory.is_dir():
        shutil.rmtree(directory, ignore_errors=True)
        logger.info("Upload: cleared %s", thread_id)
