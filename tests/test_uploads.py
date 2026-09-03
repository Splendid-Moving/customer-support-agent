"""
Photo storage.

Everything reaching this module came from the public internet, so most of these
tests are about what it refuses rather than what it stores.
"""

import pytest

from services import config, uploads
from tests.conftest import HEIC, JPEG, NOT_AN_IMAGE, PNG, WEBP

THREAD = "thread-abc123"


@pytest.mark.parametrize(
    "data,ext",
    [(JPEG, "jpg"), (PNG, "png"), (WEBP, "webp"), (HEIC, "heic")],
)
def test_real_image_formats_are_stored(data, ext):
    saved = uploads.save(THREAD, data)
    assert saved["ext"] == ext
    assert saved["id"]


def test_a_file_that_is_not_an_image_is_refused():
    """Whatever it is called and whatever type the browser claims."""
    with pytest.raises(uploads.UploadRejected):
        uploads.save(THREAD, NOT_AN_IMAGE)


def test_an_empty_file_is_refused():
    with pytest.raises(uploads.UploadRejected):
        uploads.save(THREAD, b"")


def test_an_oversized_photo_is_refused():
    huge = JPEG + b"\x00" * config.MAX_UPLOAD_BYTES
    with pytest.raises(uploads.UploadRejected, match="MB"):
        uploads.save(THREAD, huge)


def test_the_per_thread_limit_holds():
    for _ in range(config.MAX_UPLOADS_PER_THREAD):
        uploads.save(THREAD, JPEG)
    with pytest.raises(uploads.UploadRejected, match="limit"):
        uploads.save(THREAD, JPEG)


@pytest.mark.parametrize("evil", ["../../etc", "..", "a/b", "", "x" * 100, "thread id"])
def test_a_thread_id_cannot_escape_its_directory(evil):
    """Without this check the upload endpoint writes anywhere on the filesystem."""
    with pytest.raises(uploads.UploadRejected):
        uploads.save(evil, JPEG)


def test_threads_cannot_see_each_others_photos():
    mine = uploads.save("thread-mine", JPEG)
    uploads.save("thread-yours", PNG)
    assert [f["id"] for f in uploads.list_for("thread-mine")] == [mine["id"]]


def test_collect_returns_only_the_photos_that_were_asked_for():
    keep = uploads.save(THREAD, JPEG)
    uploads.save(THREAD, PNG)
    files = uploads.collect(THREAD, [keep["id"]])
    assert len(files) == 1
    assert files[0]["mime"] == "image/jpeg"
    assert files[0]["content"] == JPEG


def test_an_id_that_does_not_exist_is_skipped_not_fatal():
    """A customer who removed a photo must not lose the whole estimate."""
    keep = uploads.save(THREAD, JPEG)
    assert len(uploads.collect(THREAD, [keep["id"], "made-up-id"])) == 1


def test_attachments_are_named_for_a_human_reading_the_email():
    uploads.save(THREAD, JPEG)
    ids = [f["id"] for f in uploads.list_for(THREAD)]
    assert uploads.collect(THREAD, ids)[0]["filename"] == "photo-1.jpg"


def test_photos_are_deleted_once_the_lead_is_sent():
    uploads.save(THREAD, JPEG)
    uploads.discard(THREAD)
    assert uploads.list_for(THREAD) == []


def test_discarding_a_thread_that_never_uploaded_anything_is_harmless():
    uploads.discard("thread-empty")
    uploads.discard("../../etc")
