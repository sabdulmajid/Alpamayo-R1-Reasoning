"""Bind streamed Hugging Face files to an immutable repository revision."""

from __future__ import annotations

import contextlib
import re
import types
from pathlib import Path
from typing import Any, Iterator

from huggingface_hub import try_to_load_from_cache


_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def revision_qualified_path(
    repo_id: str, repo_type: str, revision: str, filename: str
) -> str:
    """Return an HfFileSystem path that includes an immutable revision."""

    if not _COMMIT_SHA.fullmatch(revision):
        raise ValueError("Dataset streaming requires a full lowercase commit SHA")
    if not repo_id or "@" in repo_id or not filename or filename.startswith("/"):
        raise ValueError("Repository and file paths are invalid")
    prefix = "datasets/" if repo_type == "dataset" else ""
    return f"{prefix}{repo_id}@{revision}/{filename}"


@contextlib.contextmanager
def _open_revision_pinned_file(
    interface: Any,
    filename: str,
    mode: str = "rb",
    maybe_stream: bool = False,
) -> Iterator[Any]:
    snapshot = dict(interface.repo_snapshot_info)
    revision = str(snapshot.get("revision", ""))
    cached = try_to_load_from_cache(
        filename=filename,
        cache_dir=interface.cache_dir,
        **snapshot,
    )
    if isinstance(cached, str):
        with Path(cached).open(mode) as handle:
            yield handle
        return
    if not maybe_stream:
        raise FileNotFoundError(
            f"filename={filename!r} is not cached; enable streaming to read it"
        )
    remote_path = revision_qualified_path(
        str(snapshot["repo_id"]),
        str(snapshot.get("repo_type", "model")),
        revision,
        filename,
    )
    with interface.fs.open(remote_path, mode) as handle:
        yield handle


def pin_streaming_revision(interface: Any, expected_revision: str) -> Any:
    """Replace one dataset interface's streaming path with a revision-bound path."""

    snapshot = getattr(interface, "repo_snapshot_info", None)
    if not isinstance(snapshot, dict):
        raise TypeError("Dataset interface has no repository snapshot metadata")
    actual_revision = str(snapshot.get("revision", ""))
    if actual_revision != expected_revision:
        raise ValueError(
            f"Dataset interface revision {actual_revision!r} differs from "
            f"{expected_revision!r}"
        )
    revision_qualified_path(
        str(snapshot.get("repo_id", "")),
        str(snapshot.get("repo_type", "model")),
        actual_revision,
        "revision-check",
    )
    interface.open_file = types.MethodType(_open_revision_pinned_file, interface)
    interface.revision_pinned_streaming = True
    return interface
