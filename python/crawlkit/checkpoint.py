# python/crawlkit/checkpoint.py
"""
Crash-safe crawl checkpoint.

A checkpoint is written to disk after every page so that a resumed crawl
skips already-processed URLs and picks up polling from the same job_id.

File format: JSON (human-readable, inspectable, editable manually).

Usage
-----
    # Starting a new crawl
    ck = CrawlCheckpoint.new(job_id="abc123", path=".crawl_checkpoint.json")
    ck.save()

    # After processing each page
    ck.add_processed(url)
    ck.credits_used = new_total
    ck.save()

    # On restart
    ck = CrawlCheckpoint.load(".crawl_checkpoint.json")
    if ck is None:
        ck = CrawlCheckpoint.new(...)
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_PATH = ".crawl_checkpoint.json"


@dataclass
class CheckpointConfig:
    """
    Configuration for checkpoint behaviour.

    Args:
        path:    Where to write the checkpoint file.
        enabled: Set to False to disable checkpointing (useful in tests).
    """

    path: str = _DEFAULT_PATH
    enabled: bool = True


class CrawlCheckpoint:
    """
    Mutable checkpoint state for one crawl job.

    Uses atomic write (temp file + os.replace) so a crash mid-write
    never produces a corrupt checkpoint.
    """

    def __init__(
        self,
        job_id: str,
        path: str = _DEFAULT_PATH,
        processed_urls: set[str] | None = None,
        credits_used: int = 0,
    ) -> None:
        self.job_id = job_id
        self.path = path
        self.processed_urls: set[str] = processed_urls or set()
        self.credits_used = credits_used

    @classmethod
    def new(cls, job_id: str, path: str = _DEFAULT_PATH) -> CrawlCheckpoint:
        """Create a fresh checkpoint immediately after receiving a job_id."""
        ck = cls(job_id=job_id, path=path)
        logger.debug("New checkpoint: job_id=%s path=%s", job_id, path)
        return ck

    @classmethod
    def load(cls, path: str = _DEFAULT_PATH) -> CrawlCheckpoint | None:
        """
        Load a checkpoint from disk.

        Returns None if the file doesn't exist or is corrupt.
        Caller should start a fresh crawl in that case.
        """
        p = Path(path)
        if not p.exists():
            logger.debug("No checkpoint file at %s", path)
            return None
        try:
            raw = p.read_text(encoding="utf-8")
            data = json.loads(raw)
            ck = cls(
                job_id=data["job_id"],
                path=path,
                processed_urls=set(data.get("processed_urls", [])),
                credits_used=int(data.get("credits_used", 0)),
            )
            logger.info(
                "Loaded checkpoint: job_id=%s, %d processed URLs, %d credits",
                ck.job_id,
                len(ck.processed_urls),
                ck.credits_used,
            )
            return ck
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Corrupt checkpoint at %s (%s) -- ignoring", path, exc)
            return None

    def add_processed(self, url: str) -> None:
        self.processed_urls.add(url)

    def is_processed(self, url: str) -> bool:
        return url in self.processed_urls

    def save(self) -> None:
        """
        Persist the checkpoint atomically.

        Writes to a sibling temp file then calls os.replace().
        On POSIX, os.replace() is guaranteed atomic -- readers never see
        a partial write.
        """
        data = {
            "job_id": self.job_id,
            "processed_urls": sorted(self.processed_urls),
            "credits_used": self.credits_used,
        }
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=p.parent, prefix=".ck_tmp_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, p)
            logger.debug(
                "Checkpoint saved: job_id=%s, %d URLs",
                self.job_id,
                len(self.processed_urls),
            )
        except Exception:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp_path)
            raise

    def delete(self) -> None:
        """Remove the checkpoint file after a successful crawl."""
        try:
            os.unlink(self.path)
            logger.debug("Checkpoint deleted: %s", self.path)
        except FileNotFoundError:
            pass
