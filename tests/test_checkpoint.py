# tests/test_checkpoint.py
import os

import pytest
from crawlkit.checkpoint import CrawlCheckpoint


@pytest.fixture
def tmp_ck_path(tmp_path):
    return str(tmp_path / "test_checkpoint.json")


class TestNewCheckpoint:
    def test_new_has_empty_processed_urls(self):
        ck = CrawlCheckpoint.new(job_id="job-001")
        assert ck.processed_urls == set()

    def test_new_has_correct_job_id(self):
        ck = CrawlCheckpoint.new(job_id="job-abc")
        assert ck.job_id == "job-abc"

    def test_new_has_zero_credits(self):
        ck = CrawlCheckpoint.new(job_id="job-001")
        assert ck.credits_used == 0


class TestSaveAndLoad:
    def test_save_creates_file(self, tmp_ck_path):
        ck = CrawlCheckpoint.new(job_id="job-save", path=tmp_ck_path)
        ck.save()
        assert os.path.exists(tmp_ck_path)

    def test_load_returns_same_job_id(self, tmp_ck_path):
        CrawlCheckpoint.new(job_id="job-save", path=tmp_ck_path).save()
        loaded = CrawlCheckpoint.load(tmp_ck_path)
        assert loaded is not None
        assert loaded.job_id == "job-save"

    def test_load_returns_same_credits(self, tmp_ck_path):
        ck = CrawlCheckpoint.new(job_id="job-save", path=tmp_ck_path)
        ck.credits_used = 42
        ck.save()
        loaded = CrawlCheckpoint.load(tmp_ck_path)
        assert loaded is not None
        assert loaded.credits_used == 42


class TestLoadMissingFile:
    def test_load_returns_none_for_missing_file(self, tmp_path):
        result = CrawlCheckpoint.load(str(tmp_path / "nonexistent.json"))
        assert result is None

    def test_load_returns_none_for_corrupt_json(self, tmp_path):
        p = tmp_path / "corrupt.json"
        p.write_text("not valid json {{{")
        result = CrawlCheckpoint.load(str(p))
        assert result is None

    def test_load_returns_none_for_missing_required_key(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text('{"processed_urls": []}')
        result = CrawlCheckpoint.load(str(p))
        assert result is None


class TestProcessedUrlsPersistence:
    def test_processed_urls_survive_round_trip(self, tmp_ck_path):
        ck = CrawlCheckpoint.new(job_id="job-urls", path=tmp_ck_path)
        urls = {"https://example.com/a", "https://example.com/b", "https://example.com/c"}
        for url in urls:
            ck.add_processed(url)
        ck.save()
        loaded = CrawlCheckpoint.load(tmp_ck_path)
        assert loaded is not None
        assert loaded.processed_urls == urls

    def test_is_processed_returns_correct_value(self):
        ck = CrawlCheckpoint.new(job_id="job-check")
        ck.add_processed("https://example.com/page")
        assert ck.is_processed("https://example.com/page") is True
        assert ck.is_processed("https://example.com/other") is False

    def test_multiple_saves_accumulate_urls(self, tmp_ck_path):
        ck = CrawlCheckpoint.new(job_id="job-accum", path=tmp_ck_path)
        ck.add_processed("https://example.com/a")
        ck.save()
        ck.add_processed("https://example.com/b")
        ck.save()
        loaded = CrawlCheckpoint.load(tmp_ck_path)
        assert loaded is not None
        assert "https://example.com/a" in loaded.processed_urls
        assert "https://example.com/b" in loaded.processed_urls


class TestCheckpointDelete:
    def test_delete_removes_file(self, tmp_ck_path):
        ck = CrawlCheckpoint.new(job_id="job-del", path=tmp_ck_path)
        ck.save()
        ck.delete()
        assert not os.path.exists(tmp_ck_path)

    def test_delete_is_safe_when_no_file(self, tmp_ck_path):
        ck = CrawlCheckpoint.new(job_id="job-del", path=tmp_ck_path)
        ck.delete()
