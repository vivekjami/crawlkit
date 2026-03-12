from crawlkit.crawlkit_rs import (
    batch_fingerprint_pages,
    content_fingerprint,
    content_has_changed,
)


class TestFingerprint:
    def test_identical_content_same_hash(self):
        h1 = content_fingerprint("hello world")
        h2 = content_fingerprint("hello world")
        assert h1 == h2

    def test_date_noise_ignored(self):
        a = "Docs updated 01/15/2025. The API is /v1/scrape."
        b = "Docs updated 03/10/2026. The API is /v1/scrape."
        assert not content_has_changed(a, b), "Date change should not register as content change"

    def test_real_content_change_detected(self):
        a = "The endpoint is /v1/scrape."
        b = "The endpoint is /v2/scrape."
        assert content_has_changed(a, b), "Real content change must be detected"

    def test_whitespace_normalised(self):
        a = "hello   world"
        b = "hello world"
        assert not content_has_changed(a, b), "Whitespace differences should not register"

    def test_hash_is_64_char_hex(self):
        h = content_fingerprint("anything")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_batch_returns_correct_count(self):
        pages = [("https://example.com/a", "content a"), ("https://example.com/b", "content b")]
        results = batch_fingerprint_pages(pages)
        assert len(results) == 2

    def test_batch_preserves_urls(self):
        pages = [("https://example.com/page", "some content")]
        results = batch_fingerprint_pages(pages)
        assert results[0][0] == "https://example.com/page"
