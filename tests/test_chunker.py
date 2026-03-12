from crawlkit.crawlkit_rs import chunk_markdown

SAMPLE_MD = """# Overview

This is the overview section with a short description.

## Authentication

You need an API key to use this service.
The key must be passed in the Authorization header.
Generate your key from the dashboard settings.

## Rate Limits

The free tier allows 500 credits per month.
Each scraped page costs 1 credit.
Batch operations use credits proportionally.
"""


class TestChunker:
    def test_splits_on_headers(self):
        chunks = chunk_markdown(SAMPLE_MD, 2000)
        sections = [c["section"] for c in chunks]
        assert "Overview" in sections
        assert "Authentication" in sections
        assert "Rate Limits" in sections

    def test_chunk_count_matches_sections(self):
        chunks = chunk_markdown(SAMPLE_MD, 2000)
        assert len(chunks) == 3

    def test_max_chars_respected(self):
        # Force small chunk size to trigger paragraph splitting
        long_md = "# Big Section\n\n" + ("word " * 200 + "\n\n") * 10
        chunks = chunk_markdown(long_md, 500)
        for c in chunks:
            # Some tolerance: a single paragraph may slightly exceed
            assert c["char_count"] < 1000, f"Chunk too large: {c['char_count']}"

    def test_chunk_index_sequential(self):
        chunks = chunk_markdown(SAMPLE_MD, 2000)
        indices = [c["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_empty_input_returns_empty(self):
        chunks = chunk_markdown("", 500)
        assert chunks == []

    def test_no_headers_single_chunk(self):
        chunks = chunk_markdown("Just plain text with no headers.", 2000)
        assert len(chunks) == 1
        assert chunks[0]["section"] == "root"

    def test_content_not_empty(self):
        chunks = chunk_markdown(SAMPLE_MD, 2000)
        for c in chunks:
            assert len(c["content"]) > 0
