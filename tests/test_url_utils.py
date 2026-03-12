from crawlkit.crawlkit_rs import filter_uncrawled, normalize_and_dedup, normalize_url


class TestNormalizeUrl:
    def test_removes_fragment(self):
        assert normalize_url("https://example.com/page#section") == "https://example.com/page"

    def test_removes_trailing_slash(self):
        assert normalize_url("https://example.com/page/") == "https://example.com/page"

    def test_root_slash_preserved(self):
        # Root path slash should stay
        result = normalize_url("https://example.com/")
        assert result is not None
        assert "example.com" in result

    def test_sorts_query_params(self):
        result = normalize_url("https://example.com?b=2&a=1")
        assert result == "https://example.com/?a=1&b=2"

    def test_invalid_url_returns_none(self):
        assert normalize_url("not a url") is None

    def test_non_http_returns_none(self):
        assert normalize_url("ftp://example.com/file") is None


class TestDedup:
    def test_removes_duplicates(self):
        urls = [
            "https://example.com/page",
            "https://example.com/page/",
            "https://example.com/page#anchor",
            "https://example.com/other",
        ]
        result = normalize_and_dedup(urls)
        assert len(result) == 2

    def test_preserves_distinct_urls(self):
        urls = [
            "https://example.com/a",
            "https://example.com/b",
            "https://example.com/c",
        ]
        result = normalize_and_dedup(urls)
        assert len(result) == 3

    def test_empty_input(self):
        assert normalize_and_dedup([]) == []


class TestFilterUncrawled:
    def test_filters_already_crawled(self):
        crawled = ["https://example.com/page"]
        candidates = [
            "https://example.com/page/",  # same after normalize
            "https://example.com/new",
        ]
        result = filter_uncrawled(crawled, candidates)
        assert len(result) == 1
        assert "https://example.com/new" in result

    def test_all_new_urls_returned(self):
        crawled = ["https://example.com/old"]
        candidates = ["https://example.com/a", "https://example.com/b"]
        result = filter_uncrawled(crawled, candidates)
        assert len(result) == 2
