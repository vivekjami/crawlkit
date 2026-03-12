from crawlkit.crawlkit_rs import extract_clean_text, token_comparison

SAMPLE_HTML = """<!DOCTYPE html>
<html>
<head><title>Test</title></head>
<body>
  <nav><a href="/">Home</a><a href="/docs">Docs</a></nav>
  <script>console.log('tracking')</script>
  <style>.nav { display: none; }</style>
  <main>
    <h1>API Reference</h1>
    <p>The scrape endpoint accepts POST requests.</p>
    <h2>Authentication</h2>
    <p>Pass your API key in the Authorization header.</p>
  </main>
  <footer>Copyright 2025</footer>
</body>
</html>"""


class TestHtmlExtract:
    def test_removes_nav(self):
        clean = extract_clean_text(SAMPLE_HTML)
        assert "Home" not in clean or "API Reference" in clean

    def test_removes_scripts(self):
        clean = extract_clean_text(SAMPLE_HTML)
        assert "console.log" not in clean

    def test_removes_styles(self):
        clean = extract_clean_text(SAMPLE_HTML)
        assert "display: none" not in clean

    def test_keeps_main_content(self):
        clean = extract_clean_text(SAMPLE_HTML)
        assert "API Reference" in clean
        assert "scrape endpoint" in clean

    def test_removes_footer(self):
        clean = extract_clean_text(SAMPLE_HTML)
        assert "Copyright" not in clean

    def test_token_reduction_positive(self):
        clean = extract_clean_text(SAMPLE_HTML)
        _, _, reduction = token_comparison(SAMPLE_HTML, clean)
        assert reduction > 0

    def test_token_comparison_returns_tuple(self):
        clean = extract_clean_text(SAMPLE_HTML)
        result = token_comparison(SAMPLE_HTML, clean)
        assert len(result) == 3
        html_tok, clean_tok, pct = result
        assert isinstance(html_tok, int)
        assert isinstance(clean_tok, int)
        assert isinstance(pct, float)
