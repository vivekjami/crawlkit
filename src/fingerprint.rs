use rayon::prelude::*;
use regex::Regex;
use std::sync::OnceLock;

// OnceLock = compile regex once, reuse across threads. Never pay the
// compilation cost more than once.
static DATE_RE: OnceLock<Regex> = OnceLock::new();
static TIMESTAMP_RE: OnceLock<Regex> = OnceLock::new();

fn date_re() -> &'static Regex {
    DATE_RE.get_or_init(|| {
        // Matches: 01/15/2025, 1-5-24, 2024-01-15
        Regex::new(r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b").unwrap()
    })
}

fn timestamp_re() -> &'static Regex {
    TIMESTAMP_RE.get_or_init(|| {
        // Matches: 2024-01-15T12:34:56
        Regex::new(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}").unwrap()
    })
}

/// Strip dynamic noise before hashing so timestamps don't cause false
/// positives. Normalise whitespace so formatting changes don't either.
pub fn normalize_for_fingerprint(content: &str) -> String {
    let s = date_re().replace_all(content, "__DATE__");
    let s = timestamp_re().replace_all(&s, "__TIMESTAMP__");
    // Collapse all whitespace to single spaces — HTML formatters sometimes
    // reflow text without changing content.
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// BLAKE3 hash of normalized content. Returns 64-char hex string.
pub fn fingerprint(content: &str) -> String {
    let normalized = normalize_for_fingerprint(content);
    blake3::hash(normalized.as_bytes()).to_hex().to_string()
}

/// Returns true if the content has meaningfully changed.
pub fn has_changed(old_content: &str, new_content: &str) -> bool {
    fingerprint(old_content) != fingerprint(new_content)
}

/// Fingerprint a batch of (url, content) pairs in parallel.
/// Rayon distributes work across all available CPU cores.
pub fn batch_fingerprint(pages: &[(String, String)]) -> Vec<(String, String)> {
    pages
        .par_iter()
        .map(|(url, content)| (url.clone(), fingerprint(content)))
        .collect()
}
