use dashmap::DashMap;
use rayon::prelude::*;
use url::Url;

/// Normalize a single URL: remove fragment, strip trailing slash,
/// sort query parameters alphabetically.
/// Returns None if the URL is unparseable.
pub fn normalize_url(raw: &str) -> Option<String> {
    let mut parsed = Url::parse(raw.trim()).ok()?;

    // Only handle http/https
    if parsed.scheme() != "http" && parsed.scheme() != "https" {
        return None;
    }

    // Strip fragment (#section)
    parsed.set_fragment(None);

    // Strip trailing slash from non-root paths
    {
        let path = parsed.path().to_string();
        if path.len() > 1 && path.ends_with('/') {
            parsed.set_path(path.trim_end_matches('/'));
        }
    }

    // Sort query parameters
    if parsed.query().is_some() {
        let mut pairs: Vec<(String, String)> = parsed
            .query_pairs()
            .map(|(k, v)| (k.into_owned(), v.into_owned()))
            .collect();

        if pairs.is_empty() {
            parsed.set_query(None);
        } else {
            pairs.sort_by(|a, b| a.0.cmp(&b.0));
            let query = pairs
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect::<Vec<_>>()
                .join("&");
            parsed.set_query(Some(&query));
        }
    }

    Some(parsed.to_string())
}

/// Normalize and deduplicate a list of URLs in parallel.
/// The result order is non-deterministic (parallel dedup).
pub fn normalize_and_dedup(urls: &[String]) -> Vec<String> {
    // DashMap is a concurrent HashMap — safe to insert from multiple Rayon threads
    let seen: DashMap<String, ()> = DashMap::new();
    let mut result: Vec<String> = Vec::new();

    // We can't use par_iter() + collect() here while maintaining first-seen
    // semantics, so we normalize in parallel and dedup sequentially.
    let normalized: Vec<Option<String>> = urls.par_iter().map(|url| normalize_url(url)).collect();

    for norm in normalized.into_iter().flatten() {
        if seen.insert(norm.clone(), ()).is_none() {
            result.push(norm);
        }
    }

    result
}

/// Given a set of already-crawled URLs and a set of candidates,
/// return only candidates that haven't been crawled yet.
/// Both sets are normalized before comparison.
pub fn filter_uncrawled(crawled: &[String], candidates: &[String]) -> Vec<String> {
    let crawled_set: DashMap<String, ()> = DashMap::new();

    // Build crawled set in parallel
    crawled.par_iter().for_each(|url| {
        if let Some(normalized) = normalize_url(url) {
            crawled_set.insert(normalized, ());
        }
    });

    // Filter candidates
    candidates
        .par_iter()
        .filter_map(|url| {
            let normalized = normalize_url(url)?;
            if !crawled_set.contains_key(&normalized) {
                Some(normalized)
            } else {
                None
            }
        })
        .collect()
}
