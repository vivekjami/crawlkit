use scraper::{Html, Selector};

// Elements that are reliably noise
const NOISE_TAGS: &[&str] = &[
    "script", "style", "nav", "header", "footer", "aside", "noscript", "iframe", "svg", "form",
    "button", "input", "select", "textarea",
];

// Elements that contain primary content — we extract from these
const CONTENT_TAGS: &[&str] = &[
    "main",
    "article",
    "section",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "li",
    "td",
    "th",
    "blockquote",
    "pre",
    "code",
];

/// Extract clean text from HTML.
/// Strategy: parse the document, select content elements,
/// collect their text nodes while skipping noise elements.
pub fn extract_clean_text(html: &str) -> String {
    let document = Html::parse_document(html);

    // Build noise element ID set so we can skip their descendants
    let noise_sel = Selector::parse(&NOISE_TAGS.join(",")).unwrap();
    let mut noise_ids = std::collections::HashSet::new();
    for el in document.select(&noise_sel) {
        noise_ids.insert(el.id());
        for desc in el.descendants() {
            noise_ids.insert(desc.id());
        }
    }

    // Collect text from content elements, skipping anything in noise
    let content_sel = Selector::parse(&CONTENT_TAGS.join(",")).unwrap();
    let mut seen = std::collections::HashSet::new();
    let mut parts: Vec<String> = Vec::new();

    for el in document.select(&content_sel) {
        if noise_ids.contains(&el.id()) {
            continue;
        }
        // Gather text from this element (all descendant text nodes)
        let text: String = el
            .descendants()
            .filter(|n| !noise_ids.contains(&n.id()))
            .filter_map(|n| n.value().as_text())
            .map(|t| t.trim())
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join(" ");

        let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
        if !text.is_empty() && !seen.contains(&text) {
            seen.insert(text.clone());
            parts.push(text);
        }
    }

    parts.join("\n")
}

/// Rough token estimate: word count × 1.3
/// (OpenAI tokenizer averages ~1.3 tokens per English word)
pub fn estimate_tokens(text: &str) -> usize {
    let word_count = text.split_whitespace().count();
    ((word_count as f64) * 1.3).ceil() as usize
}

/// Returns (html_tokens, clean_tokens, reduction_percent)
pub fn token_comparison(html: &str, clean_text: &str) -> (usize, usize, f64) {
    let html_tokens = estimate_tokens(html);
    let clean_tokens = estimate_tokens(clean_text);
    let reduction = if html_tokens > 0 {
        (1.0 - clean_tokens as f64 / html_tokens as f64) * 100.0
    } else {
        0.0
    };
    (html_tokens, clean_tokens, reduction)
}
