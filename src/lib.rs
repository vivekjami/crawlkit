use pyo3::Python;
use pyo3::prelude::*;
use pyo3::types::PyDict;

mod chunker;
mod fingerprint;
mod html_extract;
mod url_utils;

// ---------------------------------------------------------------------------
// fingerprint
// ---------------------------------------------------------------------------

#[pyfunction]
fn content_fingerprint(content: &str) -> PyResult<String> {
    Ok(fingerprint::fingerprint(content))
}

#[pyfunction]
fn content_has_changed(old: &str, new: &str) -> PyResult<bool> {
    Ok(fingerprint::has_changed(old, new))
}

/// pages: list of (url, content) tuples
/// Returns: list of (url, fingerprint) tuples
#[pyfunction]
fn batch_fingerprint_pages(
    py: Python<'_>,
    pages: Vec<(String, String)>,
) -> PyResult<Vec<(String, String)>> {
    // Release GIL — Rayon will use multiple threads
    py.detach(|| Ok(fingerprint::batch_fingerprint(&pages)))
}

// ---------------------------------------------------------------------------
// chunker
// ---------------------------------------------------------------------------

#[pyfunction]
fn chunk_markdown(py: Python<'_>, markdown: &str, max_chars: usize) -> PyResult<Vec<Py<PyDict>>> {
    let chunks = chunker::chunk_markdown(markdown, max_chars);
    chunks
        .into_iter()
        .map(|c| {
            let dict = PyDict::new(py);
            dict.set_item("content", c.content)?;
            dict.set_item("section", c.section)?;
            dict.set_item("char_count", c.char_count)?;
            dict.set_item("chunk_index", c.chunk_index)?;
            Ok(dict.unbind())
        })
        .collect()
}

#[pyfunction]
fn batch_chunk_documents(
    py: Python<'_>,
    documents: Vec<(String, String)>,
    max_chars: usize,
) -> PyResult<Vec<(String, Vec<Py<PyDict>>)>> {
    let results = py.detach(|| chunker::batch_chunk(&documents, max_chars));
    results
        .into_iter()
        .map(|(url, chunks)| {
            let py_chunks: PyResult<Vec<Py<PyDict>>> = chunks
                .into_iter()
                .map(|c| {
                    let dict = PyDict::new(py);
                    dict.set_item("content", c.content)?;
                    dict.set_item("section", c.section)?;
                    dict.set_item("char_count", c.char_count)?;
                    dict.set_item("chunk_index", c.chunk_index)?;
                    Ok(dict.unbind())
                })
                .collect();
            Ok((url, py_chunks?))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// url_utils
// ---------------------------------------------------------------------------

#[pyfunction]
fn normalize_url(url: &str) -> PyResult<Option<String>> {
    Ok(url_utils::normalize_url(url))
}

#[pyfunction]
fn normalize_and_dedup(py: Python<'_>, urls: Vec<String>) -> PyResult<Vec<String>> {
    py.detach(|| Ok(url_utils::normalize_and_dedup(&urls)))
}

#[pyfunction]
fn filter_uncrawled(
    py: Python<'_>,
    crawled: Vec<String>,
    candidates: Vec<String>,
) -> PyResult<Vec<String>> {
    py.detach(|| Ok(url_utils::filter_uncrawled(&crawled, &candidates)))
}

// ---------------------------------------------------------------------------
// html_extract
// ---------------------------------------------------------------------------

#[pyfunction]
fn extract_clean_text(html: &str) -> PyResult<String> {
    Ok(html_extract::extract_clean_text(html))
}

#[pyfunction]
fn token_comparison(html: &str, clean_text: &str) -> PyResult<(usize, usize, f64)> {
    Ok(html_extract::token_comparison(html, clean_text))
}

// ---------------------------------------------------------------------------
// module
// ---------------------------------------------------------------------------

#[pymodule]
fn crawlkit_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // fingerprint
    m.add_function(wrap_pyfunction!(content_fingerprint, m)?)?;
    m.add_function(wrap_pyfunction!(content_has_changed, m)?)?;
    m.add_function(wrap_pyfunction!(batch_fingerprint_pages, m)?)?;
    // chunker
    m.add_function(wrap_pyfunction!(chunk_markdown, m)?)?;
    m.add_function(wrap_pyfunction!(batch_chunk_documents, m)?)?;
    // url_utils
    m.add_function(wrap_pyfunction!(normalize_url, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_and_dedup, m)?)?;
    m.add_function(wrap_pyfunction!(filter_uncrawled, m)?)?;
    // html_extract
    m.add_function(wrap_pyfunction!(extract_clean_text, m)?)?;
    m.add_function(wrap_pyfunction!(token_comparison, m)?)?;
    Ok(())
}
