use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct Chunk {
    pub content: String,
    pub section: String,
    pub char_count: usize,
    pub chunk_index: usize,
}

/// Split a markdown document into semantic chunks at header boundaries.
/// If a section exceeds max_chars, it is split further at paragraph boundaries.
pub fn chunk_markdown(markdown: &str, max_chars: usize) -> Vec<Chunk> {
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut current_section = String::from("root");
    let mut current_content = String::new();
    let mut chunk_index = 0usize;

    let flush = |section: &str,
                 content: &str,
                 max_chars: usize,
                 chunks: &mut Vec<Chunk>,
                 idx: &mut usize| {
        let trimmed = content.trim();
        if trimmed.is_empty() {
            return;
        }
        if trimmed.len() <= max_chars {
            chunks.push(Chunk {
                content: trimmed.to_string(),
                section: section.to_string(),
                char_count: trimmed.len(),
                chunk_index: *idx,
            });
            *idx += 1;
        } else {
            // Oversized: split at paragraph boundaries
            for sub in split_by_paragraphs(trimmed, max_chars) {
                chunks.push(Chunk {
                    char_count: sub.len(),
                    content: sub,
                    section: section.to_string(),
                    chunk_index: *idx,
                });
                *idx += 1;
            }
        }
    };

    for line in markdown.lines() {
        if line.starts_with('#') {
            flush(
                &current_section,
                &current_content,
                max_chars,
                &mut chunks,
                &mut chunk_index,
            );
            current_content.clear();
            // Extract section title: strip all leading '#' and trim
            current_section = line.trim_start_matches('#').trim().to_string();
        } else {
            current_content.push_str(line);
            current_content.push('\n');
        }
    }

    // Flush the final section
    flush(
        &current_section,
        &current_content,
        max_chars,
        &mut chunks,
        &mut chunk_index,
    );

    chunks
}

fn split_by_paragraphs(content: &str, max_chars: usize) -> Vec<String> {
    let mut result: Vec<String> = Vec::new();
    let mut current = String::new();

    for paragraph in content.split("\n\n") {
        let para = paragraph.trim();
        if para.is_empty() {
            continue;
        }
        // If adding this paragraph would exceed the limit, flush first
        if !current.is_empty() && current.len() + para.len() + 2 > max_chars {
            result.push(current.trim().to_string());
            current.clear();
        }
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(para);
    }

    if !current.trim().is_empty() {
        result.push(current.trim().to_string());
    }

    if result.is_empty() {
        // Single paragraph larger than max_chars — return as-is (no sensible split point)
        result.push(content.trim().to_string());
    }

    result
}

/// Chunk a batch of (url, markdown) pairs in parallel.
pub fn batch_chunk(documents: &[(String, String)], max_chars: usize) -> Vec<(String, Vec<Chunk>)> {
    documents
        .par_iter()
        .map(|(url, content)| (url.clone(), chunk_markdown(content, max_chars)))
        .collect()
}
