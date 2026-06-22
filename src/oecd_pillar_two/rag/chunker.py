# chunker.py
# Semantic paragraph-boundary chunking for RAG corpus.
# Replaces fixed-line/page-based chunking with paragraph-aware splitting.

from __future__ import annotations

import json
import re


_MIN_CHUNK_CHARS = 200
_MAX_CHUNK_CHARS = 2000
_PARAGRAPH_PATTERN = re.compile(r"\n{2,}")
_SECTION_PATTERN = re.compile(
    r"^(#+ .+|(?:Chapter|Section|Part|Appendix)\s+\d+[\.:]?\s+.+)",
    re.IGNORECASE | re.MULTILINE,
)


def chunk_text(text: str, *, page: int = 0, source: str = "") -> list[dict]:
    """Split long text into semantic chunks at paragraph boundaries.

    Returns a list of dicts with keys: text, page, section, chunk_index.
    """
    if not text or not text.strip():
        return []

    # Detect section markers
    sections = list(_SECTION_PATTERN.finditer(text))
    if not sections:
        # Fall back to paragraph splitting
        return _chunk_paragraphs(text, page=page, source=source)

    chunks = []
    for i, match in enumerate(sections):
        section_title = match.group(0).strip()
        start = match.end()
        end = sections[i + 1].start() if i + 1 < len(sections) else len(text)
        section_text = text[start:end].strip()

        if len(section_text) <= _MAX_CHUNK_CHARS:
            chunks.append({
                "text": f"{section_title}\n\n{section_text}",
                "page": page,
                "section": section_title,
                "chunk_index": i,
            })
        else:
            subs = _chunk_paragraphs(section_text, page=page, source=source)
            for sub in subs:
                sub["section"] = section_title
                chunks.append(sub)

    return chunks


def _chunk_paragraphs(text: str, *, page: int = 0, source: str = "") -> list[dict]:
    """Split text at double-newline boundaries, respecting chunk size limits."""
    paragraphs = _PARAGRAPH_PATTERN.split(text.strip())
    # Filter out very short fragments
    paragraphs = [p.strip() for p in paragraphs if len(p.strip()) >= 20]

    if not paragraphs:
        return []

    chunks = []
    current = ""
    chunk_index = 0
    context_before = ""
    context_after = ""

    for i, para in enumerate(paragraphs):
        candidate = f"{current}\n\n{para}".strip() if current else para

        if len(candidate) > _MAX_CHUNK_CHARS and current:
            # Finalize current chunk
            context_after = paragraphs[i + 1][:200] if i + 1 < len(paragraphs) else ""
            chunks.append({
                "text": current,
                "page": page,
                "section": "",
                "chunk_index": chunk_index,
                "context_before": context_before[:200],
                "context_after": context_after[:200],
            })
            chunk_index += 1
            context_before = current[-200:]
            current = para
        else:
            current = candidate

        if len(current) < _MIN_CHUNK_CHARS and i < len(paragraphs) - 1:
            continue

    # Final chunk
    if current.strip():
        chunks.append({
            "text": current,
            "page": page,
            "section": "",
            "chunk_index": chunk_index,
            "context_before": context_before[:200],
            "context_after": "",
        })

    return chunks


def chunk_dataframe_rows(df, *, chunk_size: int = 40, source: str = "") -> list[dict]:
    """Split a DataFrame into logical row groups for RAG indexing.

    Preserves column headers in each chunk for self-documentation.
    """
    chunks = []
    columns = list(df.columns)
    header = ",".join(columns)

    for start in range(0, len(df), chunk_size):
        end = min(start + chunk_size, len(df))
        subset = df.iloc[start:end]
        text = f"{header}\n{subset.to_csv(index=False)}"
        chunks.append({
            "text": text,
            "section": f"rows_{start + 1}_{end}",
            "chunk_index": start // chunk_size,
        })
    return chunks


def chunk_json_entries(entries: list[dict], *, group_size: int = 10, source: str = "") -> list[dict]:
    """Split a list of JSON dicts into semantic groups.

    Groups entries by their natural keys when available (event_id, model_id, etc.).
    """
    if not entries:
        return []

    # Attempt to group by common grouping keys
    group_keys = ["event_id", "model_id", "specification", "category"]
    group_key = None
    for key in group_keys:
        if key in entries[0]:
            group_key = key
            break

    chunks = []
    if group_key:
        from collections import defaultdict
        grouped = defaultdict(list)
        for entry in entries:
            grouped[entry.get(group_key, "")].append(entry)
        for key, group in sorted(grouped.items()):
            text = "\n".join(json.dumps(item, ensure_ascii=False) for item in group)
            chunks.append({
                "text": text,
                "section": f"{group_key}={key}",
                "chunk_index": len(chunks),
            })
    else:
        for start in range(0, len(entries), group_size):
            end = min(start + group_size, len(entries))
            group = entries[start:end]
            text = "\n".join(json.dumps(item, ensure_ascii=False) for item in group)
            chunks.append({
                "text": text,
                "section": f"entries_{start + 1}_{end}",
                "chunk_index": start // group_size,
            })

    return chunks
