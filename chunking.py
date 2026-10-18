def chunk_text(text: str, *, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    """Split text into overlapping chunks of `chunk_size` words."""
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})")

    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(words):
        chunks.append(" ".join(words[start : start + chunk_size]))
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks