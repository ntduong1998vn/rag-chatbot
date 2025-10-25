import re
from typing import List

JP_SPLIT = re.compile(r"(?<=[。！？\?])\s*")


def split_japanese(text: str, max_chars: int = 800, overlap: int = 100) -> List[str]:
    sents = [s for s in JP_SPLIT.split(text) if s.strip()]
    chunks, buf = [], ""
    for s in sents:
        if len(buf) + len(s) <= max_chars:
            buf += s if not buf else s
        else:
            if buf:
                chunks.append(buf.strip())
            # tạo overlap ký tự cuối
            if overlap > 0 and len(buf) > overlap:
                buf = buf[-overlap:] + s
            else:
                buf = s
    if buf.strip():
        chunks.append(buf.strip())
    return chunks
