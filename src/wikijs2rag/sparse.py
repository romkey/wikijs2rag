"""Keyword (sparse) vectors, BM25-style, computed without a model.

Scheme ``wikijs2rag-bm25-crc32-v1``, recorded in collection metadata so any
client can build matching query vectors:

- Tokens: lowercase the text, take every run of ``[a-z0-9]`` (Unicode
  letters and digits count), drop tokens of one character.
- Index: ``zlib.crc32(token.encode("utf-8"))`` (0 to 2**32-1).
- Document value: BM25 term-frequency saturation with k1=1.2 and no length
  normalization, ``tf * 2.2 / (tf + 1.2)``.
- Query value: 1.0 per distinct token.

The collection's sparse vector uses Qdrant's IDF modifier, so Qdrant
supplies the inverse document frequency.
"""

import re
import zlib
from collections import Counter

from qdrant_client.models import SparseVector

SCHEME = "wikijs2rag-bm25-crc32-v1"
_K1 = 1.2
_WORD = re.compile(r"[^\W_]+")


def tokens(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if len(t) > 1]


def _vector(weights: dict[int, float]) -> SparseVector:
    indices = sorted(weights)
    return SparseVector(indices=indices, values=[weights[i] for i in indices])


def document_vector(text: str) -> SparseVector:
    weights: dict[int, float] = {}
    for token, tf in Counter(tokens(text)).items():
        index = zlib.crc32(token.encode())
        weights[index] = weights.get(index, 0.0) + tf * (_K1 + 1) / (tf + _K1)
    return _vector(weights)


def query_vector(text: str) -> SparseVector:
    return _vector({zlib.crc32(t.encode()): 1.0 for t in set(tokens(text))})
