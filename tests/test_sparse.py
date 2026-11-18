import zlib

from wikijs2rag import sparse


def test_tokens() -> None:
    assert sparse.tokens("The Laser-Cutter, a 3D printer! é_x") == ["the", "laser", "cutter", "3d", "printer"]


def test_document_vector_saturates_term_frequency() -> None:
    vec = sparse.document_vector("laser laser laser cutter")
    weights = dict(zip(vec.indices, vec.values, strict=True))
    laser, cutter = zlib.crc32(b"laser"), zlib.crc32(b"cutter")
    assert vec.indices == sorted(vec.indices)
    assert weights[cutter] == 1.0
    assert 1.0 < weights[laser] < 2.2


def test_query_vector_is_binary() -> None:
    vec = sparse.query_vector("laser laser cutter")
    assert sorted(vec.values) == [1.0, 1.0]
