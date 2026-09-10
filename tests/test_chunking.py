from documents import chunk_page_text, deterministic_chunk_id


def test_page_chunking_uses_deterministic_750_100_character_windows():
    text = "a" * 750 + "b" * 650

    chunks = chunk_page_text(text)

    assert chunks == ("a" * 750, "a" * 100 + "b" * 650)
    assert all(len(chunk) == 750 for chunk in chunks)


def test_chunk_ids_are_stable_and_include_page_source_identity():
    inputs = {
        "paper_id": "2025.test.1",
        "pdf_sha256": "a" * 64,
        "page_number": 2,
        "chunk_index": 0,
        "text": "deterministic text",
    }

    first = deterministic_chunk_id(**inputs)

    assert first == deterministic_chunk_id(**inputs)
    assert first != deterministic_chunk_id(**{**inputs, "page_number": 3})
    assert first != deterministic_chunk_id(**{**inputs, "pdf_sha256": "b" * 64})
    assert first != deterministic_chunk_id(**{**inputs, "text": "changed text"})
