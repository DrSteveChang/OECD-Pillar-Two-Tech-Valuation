import json

from oecd_pillar_two.rag.chunker import chunk_json_entries


def test_chunk_json_entries_groups_by_model_id():
    chunks = chunk_json_entries(
        [
            {"model_id": "did", "estimate": 0.1},
            {"model_id": "did", "estimate": 0.2},
            {"model_id": "scm", "estimate": -0.1},
        ]
    )

    assert [chunk["section"] for chunk in chunks] == ["model_id=did", "model_id=scm"]
    did_rows = [json.loads(line) for line in chunks[0]["text"].splitlines()]
    assert did_rows == [
        {"model_id": "did", "estimate": 0.1},
        {"model_id": "did", "estimate": 0.2},
    ]


def test_chunk_json_entries_chunks_ungrouped_entries():
    chunks = chunk_json_entries(
        [{"name": "row1"}, {"name": "row2"}, {"name": "row3"}],
        group_size=2,
    )

    assert [chunk["section"] for chunk in chunks] == ["entries_1_2", "entries_3_3"]
    assert chunks[0]["chunk_index"] == 0
    assert chunks[1]["chunk_index"] == 1
