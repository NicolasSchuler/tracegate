from buggy import parse_scores

CSV_DATA = """\
name,score
Alice,95
Bob,87
Carol,92
"""


def test_parse_all_data_rows() -> None:
    scores = parse_scores(CSV_DATA)
    assert scores == [95, 87, 92]
