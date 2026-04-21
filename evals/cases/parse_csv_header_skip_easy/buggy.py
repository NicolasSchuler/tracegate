import csv
import io


def parse_scores(raw_csv: str) -> list[int]:
    reader = csv.reader(io.StringIO(raw_csv))
    rows = list(reader)
    data_rows = rows[2:]  # skip header
    return [int(row[1]) for row in data_rows]
