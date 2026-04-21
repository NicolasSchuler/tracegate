import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='7 14\n1 6 3\n1 4 1\n1 5 2\n1 2 7\n1 3 5\n3 2\n3 4\n3 6\n2 3 5\n2 4 1\n1 1 5\n3 2\n3 4\n3 6\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '5 6 3 5 2 7\n2 4 1\n5 6 3 5 2 7\n4 1 5 2 7\n1 4\n2 6 3\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
