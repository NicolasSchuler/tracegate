import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5\n3 1 4 1 5\n4\n1 5 1\n2 4 3\n1 5 2\n1 3 3\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2\n0\n0\n1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
