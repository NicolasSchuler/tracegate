import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='9\n4 2 1\n4 2 2\n4 2 3\n4 2 4\n5 8 1\n5 8 2\n5 8 3\n5 8 4\n5 8 5\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0\n2\n1\n3\n0\n3\n1\n4\n2\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
