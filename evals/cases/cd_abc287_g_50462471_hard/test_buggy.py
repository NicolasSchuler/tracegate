import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\n1 1\n2 2\n3 3\n7\n3 4\n1 1 10\n3 4\n2 1 0\n2 3 0\n3 4\n3 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '11\n19\n-1\n4\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
