import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4\n5 3\n17 2\n163 79\n1000000000000000000 1000000000000000000\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Yes\nNo\nYes\nYes\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
