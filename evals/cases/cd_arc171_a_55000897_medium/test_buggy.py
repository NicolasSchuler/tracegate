import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='8\n5 2 3\n6 5 8\n3 2 2\n11 67 40\n26 22 16\n95 91 31\n80 46 56\n998 2 44353\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Yes\nNo\nNo\nNo\nYes\nNo\nYes\nYes\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
