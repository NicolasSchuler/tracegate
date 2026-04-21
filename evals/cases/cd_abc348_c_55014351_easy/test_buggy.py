import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4\n100 1\n20 5\n30 5\n40 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '40\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10\n68 3\n17 2\n99 2\n92 4\n82 4\n10 3\n100 2\n78 1\n3 1\n35 4\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '35\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
