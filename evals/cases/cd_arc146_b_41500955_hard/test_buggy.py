import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4 8 2\n1 2 4 8\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '10\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5 345 3\n111 192 421 390 229\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '461\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
