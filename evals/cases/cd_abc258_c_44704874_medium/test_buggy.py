import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 3\nabc\n2 2\n1 1\n2 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'b\na\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10 8\ndsuccxulnl\n2 4\n2 7\n1 2\n2 7\n1 1\n1 2\n1 3\n2 5\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'c\nu\nc\nu\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
