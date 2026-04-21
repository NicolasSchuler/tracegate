import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5\n30 50 70 20 60\nNYYNN\nNNYNN\nNNNYY\nYNNNN\nYNNNN\n3\n1 3\n3 1\n4 5\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1 100\n2 160\n3 180\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2\n100 100\nNN\nNN\n1\n1 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Impossible\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2\n1 1\nNY\nYN\n2\n1 2\n2 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1 2\n1 2\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
