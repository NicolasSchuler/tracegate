import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='6\n3 ABC ABC\n1 C B\n1 B C\n2 AB BA\n2 BA AB\n3 CCB ABA\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Yes\nYes\nNo\nYes\nNo\nYes\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='7\n5 ABABA BABAB\n5 ABCBC BBABA\n5 CCCCC CBABC\n5 BBAAA AAABB\n5 AAABB BBAAA\n5 ACACB BAACB\n5 ACACB BBACA\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'No\nYes\nYes\nNo\nYes\nYes\nNo\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
