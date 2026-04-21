import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\nabc\nabb\naac\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2\n2\n1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='11\nabracadabra\nbracadabra\nracadabra\nacadabra\ncadabra\nadabra\ndabra\nabra\nbra\nra\na\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '4\n3\n2\n1\n0\n1\n0\n4\n3\n2\n1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
