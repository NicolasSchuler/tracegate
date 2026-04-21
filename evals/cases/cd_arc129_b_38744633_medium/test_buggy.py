import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\n1 3\n2 4\n5 6\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0\n0\n1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10\n64 96\n30 78\n52 61\n18 28\n9 34\n42 86\n11 49\n1 79\n13 59\n70 95\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0\n0\n2\n18\n18\n18\n18\n18\n18\n21\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1\n663994021 848529561\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
