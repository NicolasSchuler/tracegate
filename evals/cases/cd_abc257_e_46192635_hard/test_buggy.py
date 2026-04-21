import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5\n5 4 3 3 2 5 3 5 3\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '95\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='20\n1 1 1 1 1 1 1 1 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '99999999999999999999\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='260522\n260522 260522 260522 260522 260522 260522 190629 260522 47202\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '99999\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='407775\n407775 407775 407775 407775 302156 407775 407775 407775 78537\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '99999\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='723857\n330739 286918 329211 539679 238506 63340 686568 361868 660016\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '66666666666\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
