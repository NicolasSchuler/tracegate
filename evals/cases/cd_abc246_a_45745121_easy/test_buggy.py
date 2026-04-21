import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='-1 -1\n-1 2\n3 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '3 -1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='-60 -40\n-60 -80\n-20 -80\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '-20 -40\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='-100 100\n100 100\n100 -100\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '-100 -100\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='-100 -100\n100 100\n100 -100\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '-100 100\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='-100 -100\n100 100\n-100 100\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '100 -100\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
