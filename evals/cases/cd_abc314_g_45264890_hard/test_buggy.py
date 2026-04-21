import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='7 3 7\n3 2\n1 1\n4 2\n1 2\n5 1\n9 3\n2 3\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2 5 7 7\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='15 5 400\n29 5\n27 4\n79 1\n27 2\n30 3\n4 1\n89 2\n88 3\n75 5\n3 1\n39 4\n12 1\n62 4\n38 2\n49 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '8 12 15 15 15 15\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 1 2\n1 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1 1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 1 2\n2 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0 1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 1 1\n3 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0 1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
