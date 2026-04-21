import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 3 4\n2 3 4\n4 1 3\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 2 3\n0 0\n0 0\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 2 1\n8316250877914272 9547116602436248\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1230865724521976\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 3 1\n5408917861648238 4859962623689873 2509054433932721\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1960099195974356\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 1 1\n9845426262703405\n5335723211046443\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '4509703051656962\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
