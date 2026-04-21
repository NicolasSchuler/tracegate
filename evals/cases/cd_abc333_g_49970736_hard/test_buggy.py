import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='0.333\n33\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1 3\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='0.45\n5\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2 5\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='0.314159265358979323\n10000\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '71 226\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='0.007735339533561113\n7203576162\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '34928144 4515398949\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='0.261500529602049246\n5409254538\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1377033204 5265890689\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
