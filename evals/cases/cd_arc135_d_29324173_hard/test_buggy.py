import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 3\n1 2 3\n4 5 6\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '9\n0 -3 -1\n3 0 2\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 2\n1000000000 -1000000000\n-1000000000 1000000000\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '4000000000\n2000000000 0\n0 2000000000\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 4\n0 2 0 -2\n-3 -1 2 0\n-3 -3 2 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0\n0 0 0 0\n0 0 0 0\n0 0 0 0\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 2\n131916715 271857987\n-258940184 -801308683\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1464023569\n390856899 530798171\n0 -542368499\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 3\n-923681301 998024714 359349656\n296410309 264292840 -238119906\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2551293046\n-1220091610 733731874 391467125\n0 0 -206002437\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
