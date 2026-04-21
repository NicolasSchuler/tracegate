import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='8 4 3\n1 2\n3 2\n2 3\n3 1\n1 4\n2 1\n3 3\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '4\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10000000000 1 1\n1 10000000000\n1 10000000000\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '10000000000\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1000 4 7\n19 79\n33 463\n19 178\n33 280\n19 255\n33 92\n34 25\n19 96\n12 11\n19 490\n33 31\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '380\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 1 1\n593477330 1\n294073833 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 1 1\n983778696 1\n983778696 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
