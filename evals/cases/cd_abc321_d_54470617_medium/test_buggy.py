import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 2 7\n3 5\n6 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '24\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 3 2\n1\n1 1 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '6\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='7 12 25514963\n2436426 24979445 61648772 23690081 33933447 76190629 62703497\n11047202 71407775 28894325 31963982 22804784 50968417 30302156 82631932 61735902 80895728 23078537 7723857\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2115597124\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 1 166048436\n92686568\n73361868\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '166048436\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 1 1\n1\n1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
