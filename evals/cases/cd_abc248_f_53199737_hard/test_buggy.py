import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 998244353\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '7 15\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='16 999999937\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '46 1016 14288 143044 1079816 6349672 29622112 110569766 330377828 784245480 453609503 38603306 44981526 314279703 408855776\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 999999929\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '4\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4 999999893\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '10 38 56\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='17 999999883\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '49 1156 17424 187760 1533652 9822440 50321752 208458358 700809270 906581749 155877980 120864279 280952473 688186176 227413361 525870646\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
