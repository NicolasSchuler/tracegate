import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='TOYOTA\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '5\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='ABCDEFG\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='AAAAAAAAAA\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '10\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='GGGGAGAAGAAAAGAAGAGGAGAAGAAAAAAAGAAGAGGGAAGAAGAAAGGGGGAGAAAGGGAAGGGAGAAGAAAGGAGAGAAGGGAGAAAAAAAAGGGG\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '21\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='OOOOOOJOOJJOJJOOOOOOJJJOOOOOOOOJOOJJJJJOJOJJJJJJJJOOJJJOJJJOOJJOJOOJOOJJOOOJOJOJOOJJJJJJJOOJOOOJOOJO\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '15\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
