import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 1\n6983 1 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '6984 1 1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='99 99\n3314 99 99\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '3315 1 1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='99 99\n2574 1 99\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2574 2 1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='99 99\n3886 99 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '3886 99 2\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='15 25\n5989 15 25\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '5990 1 1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
