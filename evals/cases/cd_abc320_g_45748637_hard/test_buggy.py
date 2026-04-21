import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='100 1\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n8\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '99\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 1\n1\n',
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
        input='3 6\n012543\n102354\n012345\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '5\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 10\n1937458062\n8124690357\n2385760149\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '6\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10 20\n01234567890123456789\n01234567890123456789\n01234567890123456789\n01234567890123456789\n01234567890123456789\n01234567890123456789\n01234567890123456789\n01234567890123456789\n01234567890123456789\n01234567890123456789\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '90\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
