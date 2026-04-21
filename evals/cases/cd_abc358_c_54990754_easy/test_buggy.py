import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 5\noooxx\nxooox\nxxooo\n',
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
        input='3 2\noo\nox\nxo\n',
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
        input='8 6\nxxoxxo\nxxoxxx\nxoxxxx\nxxxoxx\nxxoooo\nxxxxox\nxoxxox\noxoxxo\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '3\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10 10\nxxoxxxxxxo\noxxoxxxxxx\nxxxxoxxxxx\nxxxxxoxxxx\nxxxoxxxxxx\nxoxxxxxxxx\nxoxxxxoxxx\nxxxxxoxxxx\nxxxxxxxoxx\nxxoxxxxxox\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '7\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10 10\nxxoxxxxxxx\nxxxxxxxxxo\nxxxxxxoxxx\nxxxxxxxxxo\noxxxxxoxxx\nxoxxoxxxxx\nxxxxxxxoxx\nxxxxxoxxxx\nxxoxxxxxxx\nxoooxxxxox\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '6\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
