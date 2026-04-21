import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='7\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'ooxooxo\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='9\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'ooxooxoox\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='17\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'ooxooxooxooxooxoo\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='28\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'ooxooxooxooxooxooxooxooxooxo\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='21\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'ooxooxooxooxooxooxoox\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
