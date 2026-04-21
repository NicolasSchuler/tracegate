import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='atcoder\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'tcdr\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='xyz\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'xyz\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='aaaabbbbcccc\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'bbbbcccc\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='q\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'q\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='xxihohrpmlovamyuwtmnpsvzhrqdnqjimlcttat\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'xxhhrpmlvmywtmnpsvzhrqdnqjmlcttt\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
