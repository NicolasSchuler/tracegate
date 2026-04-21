import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5\nTTAAT\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'T\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='6\nATTATA\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'T\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1\nA\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'A\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='91\nAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'A\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='92\nAATAAAAAAATATAAAAAAAAATAAAAAAATAAATAAAAAAAAAAAAAAAAATAAAAAAAAAATAAAAAATAAAATAAAATAAAAAATATAA\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'A\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
