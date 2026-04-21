import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='aaba\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'aaab\nbaaa\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='z\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'z\nz\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='abracadabra\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'aabracadabr\nracadabraab\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='aabbccddee\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'aabbccddee\neeaabbccdd\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='abbccddeea\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'aabbccddee\neeaabbccdd\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
