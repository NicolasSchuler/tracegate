import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='Takahashi Chokudai\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Takahashi san\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='K Eyence\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'K san\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='V G\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'V san\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='Gpu A\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Gpu san\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='Mkxkszhkbp P\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Mkxkszhkbp san\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
