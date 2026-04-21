import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10 3 5\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '22\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1000000000 314 159\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '495273003954006262\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='158260522 877914575 602436426\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '12523196490986503\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='24979445 861648772 623690081\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '311986348743735\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='433933447 476190629 262703497\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '94149118166914131\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
