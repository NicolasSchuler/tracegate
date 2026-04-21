import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='ABC\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'A B C\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='ZZZZZZZ\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Z Z Z Z Z Z Z\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='OOXXOO\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'O O X X O O\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='KRQCVOYGNLPMX\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'K R Q C V O Y G N L P M X\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='IFUOVDHHKWPVHHADUIROYXKAKPWHTFDUBBGDUCK\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'I F U O V D H H K W P V H H A D U I R O Y X K A K P W H T F D U B B G D U C K\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
