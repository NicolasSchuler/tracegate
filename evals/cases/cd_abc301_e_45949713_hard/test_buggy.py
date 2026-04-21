import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 3 5\nS.G\no#o\n.#.\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 3 1\nS.G\n.#o\no#.\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '-1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5 10 2000000\nS.o..ooo..\n..o..o.o..\n..o..ooo..\n..o..o.o..\n..o..ooo.G\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '18\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 292 279\n.....................oS..............o...o.o........o............o.......o.......o...................o.......G.......o..o............o........................................................o.............o..................o.....o...................................o.........................o\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '14\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 298 391\n.o...............o...............o..................o....o....................o.............o..............................o...............o...............S...........................o.....o..................................G.....o...o....................o.............oo..............o.o..........\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '14\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
