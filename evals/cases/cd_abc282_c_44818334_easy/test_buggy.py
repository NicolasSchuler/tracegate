import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='8\n"a,b"c,d\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '"a,b"c.d\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5\n,,,,,\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '.....\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='20\na,"t,"c,"o,"d,"e,"r,\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'a."t,"c."o,"d."e,"r.\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2\n""\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '""\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\n","\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '","\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
