import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4 5\nabcde\nfghij\nklmno\npqrst\n1\n3 3\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'mlkon\nhgfji\ncbaed\nrqpts\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 7\natcoder\nregular\ncontest\n2\n1 1\n2 5\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'testcon\noderatc\nularreg\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 2\nac\nwa\n3\n1 1\n1 1\n1 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'ac\nwa\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 2\nwa\ncy\n8\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'wa\ncy\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 3\npph\nejq\n3\n1 1\n1 1\n1 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'pph\njeq\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
