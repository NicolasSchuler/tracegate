import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5 3\n1 2\n1 3\n4 5\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5 0\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '5\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4 6\n1 2\n1 3\n1 4\n2 3\n2 4\n3 4\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='36 30\n7 22\n28 34\n6 19\n5 6\n6 23\n12 33\n3 27\n6 22\n22 30\n1 36\n30 33\n3 30\n19 25\n17 22\n19 32\n21 32\n20 30\n19 29\n9 11\n9 21\n9 24\n3 7\n10 24\n4 14\n16 36\n10 11\n3 19\n11 28\n8 13\n25 31\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '9\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='78 35\n21 44\n53 73\n23 46\n23 36\n56 66\n6 18\n17 58\n66 76\n7 56\n6 15\n62 64\n9 34\n37 70\n36 44\n10 31\n1 39\n16 19\n11 26\n64 70\n43 61\n23 55\n12 47\n44 46\n16 17\n16 51\n41 73\n4 7\n6 46\n13 54\n13 30\n10 26\n18 25\n6 59\n24 25\n35 36\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '44\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
