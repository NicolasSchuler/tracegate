import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='7 3\n1 2 1 3 3 3\n1 1\n1 2\n4 3\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '4\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10 10\n1 1 3 1 2 3 3 5 7\n2 1\n5 1\n4 3\n6 3\n2 1\n7 3\n9 2\n1 2\n6 2\n8 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '10\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='16 5\n1 1 2 1 1 4 1 4 6 10 6 6 10 2 6\n13 10\n2 5\n15 7\n11 4\n3 4\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '8\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='35 42\n1 1 3 1 5 3 4 5 4 2 3 8 10 3 9 15 12 2 14 8 6 1 8 20 10 22 23 22 2 9 25 19 17 21\n10 7\n22 5\n7 7\n28 10\n21 4\n33 10\n16 1\n18 9\n23 2\n9 2\n30 5\n19 4\n2 9\n17 5\n20 10\n23 9\n7 10\n27 9\n23 1\n25 6\n8 9\n25 1\n10 5\n13 8\n27 5\n3 2\n35 1\n24 8\n14 4\n13 4\n35 6\n24 2\n7 6\n13 3\n17 7\n35 1\n26 5\n13 5\n5 4\n35 2\n32 3\n20 3\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '34\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='19 34\n1 2 1 3 5 5 3 3 2 1 5 3 13 8 8 11 8 16\n14 6\n19 1\n13 9\n2 10\n18 8\n3 7\n16 3\n4 7\n7 6\n11 7\n7 9\n14 1\n19 4\n3 1\n3 8\n10 5\n14 5\n6 1\n12 7\n3 10\n7 4\n18 4\n17 9\n19 4\n11 7\n14 7\n15 9\n11 5\n9 2\n3 4\n19 1\n16 8\n11 2\n10 7\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '18\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
