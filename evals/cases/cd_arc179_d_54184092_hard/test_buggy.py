import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4\n1 2\n1 3\n1 4\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '3\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10\n1 7\n7 10\n10 8\n8 3\n8 4\n10 9\n9 6\n9 5\n7 2\n',
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
        input='2\n1 2\n',
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
        input='10\n1 2\n1 3\n1 4\n2 5\n2 6\n3 7\n3 8\n4 9\n4 10\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '10\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='63\n56 24\n61 37\n41 61\n32 7\n60 5\n15 45\n5 29\n3 33\n15 23\n12 4\n17 39\n23 22\n58 57\n38 58\n40 54\n8 3\n18 16\n26 19\n49 32\n28 47\n17 12\n11 51\n24 4\n63 17\n36 59\n35 14\n32 27\n9 7\n62 4\n12 45\n46 9\n14 30\n34 28\n13 5\n34 2\n53 19\n54 62\n32 35\n9 31\n37 19\n42 56\n19 39\n29 28\n48 22\n19 16\n44 19\n6 31\n33 60\n10 63\n26 51\n50 51\n31 55\n52 43\n51 57\n8 59\n20 17\n25 37\n52 19\n6 10\n30 60\n21 44\n1 20\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '65\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
