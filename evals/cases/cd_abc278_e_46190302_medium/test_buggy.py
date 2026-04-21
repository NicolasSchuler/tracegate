import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 4 5 2 2\n2 2 1 1\n3 2 5 3\n3 4 4 3\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '4 4 3\n5 3 4\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5 6 9 3 4\n7 1 5 3 9 5\n4 5 4 5 1 2\n6 1 6 2 9 7\n4 7 1 5 8 8\n3 4 3 3 5 3\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '8 8 7\n8 9 7\n8 9 8\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='9 12 30 4 7\n2 2 2 2 2 2 2 2 2 2 2 2\n2 2 20 20 2 2 5 9 10 9 9 23\n2 29 29 29 29 29 28 28 26 26 26 15\n2 29 29 29 29 29 25 25 26 26 26 15\n2 29 29 29 29 29 25 25 8 25 15 15\n2 18 18 18 18 1 27 27 25 25 16 16\n2 19 22 1 1 1 7 3 7 7 7 7\n2 19 22 22 6 6 21 21 21 7 7 7\n2 19 22 22 22 22 21 21 21 24 24 24\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '21 20 19 20 18 17\n20 19 18 19 17 15\n21 19 20 19 18 16\n21 19 19 18 19 18\n20 18 18 18 19 18\n18 16 17 18 19 17\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 2 300 1 1\n62 106\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1 1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5 5 300 1 1\n16 226 42 184 227\n80 152 191 138 228\n102 39 7 75 116\n99 86 9 26 127\n68 105 43 236 201\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '24 24 24 24 24\n24 24 24 24 24\n24 24 24 24 24\n24 24 24 24 24\n24 24 24 24 24\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
