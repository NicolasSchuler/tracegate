import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4 5\n1 2 3 4 5\n6 7 8 9 10\n11 12 13 14 15\n16 17 18 19 20\n1 3 2 5 4\n11 13 12 15 14\n6 8 7 10 9\n16 18 17 20 19\n',
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
        input='2 2\n1 1\n1 1\n1 1\n1 1000000000\n',
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
        input='3 3\n8 1 6\n3 5 7\n4 9 2\n8 1 6\n3 5 7\n4 9 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5 5\n710511029 136397527 763027379 644706927 447672230\n979861204 57882493 442931589 951053644 152300688\n43971370 126515475 962139996 541282303 834022578\n312523039 506696497 664922712 414720753 304621362\n325269832 191410838 286751784 732741849 806602693\n806602693 732741849 286751784 191410838 325269832\n304621362 414720753 664922712 506696497 312523039\n834022578 541282303 962139996 126515475 43971370\n152300688 951053644 442931589 57882493 979861204\n447672230 644706927 763027379 136397527 710511029\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '20\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 2\n2 1\n1 2\n1 2\n2 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
