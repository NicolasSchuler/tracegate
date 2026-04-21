import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5\n3 1 4 1 5\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '4 5 13 14 26 \n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='6\n1000000000 1000000000 1000000000 1000000000 1000000000 1000000000\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1000000001 2000000001 3000000001 4000000001 5000000001 6000000001 \n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='15\n748 169 586 329 972 529 432 519 408 587 138 249 656 114 632\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '749 918 1921 2250 4861 5390 5822 6428 6836 7796 7934 8294 10109 10223 11373 \n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1\n1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2 \n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1\n1000000000\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1000000001 \n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
