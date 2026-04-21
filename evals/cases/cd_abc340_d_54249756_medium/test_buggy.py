import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2\n1 1 1\n',
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
        input='100\n536870914 1 51\n268435458 1 52\n134217730 1 53\n67108866 1 54\n33554434 1 55\n16777218 1 56\n8388610 1 57\n4194306 1 58\n2097154 1 59\n1048578 1 60\n524290 1 61\n262146 1 62\n131074 1 63\n65538 1 64\n32770 1 65\n16386 1 66\n8194 1 67\n4098 1 68\n2050 1 69\n1026 1 70\n514 1 71\n258 1 72\n130 1 73\n66 1 74\n34 1 75\n18 1 76\n10 1 77\n6 1 78\n4 1 79\n3 1 80\n3 1 81\n3 1 82\n3 1 83\n3 1 84\n3 1 85\n3 1 86\n3 1 87\n3 1 88\n3 1 89\n3 1 90\n3 1 91\n3 1 92\n3 1 93\n3 1 94\n3 1 95\n3 1 96\n3 1 97\n3 1 98\n3 1 99\n3 1 100\n1000000000 1 2\n1000000000 1 3\n1000000000 1 4\n1000000000 1 5\n1000000000 1 6\n1000000000 1 7\n1000000000 1 8\n1000000000 1 9\n1000000000 1 10\n1000000000 1 11\n1000000000 1 12\n1000000000 1 13\n1000000000 1 14\n1000000000 1 15\n1000000000 1 16\n1000000000 1 17\n1000000000 1 18\n1000000000 1 19\n1000000000 1 20\n1000000000 1 21\n1000000000 1 22\n1000000000 1 23\n1000000000 1 24\n1000000000 1 25\n1000000000 1 26\n1000000000 1 27\n1000000000 1 28\n1000000000 1 29\n1000000000 1 30\n1000000000 1 31\n1000000000 1 32\n1000000000 1 33\n1000000000 1 34\n1000000000 1 35\n1000000000 1 36\n1000000000 1 37\n1000000000 1 38\n1000000000 1 39\n1000000000 1 40\n1000000000 1 41\n1000000000 1 42\n1000000000 1 43\n1000000000 1 44\n1000000000 1 45\n1000000000 1 46\n1000000000 1 47\n1000000000 1 48\n1000000000 1 49\n1000000000 1 50\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '99\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5\n100 200 3\n50 10 1\n100 200 5\n150 1 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '350\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10\n1000 10 9\n1000 10 10\n1000 10 2\n1000 10 3\n1000 10 4\n1000 10 5\n1000 10 6\n1000 10 7\n1000 10 8\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '90\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='6\n1000000000 1000000000 1\n1000000000 1000000000 1\n1000000000 1000000000 1\n1000000000 1000000000 1\n1000000000 1000000000 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '5000000000\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
