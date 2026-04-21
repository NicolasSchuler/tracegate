import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 9\n3 1\n3 2\n1 2\n2 1\n3 1\n3 2\n1 2\n3 2\n3 3\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'No\nNo\nYes\nNo\nYes\nNo\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5 11\n3 1\n1 2\n3 2\n1 3\n1 3\n3 3\n1 4\n2 4\n3 4\n1 5\n3 5\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'No\nNo\nYes\nYes\nNo\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='100 100\n3 28\n3 7\n3 40\n2 52\n2 38\n3 53\n1 79\n2 93\n2 73\n3 59\n2 98\n2 49\n2 57\n3 19\n1 30\n3 7\n2 71\n1 54\n2 2\n2 54\n1 46\n3 12\n1 32\n2 34\n3 24\n1 40\n2 88\n1 75\n1 22\n3 23\n3 25\n2 72\n3 75\n3 35\n3 47\n2 6\n1 13\n1 53\n2 55\n1 100\n1 24\n3 99\n2 64\n3 68\n2 68\n1 37\n1 56\n3 6\n3 15\n1 96\n1 32\n1 77\n1 8\n2 47\n1 9\n3 14\n1 92\n2 43\n1 5\n1 95\n3 82\n1 90\n3 1\n3 19\n2 95\n2 9\n2 85\n1 17\n3 10\n3 39\n1 97\n3 77\n2 90\n2 84\n3 30\n3 43\n1 28\n3 16\n2 83\n1 56\n1 97\n3 72\n3 51\n1 42\n2 3\n3 47\n1 37\n2 16\n3 33\n3 51\n3 33\n3 9\n3 21\n2 35\n2 86\n1 23\n3 39\n3 30\n2 17\n1 23\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'No\nNo\nNo\nNo\nNo\nNo\nNo\nNo\nNo\nNo\nNo\nNo\nNo\nNo\nNo\nNo\nYes\nNo\nNo\nNo\nNo\nNo\nNo\nNo\nNo\nNo\nYes\nNo\nYes\nNo\nYes\nNo\nNo\nNo\nYes\nNo\nNo\nNo\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='100 100\n1 75\n1 4\n2 3\n3 10\n1 72\n2 30\n3 86\n3 34\n2 28\n3 13\n3 67\n2 67\n3 19\n2 86\n3 27\n2 4\n3 20\n3 12\n3 52\n2 7\n3 69\n3 46\n3 61\n1 22\n2 6\n2 63\n2 48\n3 53\n1 91\n2 59\n2 39\n3 7\n3 14\n3 35\n2 16\n1 13\n3 69\n3 3\n2 56\n1 53\n3 31\n3 58\n2 51\n1 57\n2 20\n1 81\n1 74\n3 79\n3 41\n1 40\n2 40\n2 89\n3 71\n3 5\n1 36\n2 82\n3 13\n1 78\n1 45\n2 87\n1 17\n1 76\n2 54\n3 38\n3 92\n2 80\n2 79\n2 60\n3 63\n2 31\n2 25\n2 94\n1 78\n3 32\n1 93\n3 61\n3 32\n3 69\n2 21\n3 43\n3 67\n1 45\n2 72\n2 62\n1 90\n1 71\n3 11\n3 11\n2 90\n3 58\n3 20\n1 35\n3 49\n3 74\n1 15\n1 77\n3 85\n1 93\n1 33\n3 21\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'No\nNo\nNo\nNo\nNo\nNo\nNo\nNo\nNo\nNo\nNo\nNo\nNo\nNo\nYes\nNo\nNo\nNo\nYes\nNo\nNo\nNo\nNo\nNo\nNo\nNo\nNo\nNo\nYes\nNo\nNo\nNo\nNo\nNo\nYes\nNo\nNo\nNo\nYes\nNo\nNo\nNo\nYes\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='54 94\n2 13\n3 3\n2 8\n3 20\n1 14\n1 42\n2 6\n3 12\n3 32\n3 51\n3 28\n3 46\n3 1\n3 5\n3 7\n2 5\n1 41\n2 4\n3 52\n2 11\n2 36\n2 25\n3 12\n3 54\n3 27\n3 35\n1 22\n2 31\n2 50\n1 49\n2 32\n1 19\n1 21\n3 20\n2 20\n2 2\n3 22\n3 41\n1 14\n3 5\n3 27\n1 52\n3 28\n3 27\n3 4\n2 40\n1 29\n2 44\n3 3\n2 18\n3 14\n3 36\n2 41\n3 54\n2 1\n1 22\n3 20\n3 53\n1 17\n3 21\n3 29\n3 19\n3 43\n1 45\n3 39\n2 33\n1 54\n3 48\n1 15\n1 46\n1 45\n1 37\n3 1\n2 15\n3 29\n3 7\n1 9\n3 29\n2 17\n1 37\n1 54\n1 51\n3 21\n2 49\n2 53\n2 52\n3 27\n3 23\n3 15\n2 29\n2 7\n1 16\n3 23\n2 21\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'No\nNo\nNo\nNo\nNo\nNo\nNo\nNo\nNo\nNo\nNo\nNo\nNo\nNo\nNo\nNo\nNo\nNo\nYes\nNo\nNo\nNo\nYes\nNo\nYes\nYes\nNo\nYes\nNo\nNo\nNo\nNo\nNo\nNo\nNo\nYes\nNo\nNo\nNo\nNo\nNo\nNo\nYes\nNo\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
