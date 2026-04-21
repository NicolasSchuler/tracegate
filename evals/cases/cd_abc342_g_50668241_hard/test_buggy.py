import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='6\n2 7 1 8 2 8\n15\n3 1\n3 3\n3 4\n1 1 5 4\n3 1\n3 3\n3 4\n1 3 6 9\n3 1\n3 3\n3 4\n2 4\n3 1\n3 3\n3 4\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2\n1\n8\n4\n4\n8\n4\n9\n9\n2\n9\n9\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='24\n721 78 541 256 970 478 370 467 344 542 43 166 619 17 592 222 983 729 338 747 62 452 815 838\n35\n3 10\n3 8\n3 8\n3 13\n3 9\n1 1 17 251\n3 3\n3 19\n3 13\n3 22\n3 1\n3 15\n3 18\n3 10\n3 15\n1 16 19 883\n1 8 23 212\n3 5\n3 13\n2 6\n3 15\n1 5 18 914\n2 17\n3 20\n1 23 23 56\n3 13\n2 25\n3 13\n3 13\n3 10\n2 16\n1 17 22 308\n3 19\n3 17\n3 7\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '542\n467\n467\n619\n344\n541\n338\n619\n452\n721\n592\n729\n542\n592\n970\n619\n592\n747\n914\n914\n914\n914\n338\n983\n914\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1\n1\n3\n1 1 1 2\n3 1\n2 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
