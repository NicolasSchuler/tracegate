import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 3\n##.\n#.#\n#..\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '499122178\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4 5\n..#..\n.###.\n#####\n..#..\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '598946613\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 4\n#...\n.#.#\n..##\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '285212675\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
