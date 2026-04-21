import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5 7\n1 2 (\n2 3 )\n3 4 (\n4 1 )\n2 4 )\n4 5 (\n5 1 )\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Yes\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 2\n1 2 )\n2 1 )\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'No\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10 20\n4 5 (\n5 6 (\n6 7 )\n2 5 )\n5 8 (\n6 3 )\n8 5 )\n1 2 (\n9 10 (\n4 7 (\n3 4 )\n8 9 (\n2 1 )\n1 4 )\n2 3 )\n3 2 (\n7 8 (\n7 4 )\n10 9 )\n9 8 )\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Yes\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 2\n1 2 (\n2 1 )\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Yes\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 2\n1 2 (\n2 1 (\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'No\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
