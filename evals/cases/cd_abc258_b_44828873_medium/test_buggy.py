import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4\n1161\n1119\n7111\n1811\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '9786\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10\n1111111111\n1111111111\n1111111111\n1111111111\n1111111111\n1111111111\n1111111111\n1111111111\n1111111111\n1111111111\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1111111111\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4\n3496\n4218\n7642\n8272\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '9872\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4\n8564\n6856\n7847\n8976\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '9885\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4\n5775\n8428\n7494\n7576\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '9855\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
