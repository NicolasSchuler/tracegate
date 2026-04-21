import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='((A)y)x\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'YAx\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='((XYZ)n(X(y)Z))\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'XYZNXYZ\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='dF(qT(plC())NnnfR(GsdccC))PO()KjsiI((ysA)eWW)ve\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'dFGsdccCrFNNnplCtQPOKjsiIwwEysAve\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='a\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'a\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='Z\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Z\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
