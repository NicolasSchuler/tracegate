import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 3\nRDU\nLRU\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1 3\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 3\nRRD\nULL\n',
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
        input='9 44\nRRDDDDRRRDDDRRRRRRDDDRDDDDRDDRDDDDDDRRDRRRRR\nRRRDLRDRDLLLLRDRRLLLDDRDLLLRDDDLLLDRRLLLLLDD\nDRDLRLDRDLRDRLDRLRDDLDDLRDRLDRLDDRLRRLRRRDRR\nDDLRRDLDDLDDRLDDLDRDDRDDDDRLRRLRDDRRRLDRDRDD\nRDLRRDLRDLLLLRRDLRDRRDRRRDLRDDLLLLDDDLLLLRDR\nRDLLLLLRDLRDRLDDLDDRDRRDRLDRRRLDDDLDDDRDDLDR\nRDLRRDLDDLRDRLRDLDDDLDDRLDRDRDLDRDLDDLRRDLRR\nRDLDRRLDRLLLLDRDRLLLRDDLLLLLRDRLLLRRRRLLLDDR\nRRRRDRDDRRRDDRDDDRRRDRDRDRDRRRRRRDDDRDDDDRRR\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '9 5\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
