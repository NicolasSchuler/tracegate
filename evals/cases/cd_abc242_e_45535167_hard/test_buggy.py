import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2\n14\nDGALWVOOVWLAGC\n14\nDGALWVOOVWLAGD\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '998244352\n0\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5\n3\nAXA\n6\nABCZAZ\n30\nQWERTYUIOPASDFGHJKLZXCVBNMQWER\n28\nJVIISNEOXHSNEAAENSHXOENSIIVJ\n31\nKVOHEEMSOZZASHENDIGOJRTJVMVSDWW\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '24\n29\n212370247\n36523399\n231364016\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
