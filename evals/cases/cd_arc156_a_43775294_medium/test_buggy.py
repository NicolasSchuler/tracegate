import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5\n3\n101\n6\n101101\n5\n11111\n6\n000000\n30\n111011100110101100101000000111\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n2\n-1\n0\n8\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
