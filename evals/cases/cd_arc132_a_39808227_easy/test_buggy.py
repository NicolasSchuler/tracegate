import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5\n5 2 3 4 1\n4 2 3 1 5\n7\n1 5\n5 1\n1 1\n2 2\n3 3\n4 4\n5 5\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '#.#.#.#\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
