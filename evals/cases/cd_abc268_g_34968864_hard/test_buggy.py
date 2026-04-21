import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\na\naa\nab\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n499122179\n499122179\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\na\naa\naaa\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n2\n3\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='26\nt\nl\nd\np\nv\ne\na\nc\nn\ny\nk\nm\nb\nq\nj\nh\no\nw\ns\ng\nu\nf\nz\nr\ni\nx\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '499122190\n499122190\n499122190\n499122190\n499122190\n499122190\n499122190\n499122190\n499122190\n499122190\n499122190\n499122190\n499122190\n499122190\n499122190\n499122190\n499122190\n499122190\n499122190\n499122190\n499122190\n499122190\n499122190\n499122190\n499122190\n499122190\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
