import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='AtCoder\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'atcoder\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='SunTORY\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'SUNTORY\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
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


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='IfuMGTtPhefPxbfSCGGugsXqvhyHkQywmLjIefIAWupvjadLp\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'ifumgttphefpxbfscggugsxqvhyhkqywmljiefiawupvjadlp\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='AXYVJISyUzAItlLINKpzUBQEndvXgCFkujlnXNLdjKcSyOkkNwaWOqAxSRkjkEmcwmOONlLnuJtbpnNMSZRlbrZxulggtosNI\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'axyvjisyuzaitllinkpzubqendvxgcfkujlnxnldjkcsyokknwawoqaxsrkjkemcwmoonllnujtbpnnmszrlbrzxulggtosni\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
