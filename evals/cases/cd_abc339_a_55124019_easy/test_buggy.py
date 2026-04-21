import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='atcoder.jp\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'jp\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='translate.google.com\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'com\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='.z\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'z\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='..........txt\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'txt\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='x..nc...vmknfza.gdr....yl..i....gz...p.....p.k..m...rou.....j.h..nofaf..q...sjb.s..m.k.w..h.n.u.a..b\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'b\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
