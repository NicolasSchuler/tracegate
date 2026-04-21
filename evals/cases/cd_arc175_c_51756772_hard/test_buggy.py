import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4\n1 10\n8 13\n3 4\n5 20\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '8 8 4 5\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\n20 24\n3 24\n1 75\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '20 20 20\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='15\n335279264 849598327\n446755913 822889311\n526239859 548830120\n181424399 715477619\n342858071 625711486\n448565595 480845266\n467825612 647639160\n160714711 449656269\n336869678 545923679\n61020590 573085537\n626006012 816372580\n135599877 389312924\n511429216 547865075\n561330066 605997004\n539239436 921749002\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '526239859 526239859 526239859 467825612 467825612 467825612 467825612 449656269 449656269 449656269 626006012 389312924 511429216 561330066 561330066\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\n4 4\n0 2\n2 3\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '4 2 2\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='7\n2 5\n0 5\n3 3\n4 5\n2 3\n3 4\n2 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '3 3 3 4 3 3 2\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
