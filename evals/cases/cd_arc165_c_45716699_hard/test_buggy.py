import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 3\r\n1 2 5\r\n2 3 6\r\n1 3 12\r\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '11\r\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10 20\r\n7 10 982219000\r\n3 10 968366179\r\n2 4 992330437\r\n5 6 984414664\r\n2 8 897295423\r\n7 9 155604979\r\n6 8 958833005\r\n2 3 973209957\r\n3 7 985173062\r\n6 10 963895817\r\n2 10 986243534\r\n4 5 721724794\r\n1 3 657562445\r\n1 6 566370694\r\n1 4 988050146\r\n1 9 967817807\r\n4 9 796531581\r\n5 9 983960054\r\n1 10 964450079\r\n8 9 959369491\r\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '952136560\r\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10 20\r\n5 6 871895994\r\n8 10 873709822\r\n3 5 454175869\r\n6 10 980782191\r\n2 6 901290987\r\n1 8 298092290\r\n4 8 693116157\r\n4 5 947939338\r\n7 8 934395075\r\n7 9 759563833\r\n5 8 779870031\r\n4 6 919637355\r\n2 9 822858749\r\n4 10 855497285\r\n3 7 954942051\r\n1 2 950411658\r\n4 7 665939990\r\n3 4 634533617\r\n5 7 908372507\r\n1 9 591466693\r\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '759563833\r\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 2\r\n1 2 1\r\n2 3 1\r\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2\r\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 3\r\n1 2 1\r\n2 3 1\r\n1 3 1\r\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\r\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
