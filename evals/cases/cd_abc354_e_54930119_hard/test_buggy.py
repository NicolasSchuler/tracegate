import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5\n1 9\n2 5\n4 9\n1 4\n2 5\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Aoki\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='9\n3 2\n1 7\n4 1\n1 8\n5 2\n9 8\n2 1\n6 8\n5 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Takahashi\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='17\n238706200 663691869\n554212658 523459745\n267125099 882307940\n704871550 619293893\n67500253 470812606\n649674692 931356241\n964477579 304596225\n575103803 719272835\n952378980 28307637\n120501805 142601679\n153285804 37301463\n34164088 453747802\n387955323 887849048\n487995916 492605055\n492461116 357725084\n558415727 114678862\n993609774 982344956\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Aoki\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='14\n747122733 43871005\n881566225 718794427\n473805830 599931100\n134631002 513841859\n91846887 600514023\n865122164 524456697\n582002266 742965541\n238717290 362668952\n537568014 830506306\n268372397 123802844\n226929492 986272556\n756321170 681171335\n286988194 551925399\n581650127 478086494\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Aoki\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='9\n995703798 982987570\n343280190 21971427\n782606248 763583254\n564065275 949912355\n756370008 671339951\n364762986 56179960\n833175771 664856690\n506098185 114337939\n284422137 135006699\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Aoki\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
