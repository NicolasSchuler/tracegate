import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='76\n1\n1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n11\n12\n13\n14\n15\n16\n17\n18\n19\n20\n100\n101\n102\n126\n127\n128\n10006\n10007\n10008\n100002\n100003\n100004\n1000002\n1000003\n1000004\n10000018\n10000019\n10000020\n100000006\n100000007\n100000008\n167772160\n167772161\n167772162\n469762048\n469762049\n469762050\n924844032\n924844033\n924844034\n998244352\n998244353\n998244354\n998244852\n998244853\n998244854\n1000000006\n1000000007\n1000000008\n1000000009\n1000000010\n1012924416\n1012924417\n1012924418\n1224736768\n1224736769\n1224736770\n2147483646\n2147483647\n2147483648\n2305843009213693950\n2305843009213693951\n2305843009213693952\n1\n1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '680\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5\n2\n3\n6\n12\n24\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '6\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='11\n1\n2\n3\n4\n5\n6\n123456789123456789\n123456789123456789\n987654321987654321\n987654321987654321\n121932631356500531347203169112635269\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '40\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='9\n4\n4\n4\n2\n2\n2\n1\n1\n1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '162\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1\n1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
