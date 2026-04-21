import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\n3 3\n2 4\n6 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '3\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1\n1 4\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1\n1 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1\n1 1000000000\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '500000000\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='26\n4979445 861648772\n3690081 433933447\n6190629 262703497\n1047202 971407775\n8894325 731963982\n2804784 450968417\n302156 982631932\n1735902 880895728\n3078537 707723857\n9330739 910286918\n2329211 404539679\n3238506 317063340\n2686568 773361868\n5660016 650287940\n9296263 462224593\n2601449 384836991\n1890310 576823355\n2177068 404011431\n8008580 954291757\n449218 155374934\n594328 164163676\n7829355 138996221\n1899080 353195922\n5531545 910748511\n34067 913575467\n338674 824284691\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '4814609082\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
