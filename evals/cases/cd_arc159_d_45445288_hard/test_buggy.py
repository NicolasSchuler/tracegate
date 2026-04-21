import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4\n1 1\n2 4\n10 11\n7 10\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '8\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4\n1 1\n1 1\n1 1\n1 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1\n1 1000000000\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1000000000\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='22\n602436426 877914575\n24979445 861648772\n433933447 623690081\n262703497 476190629\n211047202 971407775\n628894325 731963982\n450968417 822804784\n430302156 982631932\n161735902 880895728\n707723857 923078537\n189330739 910286918\n404539679 802329211\n303238506 317063340\n492686568 773361868\n125660016 650287940\n462224593 839296263\n384836991 492601449\n191890310 576823355\n404011431 782177068\n818008580 954291757\n155374934 160449218\n164163676 840594328\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '957652488\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5\n138996221 501899080\n353195922 545531545\n350034067 910748511\n470338674 913575467\n533206504 824284691\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '774579247\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
