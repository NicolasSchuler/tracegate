import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4 1 8\n2 21 17 21\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0 2 4 6 17 19 21 23\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4 3 7\n2 21 17 21\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '4 6 17 19 21\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5 1 1\n0 0 0 0 0\n',
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
        input='6 21 21\n88 44 82 110 121 80\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '41\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='12 26 48\n19629557415 14220078328 11340722069 30701452525 22333517481 720413777 11883028647 20926361028 24376768297 720413777 27999065315 13558621130\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '13558621130 14220078328 14586054825 15518998043 15970974282 16379590008 17091531049 17412316967 17836964726 18263536708 18965057557 19629557415 20282860278 20926361028 21302757781 21908867832 22333517481 22893781403 23595304394 23723463544 24376768297 24885524507 25261923402\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
