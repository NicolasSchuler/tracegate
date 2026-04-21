import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 100 50\n60 20 40\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '70\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 100 50\n60000 20000 40000\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '100\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 3 1\n1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='100 100000 1\n100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '100000\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='100 690 1\n688 988 2693 4227 6927 8090 8485 10838 10939 11029 11226 12112 14549 14703 15475 15629 17099 17865 19114 19353 20737 20753 21304 21362 22578 22712 23039 23422 25875 27379 27430 28315 28713 29114 30199 31589 33867 34379 35196 39095 39996 41502 41849 43599 44186 45325 48051 48722 49132 50752 51784 51883 52567 52914 53644 53782 53853 53890 54546 58754 58999 59077 60634 61184 61204 63555 65813 66197 68864 69832 71370 72230 72490 74134 74367 75202 75280 75579 75636 76283 77119 77772 78312 80042 80613 82303 82493 82608 83100 83161 84459 84901 88501 89776 89786 93573 94530 96497 97527 97718\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '689\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
