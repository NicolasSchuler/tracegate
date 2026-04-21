import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1\n54242 34392 23333 92971 5711 47765 32760\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '291174\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10\n90408 24492 41078 36756 68794 2060 62118 92121 61328 12685 18891 59613 23256 26016 46755 56694 97455 85238 49611 95092 85360 45143 87657 57882 92745 72852 37174 39825 60628 32289 18423 62500 41662 95309 61613 1645 45877 78395 38196 68335 4033 92215 7117 96173 52255 10360 81151 82449 33413 51239 18499 45762 95371 11402 94298 49783 23524 5427 32333 57680 36610 53516 81738 82352 90031 55180 37299 8445 65948 31009\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '325706 293910 516205 433278 372424 328696 362918 366354 290828 370264\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\n39721 95413 55007 11309 85980 9920 60790 77851 66816 73210 88421 93285 8471 52396 45509 32058 31791 193 9091 663 50762\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '358140 460450 170067\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10\n30132 27656 69600 53102 6028 59092 1916 8721 91940 22309 99142 95387 37573 91801 84557 40559 64695 78541 75541 74747 31094 11126 17471 73370 83826 31662 49037 70411 37413 30862 36716 25731 70319 1422 40126 36602 75935 33925 7813 52134 98268 5619 20252 61620 13814 32886 87993 89069 57469 4803 86226 71070 3299 64295 63620 22517 2393 45421 37109 76924 29753 23143 28693 90221 38880 35600 78685 75272 41893 41182\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '247526 446873 449734 336903 242589 310296 363103 315830 243436 401733\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10\n17566 19644 75494 86455 64445 37174 48945 92748 77787 54485 12867 68903 31475 4204 11949 91751 8881 835 1038 98140 70988 34729 9744 63031 91026 53816 46288 14154 61747 79345 71275 50723 9941 80488 80025 65854 10009 23085 66034 94858 23744 8189 17182 2090 85114 7390 48736 41520 75560 2296 89326 37862 89776 58924 26132 88998 26715 85487 22392 36584 82596 64031 47093 80116 17086 68663 28158 29044 12565 58762\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '349723 342469 283582 312788 433544 291773 277592 393314 364898 294394\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
