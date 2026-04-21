import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1\n5\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '5\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10\n-253230108 363756450 712662868 735867677 193944314 586260100 -192321079 95834122 802780784 418379342\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '-253230108 616986558 348906418 23204809 -541923363 392315786 -778581179 288155201 706946662 -384401442\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5\n-445130206 189801569 -354684803 -49687658 -204491568\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '-445130206 634931775 -544486372 304997145 -154803910\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10\n-840249197 853829789 470958158 -751917965 762048217 -92331626 649799919 -531035458 476202375 -92666141\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '-840249197 1694078986 -382871631 -1222876123 1513966182 -854379843 742131545 -1180835377 1007237833 -568868516\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10\n-472991684 -843699349 23704508 150490716 451258008 624171807 -829185026 -491315185 -76203556 -68926616\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '-472991684 -370707665 867403857 126786208 300767292 172913799 -1453356833 337869841 415111629 7276940\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
