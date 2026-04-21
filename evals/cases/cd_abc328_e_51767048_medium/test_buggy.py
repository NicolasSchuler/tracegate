import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5 6 328\n1 2 99\n1 3 102\n2 3 86\n2 4 94\n2 5 95\n3 4 81\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '33\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='6 5 998244353\n1 2 337361568\n1 6 450343304\n2 3 61477244\n2 5 745383438\n4 5 727360840\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '325437688\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='8 28 936294041850197\n1 2 473294720906780\n1 3 743030800139244\n1 4 709363019414774\n1 5 383643612490312\n1 6 557102781022861\n1 7 623179288538138\n1 8 739618599410809\n2 3 857687812294404\n2 4 893923168139714\n2 5 581822471860662\n2 6 740549363586558\n2 7 307226438833222\n2 8 447399029952998\n3 4 636318083622768\n3 5 44548707643622\n3 6 307262781240755\n3 7 12070267388230\n3 8 700247263184082\n4 5 560567890325333\n4 6 704726113717147\n4 7 588263818615687\n4 8 549007536393172\n5 6 779230871080408\n5 7 825982583786498\n5 8 713928998174272\n6 7 751331074538826\n6 8 449873635430228\n7 8 11298381761479\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '11360716373\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='8 11 718762979161467\n3 4 594997858033122\n1 7 187500283015375\n4 6 394842061450734\n2 8 575928636264691\n1 5 106359254682938\n7 8 475884841730815\n5 8 7811391099246\n1 3 679805991335737\n1 4 677422058233417\n1 8 39926130825996\n2 7 388822457146319\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '35585030949467\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5 10 945532791347950\n3 4 681763130695725\n2 4 657183674581081\n4 5 854166010566295\n1 4 290679255273368\n3 5 462161904759586\n1 3 745538785941923\n1 5 431711083529922\n2 3 649183977369314\n1 2 286069786084481\n2 5 269440224275630\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1004918137\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
