import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 19\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '5\n3 4\n4 8\n8 16\n16 18\n18 19\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='0 1024\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n0 1024\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3940649673945088 11549545024454656\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '8\n3940649673945088 3940649673949184\n3940649673949184 4503599627370496\n4503599627370496 9007199254740992\n9007199254740992 11258999068426240\n11258999068426240 11540474045136896\n11540474045136896 11549270138159104\n11549270138159104 11549545016066048\n11549545016066048 11549545024454656\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='288054454257713152 288230376160100352\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '4\n288054454257713152 288054454291267584\n288054454291267584 288089638663356416\n288089638663356416 288230376151711744\n288230376151711744 288230376160100352\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='432344644764301224 558448896716855296\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '25\n432344644764301224 432344644764301232\n432344644764301232 432344644764301248\n432344644764301248 432344644764301312\n432344644764301312 432344644764303360\n432344644764303360 432344644764827648\n432344644764827648 432344644769021952\n432344644769021952 432344644836130816\n432344644836130816 432344645104566272\n432344645104566272 432344653694500864\n432344653694500864 432344670874370048\n432344670874370048 432344739593846784\n432344739593846784 432345014471753728\n432345014471753728 432345564227567616\n432345564227567616 504403158265495552\n504403158265495552 540431955284459520\n540431955284459520 558446353793941504\n558446353793941504 558448552817197056\n558448552817197056 558448827695104000\n558448827695104000 558448896414580736\n558448896414580736 558448896683016192\n558448896683016192 558448896716570624\n558448896716570624 558448896716832768\n558448896716832768 558448896716849152\n558448896716849152 558448896716853248\n558448896716853248 558448896716855296\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
