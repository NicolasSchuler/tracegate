import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\nabc\ndef\nghi\nabc\nbef\nghi\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2 1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1\nf\nq\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1 1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10\neixfumagit\nvtophbepfe\npxbfgsqcug\nugpugtsxzq\nbvfhxyehfk\nuqyfwtmglr\njaitenfqiq\nacwvufpfvv\njhaddglpva\naacxsyqvoj\neixfumagit\nvtophbepfe\npxbfgsqcug\nugpugtsxzq\nbvfhxyehok\nuqyfwtmglr\njaitenfqiq\nacwvufpfvv\njhaddglpva\naacxsyqvoj\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '5 9\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1\ns\nu\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1 1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='11\nymnwprdgfdl\nbpdfipwhnkr\noqzuxmfdhvy\njkjujjrxsjt\nehnzmlgsvnx\nixnxlynzhvs\njsrmklweftl\nvyvbqeppets\nlcbmsihdbij\nttxahirtvxh\nrsipwaphgsy\nymnwprdgfdl\nbpdfipwhnkr\noqzuxmfdhvy\njkjujjrxsjt\nehnzmlgsvnx\nixnxlynzhvs\ntsrmklweftl\nvyvbqeppets\nlcbmsihdbij\nttxahirtvxh\nrsipwaphgsy\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '7 1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
