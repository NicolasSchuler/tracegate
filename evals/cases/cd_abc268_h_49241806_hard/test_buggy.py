import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='abcdefghijklmn\n3\nabcd\nijk\nghi\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='atcoderbeginnercontest\n1\nabc\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='aaaaaaaaa\n2\naa\nxyz\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '4\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='gxgpuamkxkszhkbpphykin\n10\nkszhkbpphykin\nph\ng\nxgp\nhykin\nuamk\nam\npua\nua\np\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '7\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='ldqfzaorahdigyojknviaztpcmxlvovafhjphvshyfiqqtqbxjjmqngqjhwkcexecmdkmzakbzrkjwqdyuxdvoossjoatryxmbwx\n46\nhyfiqqtqbx\ni\nkcexecmd\nmxlvovafhjphvshyfiqqtqbxjjmqngqjhwkcexecmdkmzakbzrkjwqdyuxdvoossjoatryxmb\nngqjhwkc\nwk\nknviazt\nngq\ndvo\ngyojknviaztpcmxlvovafhjphvshyfiqqtqbxjjmqngqjhwkcexecmdkmzakbzrkjwqdyu\njhwkcexecmdkmzakbzrkjwqdyuxdvoossjo\nq\npcmxlvovafhjphvshyfiqqtqbxjjmqngqjhwkcexecmdkm\ntqbxj\nztpcmxl\nz\nafhj\ntpcmx\na\nyojknviaztpcmxlvovafhjphvshyfiqqtqbxjjmqngqjhwkcexecmdkmzakbzrkjwqdyuxdvoossjoatryx\ngqjhwkc\nj\nv\nr\nvshyfiqqtqbxjjm\nafhjphvshyfiqqtqb\nzr\npc\nqfzaorahdigyojknviaztpcmxlvovafhjphvshyfiqqtqbxjjmqngqjhwkcexecmdkmzakbzrkjwqdyuxdvooss\njj\nfzaorahdigyojknv\nwq\nt\nviaztpcmxlvovafhjphvshyfiqqtq\ngqjhwkcexecmdkmzakbzrkjwqdyuxdvoossjoatryxmb\njphvshyfiqqtqbxjjmqngqjhw\njkn\nztp\nh\nbzrkjwqdyuxd\nqtqbxjjmqngqjhwk\nxecmdkm\nfhjphvshyfiqqtqbxjjmqngqj\ngqjhwkcexecmdkmzakbzrkjwqdyuxdvoossjoat\nhyfiqq\nx\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '51\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
